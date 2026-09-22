# 第 5 章 ModelRunner 内部

> **本章回答**：`_prepare_inputs` 到底造了哪些张量、每步的 CPU 开销花在哪？`InputBatch` 为什么这样设计？`_bookkeeping_sync` 怎么做到"全步只同步一次"？异步调度在 worker 侧靠什么实现？
> 涉及文件：`vllm/v1/worker/{gpu_model_runner,gpu_input_batch,block_table,worker_base}.py`

## 5.1 为什么这个文件有 7700 行

`gpu_model_runner.py` 是 vLLM 里最大也最"不优雅"的文件，因为它同时承担了四件事：

1. **状态同步**：把调度器的意图落到 worker 侧的持久状态（`_update_states`）。
2. **张量构造**：把"算哪些 token"翻译成 GPU 输入（`_prepare_inputs`）。
3. **执行编排**：决定用哪种 CUDA Graph、要不要 pad、要不要 ubatch（`_determine_batch_execution_and_padding`）。
4. **结果回收**：把 GPU 结果翻译回 `ModelRunnerOutput`（`_bookkeeping_sync`）。

这四件事都以"每步执行一次"为前提优化。**读这个文件的正确方式是按 `record_function_or_nullcontext("...")` 的字符串搜索**，那些名字就是天然的目录。

## 5.2 `_update_states`：持久状态的协调器

`gpu_model_runner.py:1202`。docstring 里那句自白值得抄下来：

```python
# The persistent batch optimization assumes that consecutive batches contain
# mostly the same requests. If batches have low request overlap (e.g.,
# alternating between two distinct sets of requests), this optimization
# becomes very inefficient.
```

**这是"持久化 batch"优化的适用边界**。如果你的负载是"两批请求交替"，这个优化会变成负担。

### 它做的事

| 步骤 | 说明 |
| --- | --- |
| 移除完成的请求 | 从 `self.requests`（`dict[str, CachedRequestState]`）和 `InputBatch` 中删掉；通知 `late_interaction_runner` |
| 清零新 block | `_zero_block_ids(scheduler_output.new_block_ids_to_zero)` |
| 应用 CoW 拷贝 | `copy_kv_cache_blocks_inplace(...)`（KV sharing / 混合模型） |
| 计算未调度的请求 | `unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)`，把它们从 `InputBatch` 移除，**但保留 `CachedRequestState`** |
| 添加新请求 | 构造 `CachedRequestState(...)` 并 `input_batch.add_request(...)` |
| 返回延迟修正函数 | `deferred_state_corrections_fn` |

**"移除 `InputBatch` 的行但保留 `CachedRequestState`"** 是刻意设计：被抢占/暂时没排上的请求，它的 prompt token ids、采样参数等都在 `CachedRequestState` 里，**重新入队时不需要重新传一遍**。这省的是 IPC 带宽和构造开销。

### 返回值的用途

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
...
# 4520 行附近：模型 forward 已经发出之后
deferred_state_corrections_fn()
```

**这个"延迟修正"是异步调度的关键**：上一步的采样结果在这一步开始时可能还没回来，所以状态修正要等到"GPU 已经开始跑、我们确实需要这些状态之前"才做。

### `CachedRequestState`

`gpu_input_batch.py:35`：

```python
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    prompt_embeds: torch.Tensor | None
    prompt_is_token_ids: list[bool] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    generator: torch.Generator | None
    block_ids: tuple[list[int], ...]          # ★ 每个 KV 组一份
    num_computed_tokens: int
    output_token_ids: list[int]
    lora_request: LoRARequest | None
    ...
```

注意 `block_ids` 是**元组**（每 KV 组一个列表），这与 `MultiGroupBlockTable` 对应。
`generator` 只在 `SamplingType.RANDOM_SEED` 时创建（见第 8 章）。

## 5.3 `_prepare_inputs`：逐步拆解

`gpu_model_runner.py:1975`。返回值是 `(logits_indices, spec_decode_metadata, max_num_sampled_tokens)`。

### 第 0 步（最重要的顺序优化）

```python
# 1993
# OPTIMIZATION: Start copying the block table first. This way, we can
# overlap the copy with the following CPU operations.
self.input_batch.block_table.commit_block_table(num_reqs)
```

`commit_block_table`（`block_table.py:231`）= `self.block_table.copy_to_gpu(num_reqs)`，是一次 **H2D 拷贝**。
把它放最前面，后面几十行 numpy 工作就能和这次拷贝**重叠**。

> 💡 **性能视角**：这是"**用 CPU 计算掩盖 H2D 拷贝**"的经典手法。
> 更普遍的推广是 **CUDA Stream 上的多流重叠**，但那个更复杂（要处理依赖）。这里用最简单的"顺序调整"就拿到了收益。

### 张量构造链

```python
# 1998  req_indices：把"每请求几个 token"展开成"每个 token 属于哪个请求"
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

# 2005  query_pos：每个 token 在它自己请求内的偏移；同时得到前缀和
cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)

# 2010  positions：纯 numpy 加法
positions_np = num_computed_tokens_cpu[req_indices] + query_pos

# 2026-2048  input_ids：index_select 抓取
token_indices = positions_np + req_indices * token_ids_cpu.shape[1]
torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(), 0,
                   token_indices_tensor, out=self.input_ids.cpu[:total])

# 2093-2100  query_start_loc：[0, *cu_num_tokens]，尾部填 cu_num_tokens[-1]
# 2105-2111  optimistic_seq_lens_cpu
# 2116       _compute_prev_positions：当前行 → 上一步行的映射（新行 -1）
# 2122-2127  discard_request_mask（chunked prefill 行要丢弃采样结果）
# 2169-2196  num_computed_tokens（异步模式下用自定义 kernel 在 GPU 上修正）
# 2201       positions（GPU 版）
# 2210       ★ block_table.compute_slot_mapping(...)
# 2214       _prepare_input_ids
# 2217-2247  M-RoPE 位置拷贝（逐行！）
# 2270-2305  logits_indices
# 2308       set_active_loras
```

### 六个值得展开的细节

**① `positions` 在 CPU 上用 numpy 算**

```python
positions_np = num_computed_tokens_cpu[req_indices] + query_pos
```

调度器本来就知道 `num_computed_tokens`，直接在 CPU 上加即可。用 GPU 算还要先把输入传上去，多一次 H2D。**"能在 CPU 算的就在 CPU 算，只要不需要 GPU 的并行度"** —— 但要注意这个结论只在"数据量小、且本来就在 CPU 上"时成立。

**② `torch.index_select` 优于 `np.take`**

代码注释里明说了这一点，并且用 `out=` 写进预分配的 pinned buffer。**热路径的分配次数是零**。

**③ `query_start_loc` 尾部必须单调不减**

```python
# 尾部填 cu_num_tokens[-1]
```

因为 FlashAttention 的 kernel 假设它单调。如果 pad 位置填 0，kernel 会算出负长度。**这是"下游接口约定倒逼上游数据结构"的例子。**

**④ `discard_request_mask`：采样了但不要**

```python
# rows whose optimistic length is still below num_tokens are chunked-prefill
# rows whose sampled token must be thrown away. Sampling still happens
# (SIMD uniformity) — the mask is applied later in _bookkeeping_sync.
```

chunked prefill 的中间块，其"最后一个 token"不是真正的输出（用户要的是 prompt 全部算完后才出第一个 token）。但**采样 kernel 仍然会为这些行算一遍**（因为要保证所有行的形状一致，避免分支）。

处理方式：在 `_bookkeeping_sync` 里按 mask 丢弃，并且**把随机数发生器的 offset 回退**（`gen.set_offset(gen.get_offset() - 4)`），保证随机流不被"虚耗"的采样扰动 —— **这是可复现性的保障**。

**⑤ M-RoPE 位置必须逐行拷贝**

```python
# 2217-2247
# 注释解释：mrope_positions.gpu[row, :N] 从 [3, max+1] 的分配里切出来是
# strided 的，一次 copy_() 会退化成走 pageable staging buffer，
# 让 non_blocking=True 悄悄变成同步。
```

**这是本文件最有教育意义的细节之一**：一个看起来"只是写法不同"的改动（一次 `copy_` vs 逐行 `copy_`），会**静默地让异步拷贝退化成同步**，代价是几十微秒的 GPU 等待。
为了保住非阻塞语义，宁愿多做几次小拷贝。

**⑥ `compute_logits` 只算需要采样的行**

```python
# 2270-2305
# 无投机解码：logits_indices = query_start_loc[1:] - 1
# 有投机解码：由 _calc_spec_decode_metadata 产生 target/bonus/logits 三组索引
```

然后：

```python
# 4520
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states, ...)
```

**`[vocab, hidden]` 的 GEMM 成本 ∝ 采样 token 数**。vocab 动辄 10 万+，这是整步最贵的算子之一。chunked prefill 下大多数行是 prompt token，不需要采样 —— 省掉的是数量级的成本。

## 5.4 `compute_slot_mapping`：一个 AOT 编译的 Triton kernel

`vllm/v1/worker/block_table.py:397` `ComputeSlotMappingKernel`。

```python
class ComputeSlotMappingKernel(VllmJitKernel["ComputeSlotMappingKernel.CompileKey"]):
    @dataclass
    class CompileKey:
        kv_cache_block_size: int
        blocks_per_kv_block: int
        total_cp_world_size: int
        total_cp_rank: int
        cp_kv_cache_interleave_size: int
        block_table_stride: int
        block_size: int

    @triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
    def kernel(...): ...
```

### 三个设计要点

**① `do_not_specialize=["num_tokens", "max_num_tokens"]`**

Triton 默认为每个标量参数值重新编译。batch 大小每步都变 —— 不声明这两个参数"不做特化"，就会每步走 JIT 慢路径。**这是 Triton 性能调优的第一课。**

**② grid = `num_reqs + 1`，最后那个 program 专门写 padding**

```text
每个真实 program：遍历自己请求的 token 范围（1024 一块），算出 slot_ids
最后一个 program：把 [num_tokens, max_num_tokens) 填成 PAD_ID
```

这个技巧很优雅：**把"清理 padding"变成一次 kernel 内的固定工作量**，不需要额外的 kernel 或 CPU 循环。

**③ 用 `dispatch` 做标量特化**

```python
def dispatch(...):
    # triton_scalar_specialization_rep on block_table_stride and block_size
```

**特化在"每步不变的量"上（block_size、stride），而不是"每步都变的量"上（token 数）**。这是 Triton 特化的正确用法。

### slot 的语义

```text
slot_ids = block_numbers * block_size + slot_offsets
```

即"扁平偏移到分页 KV 池"。padding 位置填 `-1`，消费端（`reshape_and_cache`）遇到负数就跳过。

`SlotMappingMode`（`block_table.py:52`）：
- `TOKEN_TO_KV_SLOT`：标准情况；
- `NONE`：Mamba/GDN 组 —— 它们的"块表"是 recurrent state 索引，**没有 per-token 槽位**。

### `map_to_kernel_blocks`：两级块大小的翻译

```python
# block_table.py:239
@staticmethod
def map_to_kernel_blocks(kv_manager_block_ids, blocks_per_kv_block, kernel_block_arange):
    """Example:
        # kv_manager_block_ids: 32 tokens, Kernel block size: 16 tokens
        # blocks_per_kv_block = 2
        >>> kv_manager_block_ids = np.array([0, 1, 2])
        >>> Result: [0, 1, 2, 3, 4, 5]"""
    if blocks_per_kv_block == 1:
        return kv_manager_block_ids
    kernel_block_ids = (kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
                        + kernel_block_arange)
    return kernel_block_ids.reshape(-1)
```

**为什么要两级块大小？** KV 管理器按统一页大小分配（为了混合模型），而 attention kernel 有自己的 block size 要求（有的要 32/64/128）。两级映射让**管理器不需要关心 kernel 的偏好**。

## 5.5 `InputBatch`：预分配的艺术

`gpu_input_batch.py:92`。

### 预分配清单

| 类别 | 具体字段 | 注意 |
| --- | --- | --- |
| token | `token_ids_cpu_tensor` `(max_num_reqs, max_model_len)` | **不 pin**（注释：从不直接拷到 GPU，pin 是浪费） |
| | `is_token_ids_tensor` | bool，标记哪些位置是真的 token（vs 图像占位） |
| 计数 | `num_tokens_no_spec_cpu_tensor`、`num_prompt_tokens_cpu_tensor`、`num_computed_tokens_cpu_tensor` | **pin + `.numpy()` 视图双份** |
| 采样 | `temperature` / `top_p` / `top_k` | 每个都有 pinned CPU 孪生 |
| 惩罚 | frequency / presence / repetition | 同上 |
| 成员集合 | `greedy_reqs`、`random_reqs`、`top_p_reqs`、`top_k_reqs`、`*_penalties_reqs`、`has_allowed_token_ids` | **fast path 的来源** |
| block table | `MultiGroupBlockTable` | 每 KV 组一张 |
| 投机 | `spec_token_ids: list[list[int]]` | |
| 异步 | `prev_sampled_token_ids`、`prev_req_id_to_index`、`sampled_token_ids_cpu`、`async_copy_ready_event` | |

**"pin + numpy 视图双份"是什么操作？**

```python
self.num_computed_tokens_cpu_tensor = torch.empty(max_num_reqs, dtype=torch.int32, pin_memory=True)
self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()
```

同一块内存，既可以用 torch API 访问，也可以用 numpy API 访问（`_prepare_inputs` 里大量用 numpy）。**避免每步在两种表示之间转换。**

### 成员集合：把 O(batch) 判断变成 O(1)

```python
# 1117
@property
def all_greedy(self) -> bool: ...
# 1121
@property
def all_random(self) -> bool: ...
# 1125 / 1129
def no_top_p / no_top_k(self) -> bool: ...
# 1133
@property
def no_penalties(self) -> bool:
    return (len(self.frequency_penalties_reqs) == 0
            and len(self.presence_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0)
# 1148 / 1152
def max_num_logprobs / no_allowed_token_ids(self): ...
```

**这是 `InputBatch` 最重要的设计思想**：与其在采样时逐请求检查参数，不如在 `add_request` 时把请求登记到对应的集合里，采样时只看集合空不空。

维护发生在 `add_request`（`:350`）与 `remove_request`（`:528`）：

```python
if sampling_params.temperature < _SAMPLING_EPS:
    self.greedy_reqs.add(req_id)
else:
    self.random_reqs.add(req_id)
if sampling_params.frequency_penalty != 0.0:
    self.frequency_penalties_reqs.add(req_id)
...
```

> 💡 **性能视角**：采样路径上的分支是**每步每 batch 一次**的。`no_penalties=True` 时整个惩罚项的 gather/scatter 都不执行 —— 这可能是 `[num_reqs, vocab]` 级别的操作。
> **推论**：让 batch 里的请求共享采样参数（例如全部 greedy），能让整批走最快的路径。这是"参数一致性"带来的隐性性能收益。

### `refresh_metadata` 与 `_make_sampling_metadata`

```python
# 838
def refresh_metadata(self):
    """Called only when the batch composition changed."""
    ...
# 858
def _make_sampling_metadata(self) -> SamplingMetadata:
```

**只在 batch 组成变化时才重建 `SamplingMetadata`**。如果连续几步的 batch 成员不变（高并发稳态下很常见），这个开销就是零。

### `condense` 与 `swap_states`

```python
# 706
def condense(self) -> None: ...
# 584
def swap_states(self, i1: int, i2: int) -> None: ...
```

`condense` 把非空行向上滑，`batch_update_builder` 记录了哪些行被删，**没有删除时 early-return**（`:711-715`）。

`swap_states` 的注释里有关键优化：

```python
# 只拷贝活跃 token 前缀（_get_active_token_count = num_tokens_no_spec + len(spec_token_ids)）
# 而不是整行 max_model_len
```

以及它会把移动记录进 `batch_update_builder.moved`，供 logits processor 的状态跟随移动。

### `_get_active_token_count`：为什么要限制拷贝范围

```python
# 701
def _get_active_token_count(self, req_index: int) -> int:
```

`token_ids_cpu_tensor` 是 `(max_num_reqs, max_model_len)` —— 一行 32K 个 int32 = 128 KiB。交换两行如果全量拷贝就是 256 KiB 的 memcpy。**只拷"已经用到的部分"能把这降到几百字节。**

## 5.6 行序与 GPU 张量的对齐：`prev_positions`

`condense` / `swap_states` 会改变行序，但**上一步在 GPU 上的张量（采样结果、logits 等）还是旧顺序**。

`_compute_prev_positions`（`gpu_model_runner.py:1779`）建立"当前行 → 上一步行"的映射：

```python
# 新行填 -1（没有对应的上一步数据）
```

然后：

```python
# _prepare_input_ids 的异步路径
# 用 prev_positions 把 prev_sampled_token_ids scatter 到正确位置
```

**这是一个"用一次 gather/scatter 代替重排整个状态"的技巧**：与其把所有 CPU 张量按新行序重排（O(batch × state)），不如在需要的地方按映射取（O(需要的元素数)）。

## 5.7 `_bookkeeping_sync`：全步唯一的同步点

`gpu_model_runner.py:3751`。它被设计成"只同步一次"，具体做法：

### 同步路径

```python
discard_sampled_tokens_req_indices = np.nonzero(discard_request_mask.np[:num_reqs])[0]
for req_index in discard_sampled_tokens_req_indices:
    gen = self.input_batch.generators.get(int(req_index))
    if gen is not None:
        gen.set_offset(gen.get_offset() - 4)      # ★ RNG 回退
...
# 一个 gpu_sync_allowed() 块包住所有需要同步的操作
with self.synchronize_input_prep():
    ...
    if max_gen_len == 1:
        sampled_token_ids_list = self._to_list(sampled_token_ids)      # 直接 D2H
    else:
        sampled_token_ids_list = self.rejection_sampler.parse_output(...)  # 拆分 accepted/bonus
```

**关键设计**：

1. **一个同步块覆盖所有**。多个 `.cpu()` / `.tolist()` 之间如果不隔别的 GPU 操作，只付一次同步代价。
2. **routed experts 的 D2H 在 `_to_list` 之前发出**（`3795`），因为 `_to_list` 的 `event.synchronize()` 会顺带覆盖它。**把独立的小拷贝塞进已有的同步窗口。**
3. **RNG 回退**：chunked prefill 行被丢弃的采样会消耗随机数，必须回退 offset，否则**同样的请求在不同 batch 组成下会产出不同结果**（可复现性破坏）。

### 异步路径（`:3820-3840`）

```python
valid_sampled_token_ids = []                # 空！
invalid_req_indices = [...]
# 采样结果留在 GPU 上
self.input_batch.set_async_sampled_token_ids(...)
```

**完全不 D2H、完全不 `synchronize`。** 下一步的 `_prepare_input_ids` 在 GPU 上 scatter 使用。

> 💡 **性能视角**：一次 `synchronize` 在 decode 步的占比可能到 5~15%（取决于 batch 大小）。
> 异步路径把这个成本归零，代价是**状态推理的复杂度**（每一步都要考虑"上一步的结果还没回来"）。
> 相关字段：`num_output_placeholders`、`num_in_flight_tokens`、`num_stale_output_tokens`、`drop_stale_output`。

## 5.8 `sample_tokens` 与 drafter 的三种时序

`sample_tokens`（`gpu_model_runner.py:4628`）：

```python
# 4661  取出并清空 execute_model_state（强制 execute/sample 配对）
# 4667  apply_grammar_bitmask(...)  ← 结构化输出
# 4676  _update_states_after_model_execute(...)
# 4686  PP + 异步：_pp_broadcast_prev_sampled_token_ids
# 4700-4795  drafter 分派（三种 regime）
# 4820  finalize_kv_connector()  ← 放在 drafter 之后
# 4824  eplb_step()
```

### drafter 的三种时序

| Regime | 条件 | 采样结果从哪来 |
| --- | --- | --- |
| **模型型 drafter**（EAGLE / draft model） | `drafter_runs_model_forward` | **直接用 GPU 上的采样结果**，不等 bookkeeping —— 最快的路径 |
| **GPU n-gram** | `spec_config.use_ngram_gpu()` | `NgramProposerGPU.update_token_ids_ngram(...)` |
| **CPU 型 drafter**（ngram / suffix） | `draft_after_bookkeeping = True` | 必须等 `_bookkeeping_sync` 之后，因为需要 CPU 上的 token id |

**分派依据是"drafter 需要的数据在 CPU 还是 GPU 上"**。这个分类很值得记住：它解释了为什么不同的投机解码方法在性能上差异明显。

### `finalize_kv_connector` 为什么在 drafter 之后

因为**草稿模型也可能需要保存 KV**。顺序错了会丢掉草稿的 KV。

### `input_fits_in_drafter` 与"草案清零"

```python
# 4785 附近
# 当批次放不进 drafter 时，草案张量要被清零
```

注释解释了原因：**残留的旧草案会污染 Mamba 的 recurrent state，以及在接近 `max_model_len` 时污染 logprobs**。这类"看起来无害但会静默出错"的地方，是框架代码里最需要注意的一类。

## 5.9 `_determine_batch_execution_and_padding`

`gpu_model_runner.py:4015`。这里把"运行时 batch"变成"CUDA Graph key"。

```python
uniform_decode = self._is_uniform_decode(max_num_scheduled_tokens,
                                        uniform_decode_query_len, num_tokens, num_reqs)
num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)   # SP 要求 %tp==0
...
# DP 协调：所有 rank 必须对 batch 大小达成一致
if self.parallel_config.data_parallel_size > 1:
    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = \
        coordinate_batch_across_dp(...)
    num_tokens_padded = ...   # 用 DP 商定的值替换
    cudagraph_mode, batch_desc = dispatch_cudagraph(num_tokens_padded,
                                                   valid_modes={CUDAGraphMode(synced_cudagraph_mode)})
    assert batch_desc.num_tokens == num_tokens_padded
```

`_is_uniform_decode`（`:3950`）的定义值得记住：

```python
max_num_scheduled_tokens == uniform_decode_query_len
    and num_tokens == max_num_scheduled_tokens * num_reqs
```

即**每个请求的 query 长度都相同**。`uniform_decode_query_len = 1 + num_speculative_tokens`。

**为什么 DP 必须商定 batch 大小？** 因为 MoE 的 all-to-all 是**集合通信**，如果各 rank 的 token 数不同，通信会挂死。所以要先 `coordinate_batch_across_dp` 取一个公共值（通常是最大值），再把小的 pad 上去。

`CUDAGraphStat`（`:4119`）记录 `num_unpadded_tokens` / `num_padded_tokens` / `num_paddings` —— **padding 的开销是被测量的，不是被假设的**。

## 5.10 `_build_attention_metadata` 的复用策略

`gpu_model_runner.py:2318`。三个层次的复用：

**① 按 KV 组而非按层**

```python
for layer_name in attn_group.layer_names:
    attn_metadata[layer_name] = cm
```

32 层单组模型 → 只 build 一次。

**② `supports_update_block_table` 时只换 block table**

```python
if cached_metadata is not None and builder.supports_update_block_table:
    builder.update_block_table(cached_metadata, block_table_tensor, slot_mapping)
else:
    metadata = builder.build(common_prefix_len=..., common_attn_metadata=..., ...)
```

**③ 跨组的浅拷贝**

```python
# 2582-2650
cm_base = ...  # 基础 metadata
for group in groups:
    cm = copy(cm_base)          # 浅拷贝
    cm.encoder_seq_lens = ...
    cm.block_table_tensor = ...
    cm.slot_mapping = ...
```

多个组的 metadata 几乎一样，只差三个字段 —— 浅拷贝 + 覆盖比重新 build 便宜得多。

### padding 行的处理

```python
# 2350 附近
def _get_block_table(gid):
    tbl = blk_table.get_device_tensor(num_reqs_padded)
    # 尾部填 NULL_BLOCK_ID（块 0 被保留给 padding）
```

**CUDA Graph 里所有形状固定，pad 出的请求行如果携带垃圾 block 号，就会读到非法显存。**

### `max_seq_len` 的计算

```python
if for_cudagraph_capture:
    max_seq_len = self.max_model_len        # 捕获时必须用最大长度
else:
    max_seq_len = optimistic_seq_lens_cpu[:num_reqs].max()   # CPU 上算，无同步
```

**捕获时用最大值的原因**：滑动窗口模型在不同 `max_seq_len` 下会走不同的 kernel 变体。如果捕获时用了较小的值，replay 时可能拿到错误的 kernel。

## 5.11 本章小结

| 机制 | 省了什么 | 代价 |
| --- | --- | --- |
| 先 commit block table | 用 CPU 计算掩盖 H2D | 需要调整代码顺序 |
| `index_select` + `out=` | 避免分配 | — |
| `positions` 在 CPU 算 | 避免 H2D | 只在数据本就在 CPU 上时成立 |
| `logits_indices` | 数量级的 GEMM 成本 | 索引构造的复杂度 |
| `do_not_specialize` | 避免每步 JIT | — |
| grid=num_reqs+1 写 padding | 无需额外 kernel | — |
| `InputBatch` 预分配 | 每步零分配 | 启动时占满 `max_num_reqs × max_model_len` |
| 成员集合 | O(1) 判断 + 整张量 fast path | `add/remove` 时的维护 |
| `prev_positions` | 避免重排整个状态 | 映射表的维护 |
| 单次同步 + 异步路径 | 消除同步停顿 | 状态推理复杂度 |
| metadata 复用 | 避免每步重建 | 需要 `supports_update_block_table` |

⚠️ **易错点**
- 逐行 vs 一次性拷贝会影响 `non_blocking` 是否真正生效 —— 这是本文件里最隐蔽的一类性能陷阱。
- `_update_states` 的 docstring 明说了持久化 batch 的适用边界（相邻 batch 高度重合）。
- `use_v2_model_runner` 时走 `vllm/v1/worker/gpu/` 下的另一套实现，与本文件并行存在。

📖 官方文档：`docs/design/model_runner_v2.md`

下一章 → [第 6 章 CUDA Graph 与 torch.compile](06-CUDA-Graph与torch-compile.md)
