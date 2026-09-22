# 第 6 章 CUDA Graph 与 torch.compile

> **本章回答**：CUDA Graph 到底消除了什么开销？为什么需要"分段（piecewise）"？`CudagraphDispatcher` 怎么决定这一步用哪张图？torch.compile 的融合 pass 在优化什么？AOT 编译缓存是怎么工作的？
> 涉及文件：`vllm/v1/cudagraph_dispatcher.py`、`vllm/compilation/{cuda_graph,backends,piecewise_backend,decorators,partition_rules,breakable_cudagraph}.py`、`vllm/compilation/passes/`

## 6.1 先量化收益：CUDA Graph 消除的是什么

一次 kernel launch 的 CPU 开销约 **3~10 μs**（含 Python → C++ → CUDA driver 的路径）。一个 decode step 可能有 **几百到上千个 kernel**（每层：QKV、RoPE、RMSNorm、attention、O projection、MLP 三个、all-reduce……乘以层数）。

```text
40 层模型，每层 ~10 个 kernel = 400 个 kernel
400 × 5 μs = 2 ms  ← 纯 CPU launch 开销
```

而一个 decode step 的**实际 GPU 计算时间**可能只有 3~5 ms（小 batch 时更少）。

**结论：CPU launch 开销可以轻松占到一步的 30%~50%。** CUDA Graph 把整串 launch 录制成一张图，replay 时只有一次 launch → 这部分开销几乎归零。

> 💡 **性能视角**：这解释了两件事：
> 1. 为什么 **batch 越小，CUDA Graph 收益越大**（计算时间缩短了，但 launch 数量不变）；
> 2. 为什么 `--enforce-eager` 会让性能腰斩 —— 它不是"少了一个优化"，而是"多付了一笔固定税"。

## 6.2 为什么需要"分段"：attention 是拦路虎

CUDA Graph 要求**所有张量地址和形状在 replay 时不变**。但 attention 的输入形状随 batch 组成而变（不同请求的 KV 长度不同），而且很多 attention kernel 本身不支持在 graph capture 期间运行。

于是有了 **piecewise（分段）编译**的思路：

```text
原本：  [整个模型的一串 kernel]  ← 中间夹着 attention，无法整图捕获

分段后： [第0层前半段] → [ATTENTION(eager)] → [第0层后半段] → [第1层前半段] → ...
         └── 可捕获 ──┘   └─── 不捕获 ───┘   └── 可捕获 ──┘
```

`CompilationConfig.splitting_ops` 就是要在此处"切开"的算子列表。默认值来自 `_attention_ops`（`vllm/config/compilation.py:765` 附近）：

```python
_attention_ops: ClassVar[list[str]] = [
    "vllm.unified_attention", "vllm.unified_attention_with_output",
    ...
]
```

**`splitting_ops=[]` 表示"不切图"（整图编译）**，此时 attention 也进图 —— 只有支持 CUDA Graph 的后端能做到，但换来的是"attention 融合"和"序列并行"这两个需要看全图的 pass 才能生效。

## 6.3 `CUDAGraphMode`

`vllm/config/compilation.py`：

| 模式 | 含义 | 需要什么 |
| --- | --- | --- |
| `NONE` | 关图 | — |
| `PIECEWISE` | 分段捕获，attention 保持 eager | piecewise 编译 |
| `FULL` | 整图捕获（含 attention） | attention 后端支持 |
| `FULL_DECODE_ONLY` | 只有 uniform decode 走 full 图 | 后端支持 decode-only |
| `FULL_AND_PIECEWISE`（默认） | decode 走 full，其他走 piecewise | 两者都要 |

**双模式配置（后两个）需要运行时在两者之间分派** —— 这就是 `CudagraphDispatcher` 的职责。

### 自动降级

不是所有后端都支持所有模式。vLLM 会**按 `AttentionCGSupport` 自动降级**（`GPUModelRunner._check_and_update_cudagraph_mode`）：

```text
要求 FULL，但后端能力是 UNIFORM_BATCH
    → 有 piecewise 编译：降为 FULL_AND_PIECEWISE
    → 没有：降为 FULL_DECODE_ONLY
要求 FULL，但后端能力是 NEVER
    → 降为 PIECEWISE
```

后端能力表见入门教程 6.4 或 `docs/design/cuda_graphs.md`。

**混合模型（如 Mamba + attention）取所有后端的 min**：

```python
# AttentionCGSupport 的枚举值是排序的（ALWAYS=3 > UNIFORM_BATCH=2 > ...）
# 所以 min() 就是整体能力
```

## 6.4 `CudagraphDispatcher`：唯一的事实来源

`vllm/v1/cudagraph_dispatcher.py:15`。类的 docstring 明确了契约：

> 它持有两套 dispatch key（PIECEWISE 和 FULL），是"运行时可以分派哪些 CUDA Graph"的**唯一事实来源**；wrapper 无条件相信 forward context。

**这个"唯一事实来源"的设计消除了整类 bug**：以前 wrapper 自己维护 cache，容易和 dispatcher 的状态不一致。

### `_compute_bs_to_padded_graph_size`：一张查表

```python
# :72
def _compute_bs_to_padded_graph_size(self) -> None:
    # 建一个长度 max_cudagraph_capture_size + 1 的列表
    # 对每个捕获区间 [start, end): bs == start → start，否则 → end（向上取整）
```

即：**给定实际 batch 大小，向上取到最近的"捕获过的 size"**。
同时它会**校验 `compile_sizes` 里的值不会被 padding 改变**，否则报错并提示"请使用 `cudagraph_capture_sizes` 里的值"。**这是启动期 fail-fast 的好例子** —— 与其等到运行时静默地用一个未编译的尺寸，不如启动就报错。

> 💡 **性能视角**：`cudagraph_capture_sizes` 的选择是一个显式权衡：
> - 捕获的尺寸越多，覆盖越细（padding 浪费越少），但**捕获耗时和显存占用都线性增长**；
> - 默认策略是"小尺寸密集、大尺寸稀疏"（比如 1,2,4,8,...,512 加一些中间值）。
> - 启动日志里的 `Capturing CUDA graphs` 会列出所有尺寸。**如果你的实际 batch 分布与捕获尺寸不匹配，会看到大量 padding。**

### `initialize_cudagraph_keys`

```python
# :166
for size in cudagraph_capture_sizes:
    for lora_case in lora_cases:
        # PIECEWISE：放宽描述符（num_reqs=None, uniform=False）
        # FULL：只有 uniform decode 且 size <= uniform_decode_query_len * max_num_seqs
```

两个关键点：

1. **PIECEWISE 的 key 被放宽**（`num_reqs=None, uniform=False`）—— 因为分段图不含 attention，形状约束只来自 token 数，与请求数无关。**一张 piecewise 图能服务很多种请求数组合。**
2. **FULL 必须精确的 `num_reqs`** —— 因为 FA3 等后端的 scheduler metadata 计算依赖它（代码注释）。

### `dispatch`：优先级搜索

```python
# :235
def dispatch(self, num_tokens, uniform_decode, has_lora, num_active_loras,
             valid_modes=None, invalid_modes=None):
    # 早退：key 未初始化 / mode==NONE / num_tokens > max_size → (NONE, desc)
    ...
    # 优先级：FULL 精确 key → PIECEWISE 放宽 key → NONE
```

**搜索顺序是 `FULL > PIECEWISE > NONE`** —— 优先用"覆盖更完整"的图。

LoRA 的处理也值得一提：
- 开启 `cudagraph_specialize_lora` 时，用 `bisect_left` 在捕获的 LoRA 数量列表里**向上取整**（3 个 LoRA 复用 4 个 LoRA 的图）；
- 否则强制用 `max_loras + 1`（一个"通配"图）。

`invalid_modes` 用来排除不可用的模式：

```python
# cascade attention / encoder 输出 → invalid_modes={FULL}
```

## 6.5 `CUDAGraphWrapper`：嵌套设计的核心

`vllm/compilation/cuda_graph.py:145`。每个 wrapper 绑定一个 `runtime_mode`（只能是 `PIECEWISE` 或 `FULL`）。

### `__call__` 的四步逻辑

```python
def __call__(self, *args, **kwargs):
    # ① 没有 forward context → 直接调用（例如 ViT encoder 路径）
    if forward_context is None:
        return self.runnable(*args, **kwargs)
    # ② runtime_mode 不匹配 → 直接调用   ★ 这就是嵌套能工作的原因
    if runtime_mode == CUDAGraphMode.NONE or runtime_mode != self.runtime_mode:
        return self.runnable(*args, **kwargs)
    # ③ 查/建 entry
    entry = self.concrete_cudagraph_entries.get(batch_descriptor)
    if entry is None:
        entry = CUDAGraphEntry(batch_descriptor=batch_descriptor, ...)
        self.concrete_cudagraph_entries[batch_descriptor] = entry
    # ④ 捕获或 replay
    if entry.cudagraph is None:
        return self._capture(...)
    return self._replay(...)
```

### 嵌套布局

```text
┌─────────────────────────────────────────────────────┐
│  CUDAGraphWrapper(runtime_mode=FULL)   ← 外包整个模型 │
│  ┌───────────────────────────────────────────────┐  │
│  │  piecewise 子图 0 → CUDAGraphWrapper(PIECEWISE)│  │
│  │  attention (eager)                             │  │
│  │  piecewise 子图 1 → CUDAGraphWrapper(PIECEWISE)│  │
│  │  ...                                           │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

- **runtime_mode = FULL**：外层 wrapper 激活（捕获整图），内层 wrapper 因为模式不匹配全部透传 → 整图捕获成功。
- **runtime_mode = PIECEWISE**：外层透传，内层各自捕获/重放 → 分段图生效。
- **runtime_mode = NONE**：全都透传 → eager。

**第 ② 步的"模式不匹配就透传"是让两套图共存而不冲突的唯一机制。**

### 捕获时的两个调优开关

`CUDAGraphOptions`（`:139`）：

```python
debug_log_enable: bool
gc_disable: bool
weak_ref_output: bool
```

- **`gc_disable`**：捕获时临时禁用 Python GC。注释解释了原因：**分段捕获时几乎每层一张图，每张图跑一次 GC 是灾难性的**。
- **`weak_ref_output`**：把输出换成弱引用，让图的静态输出 buffer 能被释放。注释明确说**这只在"最后一个子图"上安全**（因为前一个子图的输出是后一个的输入，必须保持有效）。

以及引用对象的处理：

```python
# 捕获期间临时 patch 掉 gc.collect 和 torch.accelerator.empty_cache
```

> 💡 **性能视角**：`gc_disable` 这个细节说明了 CUDA Graph 捕获的隐藏成本 —— **捕获时间是启动时间的主要成分之一**，所以任何能在捕获路径上省下的东西都值得省。
> 相关数字：`-O2` 的启动时间可能是 `-O0` 的 5~10 倍，大部分花在捕获和编译上。

### 一个重要的边界说明

docstring 里说：wrapper **不负责**保存持久 buffer、也**不负责**把运行时输入拷进静态 buffer —— 那是调用方的责任（`make_copy_and_call`，见 6.7）。
输入地址的一致性检查只在 `VLLM_LOGGING_LEVEL == "DEBUG"` 时做。

## 6.6 `support_torch_compile`：模型是怎么被编译的

`vllm/compilation/decorators.py:120`（多个 overload，实现在 `:333`）。

### 用法

```python
@support_torch_compile
class MyModel(nn.Module):
    def forward(self, input_ids, positions, ...): ...

# 或者显式指定动态维度
@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class MyModel(nn.Module): ...
```

### 它是怎么注入的

```python
# _support_torch_compile (:333)
cls.__bases__ = (*cls.__bases__, TorchCompileWithNoGuardsWrapper)
```

**用 MRO 注入一个基类，而不是替换 `forward`。** 这样做的好处是不破坏原有的继承链（模型之间的继承关系很复杂）。

### 动态维度推断

`cls_decorator_helper` 从 `inspect.signature(cls.forward)` 的类型注解推断：

| 注解 | 推断结果 |
| --- | --- |
| `torch.Tensor` / `Optional[torch.Tensor]` | 第 0 维动态 |
| `IntermediateTensors` | 每个张量的第 0 维动态 |

如果推断不出来，会**报错并要求显式传 `dynamic_arg_dims`** —— 又是一个 fail-fast。

`dynamic_arg_dims` 的值可以是 `int` / `list[int]` / `dict[int, str]`（dim → shape_id，共享 shape_id 的维度共用一个符号）。

`mark_unbacked_dims` 与 `DynamicShapesType.BACKED` / `UNBACKED` 对应（`torch._dynamo.mark_dynamic` vs `mark_unbacked`）。

### 一个必须知道的约束

```python
# 警告大意：某个参数如果在本模型生命周期内为 None，就必须一直为 None，
# 否则无法被捕获成单一张图。
```

**推论**：可选参数的"有时传有时不传"会破坏编译/图捕获。这是写自定义模型时最容易踩的坑。

### `do_not_compile` 的判定

```python
self.do_not_compile = (
    compilation_config.mode in [NONE, STOCK_TORCH_COMPILE]
    or _should_ignore_torch_compile(cls)     # @ignore_torch_compile 装饰器
    or not enable_compile                    # enable_if 谓词
)
```

为真时直接走原始 `forward`。

## 6.7 分段：`split_graph` 与 `PiecewiseBackend`

### `split_graph`

`vllm/compilation/backends.py:553`。要点：

```python
# _decompose_size_nodes (:479)   先把 size 相关节点拆开
# should_split(node, splitting_ops) (:578)  判断是否切
# getitem 节点强制归到其输入同一子图 (:577-583)
# 连续 splitting op 保持在一起 (:590-596)
torch.fx.passes.split_module.split_module(..., keep_original_order=True)   # :614-621
```

两个注释里的原因很值得记：

- **`getitem` 强制同组**：避免把整个 tuple 当成子模块的输入，违反 `standalone_compile` / `AoTAutograd` 的要求。
- **`keep_original_order=True` 必须**：否则 PyTorch 会重排节点，改变有 mutation 的图的语义。**这是个静默错误源**。

### `PiecewiseBackend`

`vllm/compilation/piecewise_backend.py:86`。两种互斥模式：

| 模式 | 何时 | 字段 |
| --- | --- | --- |
| 编译模式 | 刚编译完 | `graph` 有值，`compiled_runnables=None` |
| 预编译模式 | 从缓存/AOT artifact 加载 | `graph=None`，`compiled_runnables` 有值 |

核心是 `__call__`（`:358`）：

```python
runtime_shape = args[self.sym_shape_indices[0]]
range_entry = self._find_range_for_shape(runtime_shape)
return range_entry.runnable(*args)
```

**运行时开销 = 一次查表 + 一次函数调用。** 这就是分段编译的运行时税，非常低。

`_find_range_for_shape`（`:343`）先找 `compile_sizes` 里的精确匹配，再找包含该 shape 的 range。

**encoder 的特殊处理**（`:133-146`）：编译视觉编码器时，把最后一个 range 的 `end` 改成 `2**31-1` —— 因为**图像输入的形状不可预测**（分辨率/帧数任意）。这是一个很实用的工程妥协。

## 6.8 让变长 batch 复用静态图：`make_copy_and_call`

`vllm/compilation/backends.py:59`：

```python
def make_copy_and_call(sym_tensor_indices, input_buffers, ...):
    # 首次调用时 input_buffers[i] = runtime_tensor.clone()（惰性初始化）
    # 之后 static_tensor = input_buffers[i][:runtime_shape]
    #     static_tensor.copy_(runtime_tensor)
    # 然后调用编译好的图
```

**核心思路**：图的输入 buffer 是**固定形状**的（捕获时的最大形状），每次调用只把实际数据拷进前 `runtime_shape` 个位置。

这就解决了"图要求固定形状，但 batch 每步都在变"的矛盾 —— 用一个**超集 buffer + 前缀拷贝**来兼容。

`sym_tensor_indices` 标记哪些位置是符号形状输入。

## 6.9 融合 pass：省的是显存带宽

`vllm/compilation/passes/fusion/`。按"省了什么"分类最有教育意义：

| Pass | 融合内容 | 省了什么 | 启用级别 |
| --- | --- | --- | --- |
| `add_rms_fusion` | residual add + RMSNorm | 一次显存往返 + 一次 launch | 默认 |
| `allreduce_rms_fusion` | allreduce + RMSNorm | 同上（**TP 场景收益大**） | `-O2` |
| `act_quant_fusion` | 激活 + 量化 | 一次 kernel | `-O1`（需自定义 kernel） |
| `rms_quant_fusion` | RMSNorm + 量化 | 一次 kernel | — |
| `rope_kvcache_fusion` | RoPE + KV cache 写入 | **一次 kernel + 一次显存读** | `-O2` |
| `qk_norm_rope_fusion` | QK norm + RoPE | 一次 kernel | — |
| `collective_fusion` | 多个小 collective 合成一个 | launch 次数 | — |
| `sequence_parallelism` | allreduce → reduce-scatter + all-gather | 通信量减半 | **需要全图** |
| `attn_quant_fusion` | attention + 输出量化 | — | **需要全图** |
| `mla_*` 系列 | MLA 专用（rope+kvcache+cat） | — | — |

**为什么融合这么值钱？** 因为在 memory-bound 的场景：

```text
未融合：read A (带宽) → compute → write B → read B → compute → write C
        = 2 次写 + 2 次读显存

融合后：read A → compute → compute → write C
        = 1 次写 + 1 次读显存
```

**elementwise 操作的算力几乎没用上，全部时间花在等显存。** 融合直接把显存流量减半。

### 为什么有些 pass"需要全图"

`sequence_parallelism` 和 `attn_quant_fusion` 需要看到 attention 前后的完整数据流，而 piecewise 编译会把 attention 切成边界。所以 vLLM 有一个自动处理：**启用这些 pass 时自动设 `splitting_ops=[]`**，代价是失去 piecewise CUDA Graph。

`docs/design/cuda_graphs.md` 里明确承认这是"confusing performance tradeoffs"，并给出了长期方案：

```python
CompilationConfig.use_inductor_graph_partition = True
```

**在 Inductor 层而不是 Dynamo 层切图**（需要 torch >= 2.9），这样全图融合和分段捕获可以兼得。目前是实验特性，会增加编译时间（无法复用 piecewise 编译产物）。

### `VllmBackend` 与 pass 的注册

`vllm/compilation/backends.py:805`：

```python
class VllmBackend:
    def __init__(self, vllm_config):
        self.pass_manager = ...        # 平台 PassManager
        self.compiler_manager = CompilerManager(...)
    def configure_post_pass(self):
        # 把 VllmIRInplaceFunctionalizationPass 注册为 Inductor 的 pre_grad_custom_pass
        # 并把它加进 _cache_config_ignore_prefix
```

**最后一句很关键**：如果不把自定义 pass 加进 `_cache_config_ignore_prefix`，它就会进入 AOTAutograd 的 cache key，**导致编译缓存全部失效**。这是一个典型的"做了优化反而变慢"的坑。

`passes/` 目录结构：

```text
passes/
├─ fusion/          融合 pass（见上表）
├─ utility/         fix_functionalization, noop_elimination, post_cleanup,
│                   scatter_split_replace, split_coalescing
├─ ir/              clone_elimination, inplace_functionalization, lowering_pass
├─ inductor_pass.py InductorPass 基类 + pass_context
├─ pass_manager.py  PostGradPassManager
└─ vllm_inductor_pass.py
```

## 6.10 AOT 编译与缓存

相关环境变量（`vllm/envs.py`）：

| 变量 | 作用 |
| --- | --- |
| `VLLM_USE_AOT_COMPILE` | 启用 AOT 编译（默认在满足条件时开启） |
| `VLLM_USE_STANDALONE_COMPILE`（默认 `1`） | 用 `torch.compiler` 的 standalone compile 路径 |
| `VLLM_FORCE_AOT_LOAD` | 强制加载 AOT artifact（不做 JIT 编译） |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | 用一个大的 artifact 文件 |
| `VLLM_CACHE_ROOT` | 编译缓存目录 |
| `VLLM_COMPILE_DEPYF` | 调试：用 depyf 生成可读的编译代码 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | 见 6.11 |

`decorators.py` 里的相关函数：

```python
_model_hash_key(fn)              # :257  缓存键
_verify_source_unchanged(...)    # :267  源码校验
_try_load_aot_compiled_fn(...)   # :286  尝试加载 AOT artifact
```

**为什么编译缓存重要？** `-O2` 下首次编译可能要 1~5 分钟。缓存命中后启动时间降到十几秒。**在 k8s 里滚动更新时，这个差异决定了扩容速度。**

> 💡 **性能视角**：预编译 + 缓存是"启动时间 vs 运行性能"这个权衡的第三条路 —— **两者都要**。
> 生产实践：构建镜像时用相同配置跑一次 warmup，把 `VLLM_CACHE_ROOT` 打进镜像层。

## 6.11 breakable CUDA Graph

`vllm/compilation/breakable_cudagraph.py`（env `VLLM_USE_BREAKABLE_CUDAGRAPH`）。

普通的 CUDA Graph 是**原子**的：一旦 replay 就不能中途跳出。但在异步调度/投机解码下，有时需要"跑到一半根据结果决定下一步"。

Breakable CUDA Graph 的想法是：**把图切成若干段，段之间可以插入 CPU 判断**，同时保持每段的图捕获收益。这是比 piecewise 更细的粒度。

相关的断言在 `CudagraphDispatcher.__init__` 里：piecewise 模式要求 `splitting_ops` 里有 attention，**或者**启用了 breakable cudagraph。

## 6.12 本章小结

| 机制 | 解决的问题 | 代价 |
| --- | --- | --- |
| CUDA Graph | CPU launch 开销（可占 30~50%） | 启动慢、显存占用、形状必须固定 |
| Piecewise | attention 无法进图 | 引入分段边界，某些融合 pass 无法用 |
| `CudagraphDispatcher` | 双模式分派的唯一事实来源 | 需要在 DP 下同步模式 |
| 嵌套 Wrapper | FULL 与 PIECEWISE 共存 | "模式不匹配就透传"的约定 |
| `gc_disable` / `weak_ref_output` | 捕获时间与显存 | 只在特定位置安全 |
| `support_torch_compile` | 图编译的接入 | `None` 参数必须始终为 `None` |
| `make_copy_and_call` | 变长 batch 复用静态图 | 前缀拷贝开销 |
| 融合 pass | 显存带宽（memory-bound 的命门） | 编译时间；部分与 piecewise 互斥 |
| `_cache_config_ignore_prefix` | 防止自定义 pass 使缓存失效 | — |
| AOT + 缓存 | 启动时间 | 镜像体积、配置一致性要求 |

⚠️ **易错点**
- `use_inductor_graph_partition` 是实验特性，不是默认路径。
- `splitting_ops=[]`（不切图）会让 `PIECEWISE` 模式不可用 —— 如果你同时想要"全图融合"和"分段图"，目前只能二选一。
- 编译缓存的 key 包含 vLLM 版本、torch 版本、配置 hash。**改了配置就要重新编译**，不要以为缓存能跨配置复用。

📖 官方文档：`docs/design/cuda_graphs.md`、`docs/design/torch_compile.md`、`docs/design/fusions.md`、`docs/design/optimization_levels.md`、`docs/design/debug_vllm_compile.md`

下一章 → [第 7 章 注意力内核与 KV 布局](07-注意力内核与KV布局.md)
