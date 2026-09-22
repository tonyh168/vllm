# vLLM Attention Backend 机制详解：接口、KV Cache 与计算流程

本文聚焦于 vLLM v1 引擎中 attention backend 的**核心机制**：它到底是怎么结合 KV cache 计算 attention 的？入参和返回值是什么？只讨论 bf16 场景，不涉及量化。

---

## 1. 总览：一次 Attention 计算的完整流程

当模型的某一层（如 `LlamaAttention`）调用 attention 时，完整数据流如下：

```
模型层 (e.g. LlamaAttention)
  │
  │  调用 self.attn.forward(query, key, value)
  ▼
Attention Module (vllm/model_executor/layers/attention/attention.py)
  │
  │  ① reshape Q/K/V → [num_tokens, num_heads, head_size]
  │  ② 调用 unified_kv_cache_update(key, value)  ← 把新的 K/V 写入 cache
  │  ③ 调用 unified_attention_with_output(query, key, value, output)  ← 计算 attention
  ▼
FlashAttentionImpl.forward(...)
  │
  │  从 kv_cache 中取出所有历史 K/V
  │  结合新 Q 执行 flash_attn_varlen_func
  ▼
output: [num_tokens, num_heads, head_size]
```

关键点：**KV cache 的写入和 attention 的计算是两步分开的操作**。

---

## 2. 核心抽象类

所有代码位于 `vllm/v1/attention/backend.py`。

### 2.1 AttentionBackend

Backend 的工厂/注册类，不参与计算，只负责声明能力和返回实现类：

```python
class AttentionBackend(ABC):
    # KV cache 写入是否包含在 forward() 中
    # FlashAttention 为 False —— 写入和计算分离
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str: ...

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]: ...

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]: ...

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]: ...
```

### 2.2 AttentionImpl

实际执行 attention 计算的类。只有两个核心方法：

```python
class AttentionImpl(ABC):
    def forward(
        self,
        layer: torch.nn.Module,          # Attention 层本身
        query: torch.Tensor,             # [num_tokens, num_heads, head_size]
        key: torch.Tensor,               # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,          # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,            # [num_tokens, num_heads, head_size] 预分配
    ) -> torch.Tensor: ...

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,               # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,          # [2, num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: torch.Tensor,      # [num_tokens] 每个 token 要写入的 cache 位置
    ) -> None: ...
```

### 2.3 AttentionMetadataBuilder

负责把通用的 batch 信息转换成 backend 专用的 metadata：

```python
class AttentionMetadataBuilder(ABC):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> AttentionMetadata: ...
```

---

## 3. KV Cache 的结构与管理

### 3.1 物理形状

对于 FlashAttention backend，KV cache 的 tensor 形状为：

```
kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
           │      │           │            │            │
           │      │           │            │            └─ 每个 head 的维度 (如 128)
           │      │           │            └─ KV head 数量 (GQA 时 < Q head 数)
           │      │           └─ 每个 block 能存几个 token (如 16/32/64)
           │      └─ 物理 block 总数 (GPU 显存决定)
           └─ 0=key_cache, 1=value_cache
```

通过 `kv_cache.unbind(0)` 拆成 `key_cache` 和 `value_cache`。

### 3.2 Paged 机制：Block Table 和 Slot Mapping

vLLM 使用 paged attention：显存被切分成固定大小的 block，每个请求通过 block table 找到自己的 KV 数据。

```
block_table: [num_requests, max_num_blocks_per_request]
             每个元素是一个物理 block 的 ID

slot_mapping: [num_tokens]
              每个新 token 对应一个 cache 中的 flat slot 索引
              slot = block_id * block_size + offset_in_block
```

举例：假设 block_size=16，某个请求已经有 35 个 token：
- 占用 3 个 block（block 0: token 0-15, block 1: token 16-31, block 2: token 32-34）
- 第 36 个新 token 的 slot = block_2_id * 16 + 3

### 3.3 KV Cache 的写入：reshape_and_cache_flash

当 `forward_includes_kv_cache_update = False`（FlashAttention 的情况）时，写入由独立的 C++ kernel 完成：

```python
# 简化流程
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    key_cache, value_cache = kv_cache.unbind(0)
    # key:   [num_tokens, num_kv_heads, head_size]
    # key_cache: [num_blocks, block_size, num_kv_heads, head_size]
    # slot_mapping: [num_tokens]  → 指明每个 token 写到 cache 的哪个 slot
    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto")
```

`reshape_and_cache_flash` 做的事情很简单：对于每个 token i，把 `key[i]` 写入 `key_cache` 的第 `slot_mapping[i]` 个位置（同理 value）。`slot_mapping[i] == -1` 表示 padding，跳过。

---

## 4. Attention 的计算：forward() 详解

以 FlashAttention 为例，`forward()` 的核心逻辑：

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
    key_cache, value_cache = kv_cache.unbind(0)

    # 调用 FlashAttention kernel
    # Q 来自当前 batch 的新 token
    # K/V 来自 kv_cache 中所有历史 token（通过 block_table 索引）
    flash_attn_varlen_func(
        q=query,                          # [num_tokens, num_heads, head_size]
        k=key_cache,                      # [num_blocks, block_size, num_kv_heads, head_size]
        v=value_cache,                    # 同上
        cu_seqlens_q=attn_metadata.query_start_loc,   # Q 的累积长度
        seqused_k=attn_metadata.seq_lens,             # 每个请求的 KV 总长度
        max_seqlen_q=attn_metadata.max_query_len,
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=self.scale,
        causal=attn_metadata.causal,
        block_table=attn_metadata.block_table,        # 通过 block table 定位 KV
        out=output,
    )
    return output
```

关键参数解释：

| 参数 | 含义 |
|------|------|
| `query` | 当前 batch 中所有新 token 的 Q 向量，shape `[total_new_tokens, num_heads, head_size]` |
| `key_cache / value_cache` | 整个 KV cache pool（不是只属于当前请求的） |
| `block_table` | `[num_requests, max_blocks]`，告诉 kernel 每个请求的 KV 存在哪些物理 block |
| `cu_seqlens_q` | `[num_requests + 1]` 累积 query 长度，用于区分 batch 中不同请求的 Q 段 |
| `seqused_k` | `[num_requests]` 每个请求的 KV 序列总长（包含历史 + 新写入的） |
| `causal` | 是否使用因果 mask（decoder 模型为 True） |

返回值：`output` tensor，shape `[total_new_tokens, num_heads, head_size]`，就是标准的 attention output。

---

## 5. Prefill 和 Decode 的统一处理

vLLM v1 不区分 prefill/decode 路径，两者在同一个 `forward()` 中处理：

```
Batch 示例（3 个请求同时处理）：

请求 A: prefill 阶段，输入 "Hello world"      → query_len = 2
请求 B: decode 阶段，生成第 50 个 token       → query_len = 1
请求 C: prefill 阶段，输入 "What is vLLM..."  → query_len = 5

total_new_tokens = 2 + 1 + 5 = 8
query_start_loc = [0, 2, 3, 8]      # 累积长度
seq_lens        = [2, 50, 5]         # 每个请求的 KV 总长度
```

区别仅在于：
- **Prefill**: `query_len > 1`，Q 和 K/V 长度相同（首次输入的所有 token 都是新的）
- **Decode**: `query_len = 1`，Q 只有 1 个新 token，但 K/V 可能有几百到几千（都在 cache 中）

FlashAttention kernel 通过 variable-length 接口天然支持混合 batch。

---

## 6. CommonAttentionMetadata：所有 backend 的共享输入

由 GPU Model Runner 在每次 forward 前构建，所有 layer 共享：

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor     # [batch_size + 1] Q 累积位置
    seq_lens: torch.Tensor            # [batch_size] 每个请求的 KV 序列总长
    block_table_tensor: torch.Tensor  # [batch_size, max_blocks] 物理 block 映射
    slot_mapping: torch.Tensor        # [num_tokens] 新 token 的 cache 写入位置
    num_actual_tokens: int            # batch 中实际 token 数（不含 padding）
    max_query_len: int                # batch 中最长的 query
    max_seq_len: int                  # batch 中最长的 KV 序列
    num_reqs: int                     # 请求数
    causal: bool                      # 因果注意力
```

每个 backend 的 `MetadataBuilder.build()` 接收这个通用结构，转换为自己需要的格式（如 FlashAttention 需要 AOT scheduling 参数，FlashInfer 需要 page table 格式等）。

---

## 7. ForwardContext：连接一切的桥梁

模型层（如 `LlamaAttention`）调用 `self.attn.forward(q, k, v)` 时，并不传递 KV cache 和 metadata。这些信息通过 `ForwardContext` 传递：

```python
@dataclass
class ForwardContext:
    attn_metadata: dict[str, AttentionMetadata]  # layer_name → metadata
    slot_mapping: dict[str, torch.Tensor]        # layer_name → slot mapping
    no_compile_layers: dict[str, Attention]       # layer_name → Attention module (持有 .kv_cache)
```

流程：
1. Model Runner 设置 `ForwardContext`（包含本次 forward 的所有 metadata 和 cache 引用）
2. 模型执行 forward pass，各层调用 `self.attn.forward(q, k, v)`
3. `Attention.forward()` 内部从 `ForwardContext` 获取当前层的 `kv_cache`、`attn_metadata`、`slot_mapping`
4. 先写 cache，再算 attention

这样模型代码不需要知道 KV cache 的细节，只管产生 Q/K/V 就行。

---

## 8. 端到端示例：一个 decode step 的生命周期

假设：Llama-7B, bf16, FlashAttention, block_size=16, head_size=128, num_kv_heads=32

**场景**：batch 中有 1 个请求，已经有 100 个 token 的 context，正在生成第 101 个 token。

```
1. Model Runner 构建 CommonAttentionMetadata:
   - query_start_loc = [0, 1]         # 1 个新 token
   - seq_lens = [101]                  # KV 总长 101（100 历史 + 1 新）
   - block_table = [[3, 7, 12, 5, 9, 2, 0]]   # 7 个 block（7*16=112 slots，用了 101）
   - slot_mapping = [100]              # 第 101 个 token 写入 slot 100
                                       #   = block_table[6] * 16 + (100 % 16) = 0*16 + 4 = ...
                                       #   (实际由 scheduler 计算)

2. ForwardContext 被设置

3. 模型 forward，LlamaAttention 产生:
   - query:  [1, 32, 128]  (1 个 token, 32 头, 128 维)
   - key:    [1, 32, 128]
   - value:  [1, 32, 128]

4. Attention.forward() 被调用:

   4a. KV Cache 写入:
       reshape_and_cache_flash(
           key=[1, 32, 128],
           value=[1, 32, 128],
           key_cache=[num_blocks, 16, 32, 128],
           value_cache=[num_blocks, 16, 32, 128],
           slot_mapping=[100],   # 写入 cache 的第 100 号位置
       )

   4b. Attention 计算:
       flash_attn_varlen_func(
           q=[1, 32, 128],                    # 1 个 query token
           k=key_cache,                       # 整个 cache pool
           v=value_cache,
           cu_seqlens_q=[0, 1],              # 1 个请求，1 个 query token
           seqused_k=[101],                  # attend 到 101 个 KV token
           block_table=[[3, 7, 12, 5, 9, 2, 0]],  # 定位这 101 个 token 在 cache 中的位置
           causal=True,
           softmax_scale=1/sqrt(128),
       )

5. 返回 output: [1, 32, 128] → reshape → [1, 4096]
```

---

## 9. 接口签名速查

### 模型层调用的接口

```python
# 模型层（如 LlamaAttention）调用这个
class Attention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,    # [num_tokens, num_heads * head_size]
        key: torch.Tensor,      # [num_tokens, num_kv_heads * head_size]
        value: torch.Tensor,    # [num_tokens, num_kv_heads * head_size]
    ) -> torch.Tensor:          # [num_tokens, num_heads * head_size]
```

注意：模型层传入的是 **2D** tensor（`[num_tokens, hidden_dim]`），`Attention.forward()` 内部会 reshape 成 3D 再传给 backend。

### Backend 实现的接口

```python
class FlashAttentionImpl(AttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,             # [num_tokens, num_heads, head_size]
        key: torch.Tensor,               # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,          # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,            # [num_tokens, num_heads, head_size] (预分配)
    ) -> torch.Tensor:                   # 返回 output 本身

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,               # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,          # [2, num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: torch.Tensor,      # [num_tokens]
    ) -> None:
```

---

## 10. 关键设计决策总结

| 设计 | 原因 |
|------|------|
| KV cache 写入和 attention 计算分离 | 允许 torch.compile 更好地融合计算，写入可以被单独图化 |
| 使用 paged blocks 而非连续内存 | 避免显存碎片，支持动态 batch，类似 OS 虚拟内存分页 |
| ForwardContext 传递 cache/metadata | 模型代码无需关心 cache 细节，解耦模型层和推理引擎 |
| Prefill 和 decode 统一处理 | 简化调度，允许混合 batch，一次 kernel launch 处理所有请求 |
| 预分配 output tensor | 减少内存分配开销，支持 CUDAGraph capture |
| block_table 在 kernel 内索引 | paged attention 的核心：kernel 通过 block_table 直接访问非连续的 cache blocks |

---

## 11. 代码定位

| 你想了解的内容 | 去看哪里 |
|---------------|---------|
| 抽象基类定义 | `vllm/v1/attention/backend.py` |
| Attention Module（模型层调用入口） | `vllm/model_executor/layers/attention/attention.py` |
| FlashAttention 完整实现 | `vllm/v1/attention/backends/flash_attn.py` |
| KV cache 写入 kernel | `vllm/_custom_ops.py` → `reshape_and_cache_flash` |
| ForwardContext 定义 | `vllm/forward_context.py` |
| CommonAttentionMetadata 构建 | `vllm/v1/worker/gpu_model_runner.py` |
| KV cache 绑定到 Attention 层 | `vllm/v1/worker/utils.py` → `bind_kv_cache()` |
