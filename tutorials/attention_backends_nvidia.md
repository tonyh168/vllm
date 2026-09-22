# vLLM Attention Backend 详解（NVIDIA 平台）

本文档深入介绍 vLLM（v1 引擎）在 NVIDIA GPU 上可用的各种 Attention Backend，包括架构设计、选择机制、各 backend 特性对比，以及 MLA（Multi-head Latent Attention）专用 backend。

---

## 1. 整体架构

vLLM v1 的 attention 子系统位于 `vllm/v1/attention/` 目录下，核心组成如下：

```
vllm/v1/attention/
├── backend.py          # 抽象基类: AttentionBackend, AttentionImpl, AttentionMetadataBuilder
├── selector.py         # Backend 选择器: get_attn_backend()
├── backends/
│   ├── registry.py     # AttentionBackendEnum 注册表
│   ├── flash_attn.py   # FlashAttention backend (FA2/FA3/FA4)
│   ├── flashinfer.py   # FlashInfer backend
│   ├── triton_attn.py  # Triton 纯 Python attention backend
│   ├── flex_attention.py  # PyTorch FlexAttention backend
│   ├── turboquant_attn.py # TurboQuant KV-cache 压缩 backend
│   ├── mla/            # MLA 专用 backend 目录
│   │   ├── flashinfer_mla.py
│   │   ├── flashinfer_mla_sparse.py
│   │   ├── cutlass_mla.py
│   │   ├── flashmla.py
│   │   ├── flashmla_sparse.py
│   │   ├── flashattn_mla.py
│   │   └── triton_mla.py
│   └── ...
└── ops/                # 底层 kernel 实现
```

### 1.1 核心抽象层

**`AttentionBackend`**（`backend.py`）是所有 backend 的抽象基类，定义了：

| 方法/属性 | 说明 |
|-----------|------|
| `get_name()` | backend 标识名 |
| `get_impl_cls()` | 返回具体的 AttentionImpl 类 |
| `get_builder_cls()` | 返回 AttentionMetadataBuilder 类 |
| `get_kv_cache_shape(...)` | KV cache tensor 的形状 |
| `get_kv_cache_stride_order(...)` | KV cache 物理内存布局排列 |
| `supports_head_size(h)` | 是否支持指定的 head_size |
| `supports_compute_capability(cc)` | 是否支持指定 GPU 架构 |
| `supports_kv_cache_dtype(d)` | 是否支持指定的 KV cache 数据类型 |
| `supports_sink()` | 是否支持 attention sink |
| `supports_non_causal()` | 是否支持非因果注意力 |
| `validate_configuration(...)` | 综合验证所有配置是否合法 |

**`AttentionImpl`** 是 forward 计算的具体实现类，定义了标准 MHA 的 `forward()` 接口。

**`MLAAttentionImpl`** 是 MLA 架构专用的实现类，提供 `forward_mha()`（prefill）和 `forward_mqa()`（decode）两个接口。

---

## 2. Backend 选择机制

### 2.1 选择流程

```
用户指定 --attention-backend
       │
       ▼
┌──────────────────┐
│ get_attn_backend()│  (selector.py)
│                  │
│  读取 vllm_config.attention_config.backend
│        │
│        ▼
│  current_platform.get_attn_backend_cls()
└──────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ CudaPlatformBase             │  (platforms/cuda.py)
│                              │
│  若用户指定了 backend:        │
│    直接 validate → 使用      │
│                              │
│  若未指定:                    │
│    _get_backend_priorities() │
│    → 按优先级逐个 validate   │
│    → 选第一个通过的          │
└──────────────────────────────┘
```

### 2.2 优先级规则

优先级取决于 **GPU 架构** 和 **是否使用 MLA**：

#### 非 MLA 模型

| 优先级 | SM100 (Blackwell) | SM90 及以下 (Hopper/Ampere/...) |
|--------|--------------------|---------------------------------|
| 1 (最高) | FlashInfer | FlashAttention |
| 2 | FlashAttention | FlashInfer |
| 3 | Triton Attention | Triton Attention |
| 4 | FlexAttention | FlexAttention |
| 5 | TurboQuant | TurboQuant |

#### MLA 模型（如 DeepSeek-V2/V3）

| 优先级 | SM100 (Blackwell) | SM90 及以下 |
|--------|--------------------|----|
| 1 | FlashInfer MLA | FlashAttn MLA |
| 2 | CUTLASS MLA | FlashMLA |
| 3 | FlashAttn MLA | FlashInfer MLA |
| 4 | FlashMLA | Triton MLA |
| 5 | Triton MLA | FlashMLA Sparse |
| 6+ | Sparse variants | — |

### 2.3 用户指定方式

```bash
# 启动参数
vllm serve model_name --attention-backend FLASH_ATTN

# 或通过 attention_config
vllm serve model_name --attention-config '{"backend": "FLASHINFER"}'

# 指定 FlashAttention 版本
vllm serve model_name --attention-config '{"flash_attn_version": 3}'

# 其他 attention_config 选项
# use_trtllm_attention: 控制 FlashInfer 是否使用 TRTLLM kernel
# disable_flashinfer_prefill: 禁用 FlashInfer 的 prefill kernel（默认 True）
# use_cudnn_prefill: 使用 cuDNN 做 prefill
```

有效的 backend 名称参见 `AttentionBackendEnum`（`registry.py`）中的枚举值。

---

## 3. NVIDIA 平台各 Backend 详解

### 3.1 FlashAttention Backend

**文件**: `vllm/v1/attention/backends/flash_attn.py`  
**枚举名**: `FLASH_ATTN`

FlashAttention 是 vLLM 在 Ampere/Hopper 上的默认 attention backend，使用 Dao-AILab 的 [flash-attention](https://github.com/Dao-AILab/flash-attention) 库。

#### FlashAttention 版本选择

| GPU 架构 | 默认 FA 版本 | 说明 |
|----------|-------------|------|
| SM80 (Ampere) | FA2 | A100/A800 |
| SM90 (Hopper) | FA3 | H100/H800，支持 TMA + warp-specialization |
| SM100 (Blackwell) | FA4 | B200/B100，使用最新的 Blackwell 指令集 |

可通过 `--attention-config '{"flash_attn_version": 2}'` 强制指定 FA 版本。

各版本能力差异：

| 特性 | FA2 | FA3 | FA4 |
|------|-----|-----|-----|
| FP8 KV Cache | ✗ | ✓ (SM90) | ✓ |
| Attention Sink | ✗ | ✓ | ✓ |
| Per-head Quant Scales | ✗ | ✓ | ✓ |
| CUDAGraph | UNIFORM_BATCH | ALWAYS | ALWAYS |
| AOT Scheduling | ✗ | ✓ | ✓ |
| ALiBi | ✓ | ✗（回退 FA2） | ✗（回退 FA2） |
| Batch Invariance | ✓ | ✓ | ✗（回退 FA2） |
| Max head_size | 任意 | 任意 | 128（MLA 可 192） |

#### 关键特性

- **支持的 GPU**: SM80+（Ampere 及更新架构）
- **支持的数据类型**: fp16, bf16
- **KV Cache 类型**: auto, float16, bfloat16（FA3+ 支持 fp8 per-head quant scales）
- **Block Size**: 16 的倍数
- **KV Cache 布局**: `(2, num_blocks, block_size, num_kv_heads, head_size)`
- **Attention Sink**: SM90+ 支持
- **CUDAGraph**: 支持（uniform batch）
- **Cascade Attention**: 支持
- **非因果注意力**: 支持
- **Encoder-Decoder**: 支持
- **Batch Invariance**: 支持

#### 特殊说明

- `forward_includes_kv_cache_update = False`：FlashAttention 的 forward 不包含 KV cache 写入，写入由独立的 `reshape_and_cache_flash` kernel 完成
- 支持 HND 和 NHD 两种 KV cache 内存布局

---

### 3.2 FlashInfer Backend

**文件**: `vllm/v1/attention/backends/flashinfer.py`  
**枚举名**: `FLASHINFER`

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 是一个专门为 LLM 推理优化的 attention 库，在 Blackwell (SM100) 上是默认首选。

#### 关键特性

- **支持的 GPU**: SM75 ~ SM121（从 Turing 到 Blackwell 全覆盖）
- **支持的数据类型**: fp16, bf16
- **KV Cache 类型**: auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2
- **Block Size**: 16, 32, 64
- **Head Size**: 64, 128, 256
- **KV Cache 布局**: `(num_blocks, 2, block_size, num_kv_heads, head_size)`
- **Attention Sink**: 在 SM100 上通过 TRTLLM attention 支持
- **CUDAGraph**: 支持（ALWAYS 级别，支持混合 prefill-decode）
- **FP8 KV Cache**: 原生支持，包括 e4m3 和 e5m2 两种格式
- **NVFP4 KV Cache**: 支持（packed layout）

#### FlashInfer 的优势

1. **Paged KV Cache 原生支持**：直接在 page table 上操作，无需额外的 gather/scatter
2. **FP8 量化**：原生支持 FP8 KV cache，减少显存占用
3. **Cascade Attention**：支持 prefix 共享的级联注意力
4. **Blackwell 优化**：在 SM100 上集成了 TRTLLM attention kernel

#### TRTLLM Attention 集成

在 Blackwell (SM100) 平台上，FlashInfer 可以使用 TensorRT-LLM 的 attention kernel（通过 `use_trtllm_attention` 配置控制），这些 kernel 针对 Blackwell 硬件做了深度优化。

---

### 3.3 Triton Attention Backend

**文件**: `vllm/v1/attention/backends/triton_attn.py`  
**枚举名**: `TRITON_ATTN`

纯 Triton 实现的 attention backend，无需额外的 C++/CUDA 编译依赖。

#### 关键特性

- **支持的 GPU**: 所有 NVIDIA GPU（无 compute capability 限制）
- **支持的数据类型**: fp16, bf16, fp32
- **KV Cache 类型**: auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, int8_per_token_head, fp8_per_token_head
- **Block Size**: 16 的倍数
- **Head Size**: ≥ 32
- **Attention Sink**: 支持
- **Encoder-Decoder**: 支持
- **ALiBi sqrt**: 支持
- **Batch Invariance**: 支持
- **Per-head Quant**: 支持 per-token-head 量化

#### 适用场景

- 当 FlashAttention 和 FlashInfer 都无法使用时的兜底方案
- 需要 per-token-head 量化（int8/fp8）时
- 需要 fp32 精度调试时
- head_size 不被 FlashAttention 支持时

#### 架构特点

使用 3D 并行 softmax 分段策略（`NUM_PAR_SOFTMAX_SEGMENTS = 16`），对长序列进行分块计算再合并。

---

### 3.4 FlexAttention Backend

**文件**: `vllm/v1/attention/backends/flex_attention.py`  
**枚举名**: `FLEX_ATTENTION`

基于 PyTorch 原生的 `torch.nn.attention.flex_attention` API，通过 `torch.compile` 编译优化。

#### 关键特性

- **支持的数据类型**: fp16, bf16, fp32
- **KV Cache 类型**: auto, float16, bfloat16
- **非因果注意力**: 支持
- **Multimodal Prefix**: 支持（图像 token 的全注意力）
- **Cascade Attention**: 不支持
- **CUDAGraph**: 不支持

#### 适用场景

- 需要自定义 attention mask（通过 `mask_mod` / `score_mod`）
- 需要利用 `torch.compile` 的全图优化能力
- 实验性用途或特殊 attention pattern

#### 局限性

- 不支持 FP8 KV cache
- 不支持 CUDAGraph
- 不支持 cascade attention
- 性能通常不如 FlashAttention / FlashInfer

---

### 3.5 TurboQuant Backend

**文件**: `vllm/v1/attention/backends/turboquant_attn.py`  
**枚举名**: `TURBOQUANT`

专门为超低比特 KV cache 压缩设计的 backend。将 K/V 合并存储在一个 slot 中，使用 3-bit/4-bit 量化压缩 key，4-bit 压缩 value。

#### 关键特性

- **支持的数据类型**: fp16, bf16
- **KV Cache 类型**: turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc
- **Block Size**: 16, 32, 64, 128
- **CUDAGraph**: 支持（uniform batch）

#### Cache 布局

与其他 backend 不同，TurboQuant 不使用 `(2, num_blocks, ...)` 的 K/V 分离布局，而是：

```
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

每个 slot 内部结构：`[key_packed | value_packed | padding]`

例如对于 `turboquant_k3v4_nc`，head_dim=256 时：
- key_packed: 100 bytes
- value_fp16: 512 bytes (256 × 2)

#### 工作流程

- **Prefill**: 使用标准 FlashAttention 计算注意力，然后量化 K 并写入 TQ cache
- **Decode**: 从压缩 cache 中直接计算 TQ attention scores

#### 适用场景

- 需要极致压缩 KV cache 以服务更长序列或更大 batch
- 可以容忍一定精度损失

---

### 3.6 FlashAttention DiffKV Backend

**文件**: `vllm/v1/attention/backends/flash_attn_diffkv.py`  
**枚举名**: `FLASH_ATTN_DIFFKV`

FlashAttentionBackend 的子类，专门处理 Key 和 Value head dimension 不同的模型（如某些 GQA 变体）。

- KV Cache 布局为交错存储：`(num_blocks, block_size, num_kv_heads, head_size + head_size_v)`
- 其他特性继承自 FlashAttentionBackend

---

### 3.7 Tree Attention Backend

**文件**: `vllm/v1/attention/backends/tree_attn.py`  
**枚举名**: `TREE_ATTN`

为树状推测解码（tree-based speculative decoding）设计的 attention backend。

- **支持的数据类型**: fp16, bf16
- **Head Size**: 32-256（32 的倍数）
- **Block Size**: 16 的倍数

---

## 4. MLA 专用 Backend

MLA（Multi-head Latent Attention）是 DeepSeek-V2/V3 使用的注意力机制，将 KV 投影到低维潜在空间以减少 KV cache 占用。

### 4.1 FlashInfer MLA

**枚举名**: `FLASHINFER_MLA`  
**Sparse 变体**: `FLASHINFER_MLA_SPARSE`

- SM100 上的默认 MLA backend
- 仅支持 SM100 (Blackwell)
- 要求 `qk_nope_head_dim` 为 64/128/192
- Block size: 32, 64
- 支持 fp8 KV cache
- Sparse 版本用于 decode 阶段加速

### 4.2 CUTLASS MLA

**枚举名**: `CUTLASS_MLA`

- 仅支持 SM100 (Blackwell)
- 使用 CUTLASS SM100 workspace 实现高性能矩阵乘法
- Block size 固定为 128
- 支持 fp8 KV cache
- 支持 CUDAGraph（uniform single token decode）

### 4.3 FlashMLA

**枚举名**: `FLASHMLA`  
**Sparse 变体**: `FLASHMLA_SPARSE`

- 支持 SM90 (Hopper) 和 SM100 (Blackwell)
- Block size 固定为 64
- 支持 fp8 KV cache
- Sparse 变体：仅支持 bf16，head_size 为 512 或 576
- 源自 [FlashMLA](https://github.com/deepseek-ai/FlashMLA) 项目

### 4.4 FlashAttn MLA

**枚举名**: `FLASH_ATTN_MLA`

- 仅支持 SM90 (Hopper)，依赖 FA3
- 将 MLA 映射到标准 FlashAttention 接口
- Block size: 16 的倍数
- 在非 Blackwell 平台上优先级最高

### 4.5 Triton MLA

**枚举名**: `TRITON_MLA`

- 纯 Triton 实现，作为兜底
- 无特殊硬件要求

---

## 5. KV Cache 布局

vLLM 支持两种 KV cache 内存布局，通过 `VLLM_KV_CACHE_LAYOUT` 环境变量或自动选择：

### NHD 布局（默认）

```
逻辑形状: (2, num_blocks, block_size, num_kv_heads, head_size)
物理布局: 与逻辑形状一致
```

- 适合大多数场景
- FlashAttention (FA2) 和 FlashInfer 默认使用此布局

### HND 布局

```
逻辑形状: (2, num_blocks, block_size, num_kv_heads, head_size)
物理布局: (num_blocks, num_kv_heads, 2, block_size, head_size)
```

- 将 num_kv_heads 维度前置
- 在某些场景下对 cache 访问更友好

Backend 可以通过 `get_required_kv_cache_layout()` 声明自己必须使用的布局，系统会自动适配。FlashInfer 在 Blackwell (SM100) 上强制使用 HND 布局。

### MLA KV Cache 布局

MLA backend 使用完全不同的 cache 布局，将压缩后的 latent KV 存储在一个维度中：

```
(num_blocks, block_size, latent_dim)
```

其中 `latent_dim = kv_lora_rank + qk_rope_head_dim`，通常为 512 或 576。

---

## 6. CUDAGraph 支持等级

各 backend 对 CUDAGraph 的支持程度不同，由 `AttentionCGSupport` 枚举表示：

| 等级 | 含义 | 代表 Backend |
|------|------|-------------|
| `ALWAYS` | 全场景支持，包括混合 prefill-decode | FlashInfer |
| `UNIFORM_BATCH` | 支持 query 长度一致的 batch（含 spec-decode） | FlashAttention |
| `UNIFORM_SINGLE_TOKEN_DECODE` | 仅支持所有请求 query_len=1 的纯 decode batch | 部分 MLA backend |
| `NEVER` | 不支持 CUDAGraph | FlexAttention |

---

## 7. 特性对比总结

| 特性 | FlashAttn | FlashInfer | Triton | FlexAttn | TurboQuant |
|------|-----------|------------|--------|----------|------------|
| 最低 SM | 80 | 75 | 任意 | 任意 | 任意 |
| FP8 KV Cache | FA3+ ✓ | ✓ | ✓ | ✗ | N/A |
| NVFP4 KV Cache | ✗ | ✓ | ✗ | ✗ | N/A |
| Attention Sink | SM90+ | SM100 | ✓ | ✗ | ✗ |
| CUDAGraph | Uniform | Always | Uniform | ✗ | Uniform |
| Cascade Attention | ✓ | ✓ | ✗ | ✗ | ✗ |
| 非因果注意力 | ✓ | ✗ | ✗ | ✓ | ✗ |
| Encoder-Decoder | ✓ | ✗ | ✓ | ✗ | ✗ |
| Per-head Quant | FA3+ | ✗ | ✓ | ✗ | 内置 |
| fp32 支持 | ✗ | ✗ | ✓ | ✓ | ✗ |
| Batch Invariance | ✓ | ✓ | ✓ | ✗ | ✗ |

---

## 8. 自定义 Backend 注册

vLLM 支持通过 `register_backend` 装饰器注册自定义 backend：

```python
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

# 方式一：装饰器
@register_backend(AttentionBackendEnum.CUSTOM)
class MyCustomBackend:
    ...

# 方式二：直接注册
register_backend(
    AttentionBackendEnum.CUSTOM,
    "my.module.MyCustomBackend"
)
```

使用时通过 `--attention-backend CUSTOM` 指定。

---

## 9. 调试与排查

### 查看实际选择的 backend

vLLM 启动时会日志输出选择结果：

```
INFO: Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', ...]
```

### 常见问题

| 问题 | 可能原因 |
|------|----------|
| `No valid attention backend found` | 所有 backend 都不支持当前配置组合 |
| `head_size not supported` | FlashInfer 仅支持 64/128/256；FlashAttn 要求 8 的倍数 |
| `kv_cache_dtype not supported` | 使用了 backend 不支持的量化类型 |
| `compute capability not supported` | 在 SM75 以下使用 FlashAttention |
| `block_size not supported` | FlashMLA 只支持 64，CUTLASS MLA 只支持 128 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `VLLM_KV_CACHE_LAYOUT` | 强制 KV cache 布局 (NHD/HND) |
| `VLLM_BATCH_INVARIANT` | 启用 batch invariance 模式 |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | FlashInfer workspace 大小（默认 ~394MB） |

---

## 10. 代码阅读指引

如果你想深入理解某个 backend 的实现，建议按以下顺序阅读：

1. **`vllm/v1/attention/backend.py`** — 理解抽象接口
2. **`vllm/v1/attention/backends/registry.py`** — 所有 backend 的注册表
3. **`vllm/v1/attention/selector.py`** — 选择逻辑
4. **`vllm/platforms/cuda.py` → `_get_backend_priorities()`** — NVIDIA 平台的优先级配置
5. **具体 backend 文件** — 以 `flash_attn.py` 为起点，参照其结构理解其他 backend
