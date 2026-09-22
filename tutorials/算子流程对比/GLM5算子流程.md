# GLM4/GLM5 模型算子流程（从 Embedding 到 Token 采样）

> 以 GLM-4 Dense 和 GLM-4 MoE（即 GLM-4.5/4.6/4.7，也称 GLM5 系列）为例，
> 梳理 vLLM 中完整算子调用链。
> 模型入口：
> - Dense: `vllm/model_executor/models/glm4.py` → `Glm4ForCausalLM`
> - MoE: `vllm/model_executor/models/glm4_moe.py` → `Glm4MoeForCausalLM`

---

## 架构特点（相比 LLaMA）

1. **Partial RoPE** — 仅对 head_dim 的一半应用 RoPE（`partial_rotary_factor = 0.5`）
2. **Non-NeoX style RoPE** — 使用交错式旋转编码（`is_neox_style=False`）
3. **GLM-4 Dense 使用 4 个 RMSNorm**（额外的 post_self_attn_layernorm 和 post_mlp_layernorm）
4. **GLM-4 MoE 使用 sigmoid + grouped_topk 路由**（类似 DeepSeek）
5. **QK-Norm**（MoE 版本可选）

---

## 整体流程概览

```
input_ids
    │
    ▼
┌─────────────────────────┐
│  1. Embedding Lookup    │  PyTorch (VocabParallelEmbedding)
└────────────┬────────────┘
             │
             ▼
┌───────────────────────────────────────────────────────┐
│  2. DecoderLayer × N (循环)                            │
│  ┌──────────────────────────────────────────────────┐ │
│  │ 2.1 RMSNorm (input_layernorm)                   │ │  Custom CUDA
│  ├──────────────────────────────────────────────────┤ │
│  │ 2.2 Self-Attention (GQA + Partial RoPE)         │ │
│  │   ├─ QKV 融合投影 (Linear)                      │ │  PyTorch
│  │   ├─ [MoE版] QK-Norm (per-head RMSNorm)        │ │  Custom CUDA
│  │   ├─ Partial RoPE (仅 50% head_dim)            │ │  Custom CUDA
│  │   ├─ KV Cache 更新                              │ │  Custom CUDA
│  │   ├─ Attention Kernel                           │ │  C++/CUDA
│  │   └─ Output Projection (Linear)                 │ │  PyTorch
│  ├──────────────────────────────────────────────────┤ │
│  │ 2.2b [Dense版] post_self_attn_layernorm         │ │  Custom CUDA
│  ├──────────────────────────────────────────────────┤ │
│  │ 2.3 RMSNorm (post_attention_layernorm)          │ │  Custom CUDA
│  ├──────────────────────────────────────────────────┤ │
│  │ 2.4 MLP / MoE                                   │ │
│  │   [Dense] gate_up → SiLU → down                 │ │  PyTorch + CUDA
│  │   [MoE] grouped_topk Router → FusedMoE + Shared │ │  见详细说明
│  ├──────────────────────────────────────────────────┤ │
│  │ 2.4b [Dense版] post_mlp_layernorm               │ │  Custom CUDA
│  └──────────────────────────────────────────────────┘ │
└────────────┬──────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. Final RMSNorm       │  Custom CUDA
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. LM Head (Linear)    │  PyTorch (ParallelLMHead)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. Logits Processing   │  PyTorch
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  6. Sampling            │  PyTorch
└────────────┬────────────┘
             │
             ▼
output_token_id
```

---

## 详细算子列表

### 1. Embedding Lookup

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `VocabParallelEmbedding` | PyTorch (`nn.Embedding` + TP分片) | [vocab_parallel_embedding.py](vllm/model_executor/layers/vocab_parallel_embedding.py) |

---

### 2. Decoder Layer（重复 N 层）

#### 2.1 Input LayerNorm (RMSNorm)

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `RMSNorm` | **Custom CUDA** (`ops.fused_add_rms_norm`) | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |

---

#### 2.2 Self-Attention（GQA + Partial RoPE）

##### 2.2.1 QKV 融合投影

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `qkv_proj` | PyTorch (`QKVParallelLinear`) | [glm4.py:104](vllm/model_executor/models/glm4.py#L104) / [glm4_moe.py:257](vllm/model_executor/models/glm4_moe.py#L257) |

##### 2.2.2 QK-Norm（MoE 版本，可选）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `q_norm` (per-head RMSNorm) | **Custom CUDA** | [glm4_moe.py:292](vllm/model_executor/models/glm4_moe.py#L292) |
| `k_norm` (per-head RMSNorm) | **Custom CUDA** | [glm4_moe.py:293](vllm/model_executor/models/glm4_moe.py#L293) |

说明：仅在 `use_qk_norm=True` 时启用（GLM-4.5+ MoE 版本）。Dense 版本不使用 QK-Norm。

##### 2.2.3 RoPE 旋转位置编码（Partial, Non-NeoX）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `rotary_embedding` | **Custom CUDA** | [csrc/pos_encoding_kernels.cu](csrc/pos_encoding_kernels.cu) |

说明：GLM 的 RoPE 有两个关键区别：
- `partial_rotary_factor = 0.5`：仅对 head_dim 的前 50% 应用旋转
- `is_neox_style = False`：使用交错式（interleaved）而非 NeoX 风格的分半式

##### 2.2.4 KV Cache 更新

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `reshape_and_cache_flash` | **Custom CUDA** | [csrc/cache_kernels.cu](csrc/cache_kernels.cu) |

##### 2.2.5 Attention Kernel

| 算子 | 实现方式 | 使用场景 | 文件路径 |
|------|---------|---------|---------|
| `flash_attn_varlen_func` | **C++/CUDA** (Flash Attention) | Prefill + Decode | [flash_attn.py](vllm/v1/attention/backends/flash_attn.py) |
| FlashInfer | **C++/CUDA** (外部库) | 可选 backend | [flashinfer.py](vllm/v1/attention/backends/flashinfer.py) |

##### 2.2.6 Output Projection

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `o_proj` | PyTorch (`RowParallelLinear`, bias=False) | [glm4.py:113](vllm/model_executor/models/glm4.py#L113) / [glm4_moe.py:267](vllm/model_executor/models/glm4_moe.py#L267) |

---

#### 2.2b Post-Self-Attention LayerNorm（仅 Dense 版本）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `post_self_attn_layernorm` | **Custom CUDA** (RMSNorm, 无残差融合) | [glm4.py:189](vllm/model_executor/models/glm4.py#L189) |

说明：这是 GLM-4 Dense 的独特设计 — 在 attention 输出后、残差连接前额外加一个 RMSNorm。MoE 版本没有这个。

---

#### 2.3 Post-Attention LayerNorm (RMSNorm)

同 2.1，Custom CUDA fused_add_rms_norm（融合残差加法）。

---

#### 2.4 MLP / MoE

##### Dense 版本 (GLM-4)

GLM-4 Dense 直接使用 LLaMA MLP（`from .llama import LlamaMLP as Glm4MLP`）。

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `gate_up_proj` | PyTorch (`MergedColumnParallelLinear`) | [llama.py](vllm/model_executor/models/llama.py) (LlamaMLP) |
| `SiluAndMul` | **Custom CUDA** (`torch.ops._C.silu_and_mul`) | [csrc/activation_kernels.cu](csrc/activation_kernels.cu) |
| `down_proj` | PyTorch (`RowParallelLinear`) | [llama.py](vllm/model_executor/models/llama.py) (LlamaMLP) |

##### MoE 版本 (GLM-4.5/4.6/4.7)

GLM MoE 使用 **sigmoid + grouped_topk** 路由，与 DeepSeek V3 路由机制相同。

**Router:**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `gate` (router linear, FP32) | PyTorch (`nn.Linear`, dtype=float32) | [glm4_moe.py:146](vllm/model_executor/models/glm4_moe.py#L146) |
| `e_score_correction_bias` | PyTorch (learned bias) | [glm4_moe.py:152](vllm/model_executor/models/glm4_moe.py#L152) |
| `grouped_topk` (sigmoid + group select) | **Custom CUDA** | [csrc/moe/grouped_topk_kernels.cu](csrc/moe/grouped_topk_kernels.cu) |

说明：Router 计算过程：
1. 线性投影得到 expert scores（FP32）
2. 对 scores 做 sigmoid
3. 加上 `e_score_correction_bias`（学习的负载均衡偏置）
4. 分组 top-k 选择 experts

**Expert GEMM:**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `FusedMoE` (routed experts) | **Triton** (`fused_moe_kernel`) | [fused_moe.py](vllm/model_executor/layers/fused_moe/fused_moe.py) |
| (FP8 场景) | **C++/CUDA** (DeepGemm / CUTLASS) | [fused_moe/experts/](vllm/model_executor/layers/fused_moe/experts/) |

**Shared Expert:**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `shared_experts` (gate_up + SiLU + down) | PyTorch + Custom CUDA | [glm4_moe.py:173](vllm/model_executor/models/glm4_moe.py#L173) |

说明：输出 = routed_expert_output * routed_scaling_factor + shared_expert_output

---

#### 2.4b Post-MLP LayerNorm（仅 Dense 版本）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `post_mlp_layernorm` | **Custom CUDA** (RMSNorm, 无残差融合) | [glm4.py:192](vllm/model_executor/models/glm4.py#L192) |

说明：同 2.2b，GLM-4 Dense 在 MLP 输出后、残差连接前也加了一个 RMSNorm。

---

### 3. Final RMSNorm

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `model.norm` | **Custom CUDA** | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |

---

### 4. LM Head

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `lm_head` | PyTorch (`ParallelLMHead`) | [linear.py](vllm/model_executor/layers/linear.py) |

说明：GLM-4 支持 `tie_word_embeddings`（共享 embedding 和 lm_head 权重）。

---

### 5. Logits Processing

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `LogitsProcessor` | PyTorch（gather + 分布式合并） | [logits_processor.py](vllm/model_executor/layers/logits_processor.py) |

---

### 6. Sampling

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| Temperature / Top-K / Top-P / Multinomial | PyTorch | [sampler.py](vllm/v1/sample/sampler.py) |

---

## 算子实现分类汇总

### Custom CUDA Kernels

| 算子名称 | 功能 | 源文件 |
|---------|------|--------|
| `fused_add_rms_norm` | RMSNorm + 残差加法融合 | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |
| `rms_norm` | 独立 RMSNorm（QK-Norm, post-layernorm） | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |
| `silu_and_mul` | SiLU 激活 + 逐元素乘法 | [csrc/activation_kernels.cu](csrc/activation_kernels.cu) |
| `rotary_embedding` | Partial RoPE (interleaved) | [csrc/pos_encoding_kernels.cu](csrc/pos_encoding_kernels.cu) |
| `reshape_and_cache_flash` | KV Cache 写入 | [csrc/cache_kernels.cu](csrc/cache_kernels.cu) |
| `grouped_topk` | MoE sigmoid grouped top-k 路由 | [csrc/moe/grouped_topk_kernels.cu](csrc/moe/grouped_topk_kernels.cu) |

### Triton Kernels

| 算子名称 | 功能 | 源文件 |
|---------|------|--------|
| `fused_moe_kernel` | MoE Expert 融合 GEMM + 激活 | [fused_moe.py](vllm/model_executor/layers/fused_moe/fused_moe.py) |

### 外部 C++/CUDA 库

| 库名称 | 功能 |
|--------|------|
| Flash Attention | 注意力计算（prefill + decode） |
| FlashInfer | 可选 attention backend |
| DeepGemm / CUTLASS | FP8 MoE Expert GEMM |

### Pure PyTorch

| 操作 | 功能 |
|------|------|
| `nn.Embedding` | Embedding lookup |
| `QKVParallelLinear` / `RowParallelLinear` | 所有投影层 |
| `nn.Linear` (FP32, router) | MoE 路由器 |
| `F.sigmoid` | 路由评分激活 |
| `torch.topk / multinomial` | 采样 |

---

## 与 DeepSeek 的对比

| 特性 | GLM-4 MoE | DeepSeek V3 |
|------|-----------|-------------|
| Attention | 标准 GQA + Partial RoPE | MLA (低秩压缩) |
| KV Cache | 标准 KV Cache | 压缩 latent cache |
| Router 激活 | sigmoid | softplus + sqrt |
| Router 路由 | grouped_topk | grouped_topk |
| e_score_correction_bias | ✅ 有 | ✅ 有 |
| Shared Expert | ✅ 有 | ✅ 有 |
| routed_scaling_factor | ✅ 有 | ✅ 有 |
| MLA 专用算子 | ❌ 无 | concat_mla_q, merge_attn_states |
| FusedMoE | ✅ 标准 FusedMoE | ✅ 标准 FusedMoE |

GLM-4 MoE 的 Router 设计与 DeepSeek V3 高度相似（sigmoid + grouped_topk + bias correction），但注意力使用标准 GQA 而非 MLA，因此整体算子种类更少。
