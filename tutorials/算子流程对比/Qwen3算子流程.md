# Qwen3 模型算子流程（从 Embedding 到 Token 采样）

> 以 Qwen3（Dense）和 Qwen3-MoE 为例，梳理 vLLM 中完整算子调用链。
> 模型入口：
> - Dense: `vllm/model_executor/models/qwen3.py` → `Qwen3ForCausalLM`
> - MoE: `vllm/model_executor/models/qwen3_moe.py` → `Qwen3MoeForCausalLM`

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
┌──────────────────────────────────────────────────┐
│  2. DecoderLayer × N (循环)                       │
│  ┌─────────────────────────────────────────────┐ │
│  │ 2.1 RMSNorm (input_layernorm)              │ │  Custom CUDA
│  ├─────────────────────────────────────────────┤ │
│  │ 2.2 Self-Attention (GQA + QK-Norm)         │ │
│  │   ├─ QKV 融合投影 (Linear)                 │ │  PyTorch (cuBLAS)
│  │   ├─ Q-Norm / K-Norm (per-head RMSNorm)   │ │  Custom CUDA
│  │   ├─ RoPE 旋转位置编码                     │ │  Custom CUDA
│  │   ├─ KV Cache 更新                         │ │  Custom CUDA
│  │   ├─ Attention Kernel                      │ │  C++/CUDA (Flash Attention)
│  │   └─ Output Projection (Linear)            │ │  PyTorch (cuBLAS)
│  ├─────────────────────────────────────────────┤ │
│  │ 2.3 RMSNorm (post_attention_layernorm)     │ │  Custom CUDA
│  ├─────────────────────────────────────────────┤ │
│  │ 2.4 MLP / MoE                              │ │
│  │   [Dense] gate_up → SiLU → down            │ │  PyTorch + Custom CUDA
│  │   [MoE]  Router → FusedMoE + SharedExpert  │ │  见下文详细说明
│  └─────────────────────────────────────────────┘ │
└────────────┬─────────────────────────────────────┘
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

#### 2.2 Self-Attention（GQA + QK-Norm）

Qwen3 的注意力机制与 LLaMA 类似（GQA），但增加了 **QK-Norm**（对 Q 和 K 分别做 per-head RMSNorm）。

##### 2.2.1 QKV 融合投影

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `qkv_proj` | PyTorch (`QKVParallelLinear`) | [qwen3.py:98](vllm/model_executor/models/qwen3.py#L98) / [qwen3_moe.py:300](vllm/model_executor/models/qwen3_moe.py#L300) |

##### 2.2.2 QK-Norm（Qwen3 特有）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `q_norm` (per-head RMSNorm) | **Custom CUDA** (`ops.rms_norm`) | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |
| `k_norm` (per-head RMSNorm) | **Custom CUDA** (`ops.rms_norm`) | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |

说明：对每个 attention head 独立做 RMSNorm，维度为 `head_dim`。这是 Qwen3 相对 Qwen2 的关键改进。

##### 2.2.3 RoPE 旋转位置编码

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `rotary_embedding` | **Custom CUDA** / C++ (vllm_flash_attn) | [csrc/pos_encoding_kernels.cu](csrc/pos_encoding_kernels.cu) |

说明：Qwen3 默认 rope_theta=1000000（100万），支持长上下文。

##### 2.2.4 KV Cache 更新

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `reshape_and_cache_flash` | **Custom CUDA** | [csrc/cache_kernels.cu](csrc/cache_kernels.cu) |

##### 2.2.5 Attention Kernel

| 算子 | 实现方式 | 使用场景 | 文件路径 |
|------|---------|---------|---------|
| `flash_attn_varlen_func` | **C++/CUDA** (Flash Attention) | Prefill + Decode | [flash_attn.py](vllm/v1/attention/backends/flash_attn.py) |
| FlashInfer | **C++/CUDA** (外部库) | 可选 backend | [flashinfer.py](vllm/v1/attention/backends/flashinfer.py) |
| Triton Attention | **Triton** | 可选 backend | [triton_attn.py](vllm/v1/attention/backends/triton_attn.py) |

##### 2.2.6 Output Projection

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `o_proj` | PyTorch (`RowParallelLinear`) | [qwen3.py:107](vllm/model_executor/models/qwen3.py#L107) / [qwen3_moe.py:310](vllm/model_executor/models/qwen3_moe.py#L310) |

---

#### 2.3 Post-Attention LayerNorm (RMSNorm)

同 2.1，Custom CUDA fused_add_rms_norm。

---

#### 2.4 MLP / MoE

##### Dense 版本 (Qwen3)

Qwen3 Dense 直接复用 Qwen2 的 MLP（`from .qwen2 import Qwen2MLP as Qwen3MLP`）。

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `gate_up_proj` | PyTorch (`MergedColumnParallelLinear`) | [qwen2.py (Qwen2MLP)](vllm/model_executor/models/qwen2.py) |
| `SiluAndMul` | **Custom CUDA** (`torch.ops._C.silu_and_mul`) | [csrc/activation_kernels.cu](csrc/activation_kernels.cu) |
| `down_proj` | PyTorch (`RowParallelLinear`) | [qwen2.py (Qwen2MLP)](vllm/model_executor/models/qwen2.py) |

##### MoE 版本 (Qwen3-MoE)

Qwen3-MoE 使用标准 Top-K 路由（非 grouped_topk），部分层为 Dense，部分层为 MoE（由 `decoder_sparse_step` 控制）。

**Router:**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `gate` (router linear) | PyTorch (`ReplicatedLinear`) | [qwen3_moe.py:179](vllm/model_executor/models/qwen3_moe.py#L179) |
| topk + softmax | 由 `FusedMoE` 内部处理 | [layer.py](vllm/model_executor/layers/fused_moe/layer.py) |

说明：Qwen3-MoE 使用标准 softmax top-k（不使用 grouped_topk），通过 `FusedMoE` 的内部 router 执行。

**Expert GEMM:**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `FusedMoE` (experts) | **Triton** (`fused_moe_kernel`) | [fused_moe.py](vllm/model_executor/layers/fused_moe/fused_moe.py) |
| (FP8 场景) | **C++/CUDA** (DeepGemm / CUTLASS) | [deep_gemm_moe.py](vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py) |

**Shared Expert（可选）:**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `shared_expert` (gate_up + SiLU + down) | PyTorch + Custom CUDA | [qwen3_moe.py:198](vllm/model_executor/models/qwen3_moe.py#L198) |
| `shared_expert_gate` (sigmoid gate) | PyTorch (`F.sigmoid`) | [qwen3_moe.py:131](vllm/model_executor/models/qwen3_moe.py#L131) |

说明：Shared Expert 的输出乘以一个 sigmoid gate 后加到 routed expert 输出上。

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

---

### 5. Logits Processing

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `LogitsProcessor` | PyTorch | [logits_processor.py](vllm/model_executor/layers/logits_processor.py) |

---

### 6. Sampling

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| Temperature / Top-K / Top-P / Multinomial | PyTorch | [sampler.py](vllm/v1/sample/sampler.py) |

---

## 与 LLaMA / DeepSeek 的关键差异

| 特性 | LLaMA | Qwen3 (Dense) | Qwen3 MoE | DeepSeek V3 |
|------|-------|---------------|------------|-------------|
| Attention | GQA | GQA + **QK-Norm** | GQA + **QK-Norm** | MLA (低秩压缩) |
| MLP | Dense (SwiGLU) | Dense (SwiGLU) | MoE (标准 Top-K) | MoE (grouped Top-K) |
| Router 类型 | 无 | 无 | softmax + top-k | **grouped_topk** + softplus |
| Shared Expert | 无 | 无 | 有 (sigmoid gate) | 有 (直接加) |
| QK-Norm | 无 | **有** | **有** | 无 |
| RoPE theta | 500k (LLaMA3) | **1M** | 1M | YaRN 扩展 |

---

## 算子实现分类汇总

| 类别 | 算子 | 实现方式 |
|------|------|---------|
| Custom CUDA | RMSNorm (含 fused_add), SiluAndMul, RoPE, KV Cache | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu), [csrc/activation_kernels.cu](csrc/activation_kernels.cu), [csrc/pos_encoding_kernels.cu](csrc/pos_encoding_kernels.cu), [csrc/cache_kernels.cu](csrc/cache_kernels.cu) |
| C++/CUDA 外部库 | Flash Attention, FlashInfer | [flash_attn.py](vllm/v1/attention/backends/flash_attn.py) / [flashinfer.py](vllm/v1/attention/backends/flashinfer.py) |
| Triton | FusedMoE Expert GEMM | [fused_moe.py](vllm/model_executor/layers/fused_moe/fused_moe.py) |
| PyTorch | Embedding, Linear (QKV/O/Gate/Up/Down), Logits, Sampling | cuBLAS GEMM |
