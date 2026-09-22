# LLaMA 模型算子流程（从 Embedding 到 Token 采样）

> 以 LLaMA 2/3 为例，梳理 vLLM 中从输入 token id 到输出采样 token 的完整算子调用链。
> 模型入口：`vllm/model_executor/models/llama.py` → `LlamaForCausalLM`

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
┌─────────────────────────────────────────────┐
│  2. DecoderLayer × N (循环)                  │
│  ┌────────────────────────────────────────┐ │
│  │ 2.1 RMSNorm (input_layernorm)         │ │  Custom CUDA
│  ├────────────────────────────────────────┤ │
│  │ 2.2 Self-Attention (标准 MHA/GQA)     │ │
│  │   ├─ QKV 融合投影 (Linear)            │ │  PyTorch (cuBLAS)
│  │   ├─ RoPE 旋转位置编码                │ │  Custom CUDA
│  │   ├─ KV Cache 更新                    │ │  Custom CUDA
│  │   ├─ Attention Kernel                 │ │  C++/CUDA (Flash Attention)
│  │   └─ Output Projection (Linear)       │ │  PyTorch (cuBLAS)
│  ├────────────────────────────────────────┤ │
│  │ 2.3 RMSNorm (post_attention_layernorm)│ │  Custom CUDA
│  ├────────────────────────────────────────┤ │
│  │ 2.4 MLP (SwiGLU)                      │ │
│  │   ├─ gate_up_proj (融合 Linear)       │ │  PyTorch (cuBLAS)
│  │   ├─ SiluAndMul 激活                  │ │  Custom CUDA
│  │   └─ down_proj (Linear)               │ │  PyTorch (cuBLAS)
│  └────────────────────────────────────────┘ │
└────────────┬────────────────────────────────┘
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
│  6. Sampling            │  PyTorch (top-k, top-p, temperature)
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

说明：将 input_ids 映射为 hidden_states，维度为 `[num_tokens, hidden_size]`。支持张量并行分片。

---

### 2. Decoder Layer（重复 N 层）

每一层 `LlamaDecoderLayer`（L253）的前向流程如下：

#### 2.1 Input LayerNorm (RMSNorm)

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `RMSNorm` | **Custom CUDA** (`ops.fused_add_rms_norm`) | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |
| (fallback) `forward_native` | PyTorch (手动实现) | [layernorm.py](vllm/model_executor/layers/layernorm.py) |

说明：计算 `x → w * x / sqrt(E[x²] + eps)`，同时融合残差加法。CUDA 实现注册为 `torch.ops._C.fused_add_rms_norm`。

#### 2.2 Self-Attention（标准 MHA / GQA）

LLaMA 使用标准的 Multi-Head Attention（LLaMA 1）或 Grouped-Query Attention（LLaMA 2/3）。

##### 2.2.1 QKV 融合投影

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `qkv_proj` | PyTorch (`QKVParallelLinear`) | [llama.py:164](vllm/model_executor/models/llama.py#L164) |

说明：将 Q/K/V 三个投影融合为一次 GEMM。输出 split 为 `[q_size, kv_size, kv_size]`。

##### 2.2.2 RoPE 旋转位置编码

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `rotary_embedding` | **Custom CUDA** (`torch.ops._C.rotary_embedding`) | [csrc/pos_encoding_kernels.cu](csrc/pos_encoding_kernels.cu) |
| (备选) `apply_rotary_emb` | **C++/CUDA** (vllm_flash_attn 内置) | `vllm/vllm_flash_attn/layers/rotary.py` |
| (fallback) `forward_native` | PyTorch (rotate_half + cos/sin) | [common.py](vllm/model_executor/layers/rotary_embedding/common.py) |

说明：对 Q 和 K 同时应用 RoPE。LLaMA 使用 NeoX-style（非交错）旋转。LLaMA 3 使用扩展的 RoPE (500k base frequency)。

##### 2.2.3 KV Cache 更新

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `reshape_and_cache_flash` | **Custom CUDA** | [csrc/cache_kernels.cu](csrc/cache_kernels.cu) |

说明：将当前 step 的 K/V 写入 paged KV cache 中对应的 slot。

##### 2.2.4 Attention Kernel（核心计算）

| 算子 | 实现方式 | 使用场景 | 文件路径 |
|------|---------|---------|---------|
| `flash_attn_varlen_func` | **C++/CUDA** (Flash Attention 2/3) | Prefill | [flash_attn.py](vllm/v1/attention/backends/flash_attn.py) |
| Flash Attention (paged) | **C++/CUDA** (Flash Attention) | Decode | 同上 |
| FlashInfer | **C++/CUDA** (外部库) | 可选 backend | [flashinfer.py](vllm/v1/attention/backends/flashinfer.py) |
| Triton Attention | **Triton** (`@triton.jit`) | 可选 backend | [triton_attn.py](vllm/v1/attention/backends/triton_attn.py) |

说明：默认使用 Flash Attention backend。支持 Paged KV Cache，由 scheduler 管理 page table。Decode 阶段直接在 paged cache 上做注意力。

##### 2.2.5 Output Projection

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `o_proj` | PyTorch (`RowParallelLinear`) | [llama.py:174](vllm/model_executor/models/llama.py#L174) |

---

#### 2.3 Post-Attention LayerNorm (RMSNorm)

同 2.1，使用 `fused_add_rms_norm` Custom CUDA 算子，融合残差加法。

#### 2.4 MLP (SwiGLU)

LLaMA 使用标准的 Dense MLP（无 MoE），激活函数为 SwiGLU。

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `gate_up_proj` | PyTorch (`MergedColumnParallelLinear`) | [llama.py:94](vllm/model_executor/models/llama.py#L94) |
| `SiluAndMul` | **Custom CUDA** (`torch.ops._C.silu_and_mul`) | [csrc/activation_kernels.cu](csrc/activation_kernels.cu) |
| `down_proj` | PyTorch (`RowParallelLinear`) | [llama.py:102](vllm/model_executor/models/llama.py#L102) |

说明：`gate_up_proj` 将 gate_proj 和 up_proj 融合为一次 GEMM（输出维度为 `2 * intermediate_size`）。`SiluAndMul` 对前半部分做 SiLU 后与后半部分逐元素相乘。

---

### 3. Final RMSNorm

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `model.norm` | **Custom CUDA** (`fused_add_rms_norm`) | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |

---

### 4. LM Head

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `lm_head` | PyTorch (`ParallelLMHead` → `nn.Linear`) | [linear.py](vllm/model_executor/layers/linear.py) |

说明：将最终的 hidden_states 投影到 vocab_size 维度，得到每个 token 的 logits。

---

### 5. Logits Processing

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `LogitsProcessor` | PyTorch（gather + 分布式合并） | [logits_processor.py](vllm/model_executor/layers/logits_processor.py) |

说明：对 lm_head 的输出做 TP gather（如果使用张量并行），得到完整的 vocab logits。

---

### 6. Sampling（采样）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| Temperature scaling | PyTorch (`logits / temperature`) | [sampler.py](vllm/v1/sample/sampler.py) |
| Top-k filtering | PyTorch (`torch.topk`) | [sampler.py](vllm/v1/sample/sampler.py) |
| Top-p (nucleus) filtering | PyTorch (`torch.sort` + `cumsum`) | [sampler.py](vllm/v1/sample/sampler.py) |
| Multinomial sampling | PyTorch (`torch.multinomial`) | [sampler.py](vllm/v1/sample/sampler.py) |
| Greedy (argmax) | PyTorch (`torch.argmax`) | [sampler.py](vllm/v1/sample/sampler.py) |

---

## 算子实现分类汇总

### Custom CUDA Kernels（C++/CUDA）

| 算子名称 | 功能 | 源文件 |
|---------|------|--------|
| `fused_add_rms_norm` | RMSNorm + 残差加法融合 | [csrc/layernorm_kernels.cu](csrc/layernorm_kernels.cu) |
| `silu_and_mul` | SiLU 激活 + 逐元素乘法 | [csrc/activation_kernels.cu](csrc/activation_kernels.cu) |
| `rotary_embedding` | RoPE 位置编码 | [csrc/pos_encoding_kernels.cu](csrc/pos_encoding_kernels.cu) |
| `reshape_and_cache_flash` | KV Cache 写入 | [csrc/cache_kernels.cu](csrc/cache_kernels.cu) |

### 外部 C++/CUDA 库

| 库名称 | 功能 | 调用位置 |
|--------|------|---------|
| vllm_flash_attn | Flash Attention (prefill + decode) | [flash_attn.py](vllm/v1/attention/backends/flash_attn.py) |
| FlashInfer | 可选 attention backend | [flashinfer.py](vllm/v1/attention/backends/flashinfer.py) |

### Triton Kernels

| 算子名称 | 功能 | 源文件 |
|---------|------|--------|
| Triton Attention | 可选 attention backend | [triton_attn.py](vllm/v1/attention/backends/triton_attn.py) |

### Pure PyTorch

| 算子/操作 | 功能 | 备注 |
|----------|------|------|
| `nn.Embedding` | Embedding lookup | VocabParallelEmbedding |
| `nn.Linear` (QKVParallel/MergedColumn/RowParallel) | 所有投影层 | 矩阵乘法由 cuBLAS 执行 |
| `torch.topk / sort / multinomial` | 采样相关操作 | Sampler |
| `LogitsProcessor` | Logits 后处理 | gather + reduce |

---

## 与 DeepSeek 的关键差异

| 特性 | LLaMA | DeepSeek V2/V3 |
|------|-------|---------------|
| Attention 类型 | 标准 MHA / GQA | MLA（多头潜在注意力） |
| KV Cache 内容 | 完整 K/V 向量 | 压缩潜在向量 + rope_key |
| MLP 类型 | Dense (SwiGLU) | MoE（256 experts, grouped top-k routing） |
| 专用 CUDA kernel 数量 | 4 个 | 10+ 个（含 MoE 和 MLA 专用） |
| Triton kernel 使用 | 仅 attention 可选 | MoE GEMM 核心路径 |
| 模型复杂度 | 简单直接 | 大量融合算子和专用优化 |

---

## 关键路径性能热点

按计算量排序（从大到小）：

1. **QKV/O/Gate/Up/Down 投影 Linear** (cuBLAS GEMM) — 占总计算量 ~60-70%
2. **Attention Kernel** (Flash Attention) — 占总计算量 ~20-25%
3. **RMSNorm / RoPE / SiLU** (Custom CUDA) — 占总计算量 <5%
4. **Sampling** (PyTorch) — 可忽略

说明：LLaMA 是 Dense 模型，所有 token 都经过完整 MLP，因此 Linear 层的 GEMM 是绝对主导。相比 DeepSeek 的 MoE 结构（仅激活部分 expert），LLaMA 的计算更集中在标准矩阵乘法上。

---

## 备注

- LLaMA 1 使用标准 MHA（num_kv_heads = num_heads），LLaMA 2/3 使用 GQA（num_kv_heads < num_heads）
- LLaMA 3 引入了 500k base frequency 的扩展 RoPE
- 以上分析基于 NVIDIA GPU (CUDA) 路径
- 实际运行时的 attention backend 取决于 `VLLM_ATTENTION_BACKEND` 环境变量
- FP8 量化模式下，Linear 层会使用量化 GEMM 而非标准 cuBLAS
