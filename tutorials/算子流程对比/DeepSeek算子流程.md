# DeepSeek 模型算子流程（从 Embedding 到 Token 采样）

> 以 DeepSeek V2/V3 为例，梳理 vLLM 中从输入 token id 到输出采样 token 的完整算子调用链。
> 模型入口：`vllm/model_executor/models/deepseek_v2.py` → `DeepseekV2ForCausalLM`

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
│  │ 2.2 MLA Attention                     │ │
│  │   ├─ Q/KV 低秩投影 (Linear)           │ │  PyTorch
│  │   ├─ Q/KV LayerNorm                   │ │  Custom CUDA
│  │   ├─ Q_b / KV_b 投影 (Linear)         │ │  PyTorch
│  │   ├─ RoPE 旋转位置编码                │ │  C++/CUDA (vllm_flash_attn)
│  │   ├─ concat_mla_q (Q拼接)             │ │  Custom CUDA
│  │   ├─ KV Cache 更新                    │ │  Custom CUDA
│  │   ├─ Attention Kernel                 │ │  C++/CUDA 或 Triton
│  │   └─ Output Projection (Linear)       │ │  PyTorch
│  ├────────────────────────────────────────┤ │
│  │ 2.3 RMSNorm (post_attention_layernorm)│ │  Custom CUDA
│  ├────────────────────────────────────────┤ │
│  │ 2.4 MLP / MoE                         │ │
│  │   Dense层: gate_up_proj → SiLU → down │ │  PyTorch + Custom CUDA(SiLU)
│  │   MoE层:                              │ │
│  │     ├─ Router Gate (Linear)            │ │  PyTorch
│  │     ├─ grouped_topk (路由选择)         │ │  Custom CUDA
│  │     ├─ FusedMoE Expert GEMM           │ │  Triton / DeepGemm / CUTLASS
│  │     ├─ MoE Align/Permute              │ │  Custom CUDA
│  │     └─ Shared Expert (Linear+SiLU)    │ │  PyTorch + Custom CUDA
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
| `VocabParallelEmbedding` | PyTorch (`nn.Embedding` + TP分片) | [`vocab_parallel_embedding.py`](vllm/model_executor/layers/vocab_parallel_embedding.py) |

说明：将 input_ids 映射为 hidden_states，维度为 `[num_tokens, hidden_size]`。支持张量并行分片。

---

### 2. Decoder Layer（重复 N 层）

每一层 `DeepseekV2DecoderLayer` 的前向流程如下：

#### 2.1 Input LayerNorm (RMSNorm)

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `RMSNorm` | **Custom CUDA** (`ops.fused_add_rms_norm`) | [`csrc/layernorm_kernels.cu`](csrc/layernorm_kernels.cu) |
| (fallback) `forward_native` | PyTorch (手动实现) | [`layernorm.py`](vllm/model_executor/layers/layernorm.py) |

说明：计算 `x → w * x / sqrt(E[x²] + eps)`，同时融合残差加法（fused_add_rms_norm）。CUDA 实现注册为 `torch.ops._C.fused_add_rms_norm`。

#### 2.2 MLA (Multi-head Latent Attention)

DeepSeek V2/V3 使用 MLA 注意力机制，核心思想是对 Q/KV 做低秩压缩以减少 KV Cache。

##### 2.2.1 Q/KV 低秩下投影

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `fused_qkv_a_proj` (Q_a + KV_a) | PyTorch (`ColumnParallelLinear`) | [`deepseek_v2.py:905`](vllm/model_executor/models/deepseek_v2.py#L905) |

说明：将 hidden_states 投影到低维潜在空间。Q → `q_lora_rank`，KV → `kv_lora_rank + qk_rope_head_dim`。

##### 2.2.2 Q/KV LayerNorm

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `q_a_layernorm` | **Custom CUDA** (同 RMSNorm) | [`csrc/layernorm_kernels.cu`](csrc/layernorm_kernels.cu) |
| `kv_a_layernorm` | **Custom CUDA** (同 RMSNorm) | [`csrc/layernorm_kernels.cu`](csrc/layernorm_kernels.cu) |

##### 2.2.3 Q/KV 上投影

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `q_b_proj` | PyTorch (`ColumnParallelLinear`) | [`deepseek_v2.py:922`](vllm/model_executor/models/deepseek_v2.py#L922) |
| `kv_b_proj` | PyTorch (`ColumnParallelLinear`) | [`deepseek_v2.py:938`](vllm/model_executor/models/deepseek_v2.py#L938) |

说明：将低维潜在表示投影回高维。Q → `num_heads * qk_head_dim`，KV → `num_heads * (qk_nope_head_dim + v_head_dim)`。

##### 2.2.4 RoPE 旋转位置编码

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `apply_rotary_emb` | **C++/CUDA** (vllm_flash_attn 内置) | `vllm/vllm_flash_attn/layers/rotary.py` → C++ kernel |
| (备选) `rotary_embedding` | **Custom CUDA** | [`csrc/pos_encoding_kernels.cu`](csrc/pos_encoding_kernels.cu) |

说明：仅对 `qk_rope_head_dim` 部分的 Q 和 K 应用 RoPE。DeepSeek 使用 YaRN 扩展的 RoPE。

##### 2.2.5 Q 拼接 (concat_mla_q)

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `concat_mla_q` | **Custom CUDA** | [`csrc/concat_mla_q.cuh`](csrc/concat_mla_q.cuh) |

说明：将 `q_nope`（非位置编码部分）和 `q_pe`（位置编码部分）拼接为完整的 Q tensor。

##### 2.2.6 KV Cache 更新

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `unified_mla_kv_cache_update` | **Custom CUDA** (torch.library 注册) | `vllm/v1/attention/backends/mla/common.py` |

说明：MLA 的 KV Cache 存储压缩后的潜在向量 + rope_key，而非完整的 K/V。

##### 2.2.7 Attention Kernel（核心计算）

| 算子 | 实现方式 | 使用场景 | 文件路径 |
|------|---------|---------|---------|
| `flash_attn_varlen_func` | **C++/CUDA** (Flash Attention) | Prefill | [`flashattn_mla.py`](vllm/v1/attention/backends/mla/flashattn_mla.py) |
| FlashMLA decode | **C++/CUDA** (外部库) | Decode | [`flashmla.py`](vllm/v1/attention/ops/flashmla.py) |
| FlashInfer MLA | **C++/CUDA** (外部库) | Prefill+Decode | [`flashinfer_mla.py`](vllm/v1/attention/backends/mla/flashinfer_mla.py) |
| `triton_decode_attention` | **Triton** (`@triton.jit`) | Decode | [`triton_decode_attention.py`](vllm/v1/attention/ops/triton_decode_attention.py) |
| `sm100_cutlass_mla_decode` | **C++/CUDA (CUTLASS)** | Decode (Blackwell) | CUTLASS template kernel |
| `merge_attn_states` | **Custom CUDA** | 合并分块注意力 | [`csrc/attention/merge_attn_states.cu`](csrc/attention/merge_attn_states.cu) |

##### 2.2.8 Output Projection

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `o_proj` | PyTorch (`RowParallelLinear`) | [`deepseek_v2.py:945`](vllm/model_executor/models/deepseek_v2.py#L945) |

---

#### 2.3 Post-Attention LayerNorm (RMSNorm)

同 2.1，使用 `fused_add_rms_norm` Custom CUDA 算子。

#### 2.4 MLP / MoE 层

DeepSeek V2/V3 的前 `first_k_dense_replace` 层使用普通 Dense MLP，之后的层使用 MoE。

##### 2.4.1 Dense MLP（前几层）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `gate_up_proj` | PyTorch (`MergedColumnParallelLinear`) | [`deepseek_v2.py:212`](vllm/model_executor/models/deepseek_v2.py#L212) |
| `SiluAndMul` | **Custom CUDA** (`torch.ops._C.silu_and_mul`) | [`csrc/activation_kernels.cu`](csrc/activation_kernels.cu) |
| `down_proj` | PyTorch (`RowParallelLinear`) | [`deepseek_v2.py:220`](vllm/model_executor/models/deepseek_v2.py#L220) |

##### 2.4.2 MoE 层（主体层）

**Router（路由计算）：**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `gate` (router linear) | PyTorch (`nn.Linear`) | [`deepseek_v2.py:270`](vllm/model_executor/models/deepseek_v2.py#L270) |
| `grouped_topk` | **Custom CUDA** | [`csrc/moe/grouped_topk_kernels.cu`](csrc/moe/grouped_topk_kernels.cu) |
| `topk_softplus_sqrt` | **Custom CUDA** | [`csrc/moe/topk_softplus_sqrt_kernels.cu`](csrc/moe/topk_softplus_sqrt_kernels.cu) |
| `dsv3_router_gemm` | **Custom CUDA** | [`csrc/moe/dsv3_router_gemm_entry.cu`](csrc/moe/dsv3_router_gemm_entry.cu) |

说明：DeepSeek V3 使用 grouped top-k 路由（256 experts 分为 n_group 组，每组选 topk_group 个），评分函数为 softplus + sqrt。

**Expert 计算（GEMM）：**

| 算子 | 实现方式 | 适用场景 | 文件路径 |
|------|---------|---------|---------|
| `fused_moe_kernel` | **Triton** (`@triton.jit`) | 默认路径 | [`fused_moe.py`](vllm/model_executor/layers/fused_moe/fused_moe.py) |
| DeepGemm | **外部 CUDA 库** | FP8 精度 | [`deep_gemm_moe.py`](vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py) |
| CUTLASS MoE | **Custom CUDA (CUTLASS)** | FP8/INT8 | [`cutlass_moe.py`](vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py) |
| MXFP8 CUTLASS | **Custom CUDA (CUTLASS)** | MXFP8 | [`csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu`](csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu) |

说明：Triton `fused_moe_kernel` 是默认的 Expert 计算路径，将 gate_proj + up_proj + activation + down_proj 融合在一个 kernel 中。FP8 量化时会优先使用 DeepGemm 或 CUTLASS。

**MoE 辅助算子：**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `moe_align_block_size` | **Custom CUDA** | [`csrc/moe/moe_align_sum_kernels.cu`](csrc/moe/moe_align_sum_kernels.cu) |
| `moe_permute / unpermute` | **Custom CUDA** | [`csrc/moe/moe_permute_unpermute_op.cu`](csrc/moe/moe_permute_unpermute_op.cu) |
| `topk_weights * expert_output` | PyTorch (逐元素乘加) | fused_moe layer |

**Shared Expert（共享专家）：**

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `shared_experts.gate_up_proj` | PyTorch (`MergedColumnParallelLinear`) | [`deepseek_v2.py:305`](vllm/model_executor/models/deepseek_v2.py#L305) |
| `SiluAndMul` | **Custom CUDA** | [`csrc/activation_kernels.cu`](csrc/activation_kernels.cu) |
| `shared_experts.down_proj` | PyTorch (`RowParallelLinear`) | [`deepseek_v2.py:305`](vllm/model_executor/models/deepseek_v2.py#L305) |

说明：Shared Expert 的输出与 Routed Expert 的输出相加。

---

### 3. Final RMSNorm

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `model.norm` | **Custom CUDA** (`fused_add_rms_norm`) | [`csrc/layernorm_kernels.cu`](csrc/layernorm_kernels.cu) |

---

### 4. LM Head

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `lm_head` | PyTorch (`ParallelLMHead` → `nn.Linear`) | [`linear.py`](vllm/model_executor/layers/linear.py) |

说明：将最终的 hidden_states 投影到 vocab_size 维度，得到每个 token 的 logits。

---

### 5. Logits Processing

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| `LogitsProcessor` | PyTorch（gather + 分布式合并） | [`logits_processor.py`](vllm/model_executor/layers/logits_processor.py) |

说明：对 lm_head 的输出做 TP gather（如果使用张量并行），得到完整的 vocab logits。

---

### 6. Sampling（采样）

| 算子 | 实现方式 | 文件路径 |
|------|---------|---------|
| Temperature scaling | PyTorch (`logits / temperature`) | [`sampler.py`](vllm/v1/sample/sampler.py) |
| Top-k filtering | PyTorch (`torch.topk`) | [`sampler.py`](vllm/v1/sample/sampler.py) |
| Top-p (nucleus) filtering | PyTorch (`torch.sort` + `cumsum`) | [`sampler.py`](vllm/v1/sample/sampler.py) |
| Multinomial sampling | PyTorch (`torch.multinomial`) | [`sampler.py`](vllm/v1/sample/sampler.py) |
| Greedy (argmax) | PyTorch (`torch.argmax`) | [`sampler.py`](vllm/v1/sample/sampler.py) |

---

## 算子实现分类汇总

### Custom CUDA Kernels（C++/CUDA）

| 算子名称 | 功能 | 源文件 |
|---------|------|--------|
| `fused_add_rms_norm` | RMSNorm + 残差加法融合 | [`csrc/layernorm_kernels.cu`](csrc/layernorm_kernels.cu) |
| `silu_and_mul` | SiLU 激活 + 逐元素乘法 | [`csrc/activation_kernels.cu`](csrc/activation_kernels.cu) |
| `rotary_embedding` | RoPE 位置编码 | [`csrc/pos_encoding_kernels.cu`](csrc/pos_encoding_kernels.cu) |
| `concat_mla_q` | MLA Q 向量拼接 | [`csrc/concat_mla_q.cuh`](csrc/concat_mla_q.cuh) |
| `merge_attn_states` | 分块注意力输出合并 | [`csrc/attention/merge_attn_states.cu`](csrc/attention/merge_attn_states.cu) |
| `grouped_topk` | MoE 分组 Top-K 路由 | [`csrc/moe/grouped_topk_kernels.cu`](csrc/moe/grouped_topk_kernels.cu) |
| `topk_softplus_sqrt` | DeepSeek V3 路由激活函数 | [`csrc/moe/topk_softplus_sqrt_kernels.cu`](csrc/moe/topk_softplus_sqrt_kernels.cu) |
| `dsv3_router_gemm` | DeepSeek V3 路由 GEMM | [`csrc/moe/dsv3_router_gemm_entry.cu`](csrc/moe/dsv3_router_gemm_entry.cu) |
| `moe_align_block_size` | MoE token 对齐 | [`csrc/moe/moe_align_sum_kernels.cu`](csrc/moe/moe_align_sum_kernels.cu) |
| `moe_permute / unpermute` | MoE token 重排 | [`csrc/moe/moe_permute_unpermute_op.cu`](csrc/moe/moe_permute_unpermute_op.cu) |
| `sm100_cutlass_mla_decode` | Blackwell MLA decode | CUTLASS template |

### Triton Kernels

| 算子名称 | 功能 | 源文件 |
|---------|------|--------|
| `fused_moe_kernel` | MoE Expert 融合 GEMM + 激活 | [`fused_moe.py`](vllm/model_executor/layers/fused_moe/fused_moe.py) |
| `triton_decode_attention` | Decode 阶段注意力 | [`triton_decode_attention.py`](vllm/v1/attention/ops/triton_decode_attention.py) |

### 外部 C++/CUDA 库

| 库名称 | 功能 | 调用位置 |
|--------|------|---------|
| vllm_flash_attn | Flash Attention (prefill) + RoPE | `vllm/vllm_flash_attn/` |
| FlashMLA | MLA 专用 decode attention | [`flashmla.py`](vllm/v1/attention/ops/flashmla.py) |
| FlashInfer | 高性能 attention backend | [`flashinfer_mla.py`](vllm/v1/attention/backends/mla/flashinfer_mla.py) |
| DeepGemm | FP8 grouped GEMM | [`deep_gemm_moe.py`](vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py) |
| DeepEP | Expert Parallelism 通信 | [`deepep_ht.py`](vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py) |

### Pure PyTorch

| 算子/操作 | 功能 | 备注 |
|----------|------|------|
| `nn.Embedding` | Embedding lookup | VocabParallelEmbedding |
| `nn.Linear` (ColumnParallel/RowParallel) | 所有投影层 (Q/KV/O/Gate/Up/Down) | 矩阵乘法由 cuBLAS 执行 |
| `torch.topk / sort / multinomial` | 采样相关操作 | Sampler |
| `LogitsProcessor` | Logits 后处理 | gather + reduce |

---

## 关键路径性能热点

按计算量排序（从大到小）：

1. **MoE Expert GEMM** (Triton/DeepGemm/CUTLASS) — 占总计算量 ~60-70%
2. **Attention Kernel** (Flash Attention/FlashMLA) — 占总计算量 ~15-20%
3. **Q/KV/O 投影 Linear** (cuBLAS) — 占总计算量 ~10-15%
4. **RMSNorm / RoPE / Activation** (Custom CUDA) — 占总计算量 <5%

---

## 备注

- 以上分析基于 NVIDIA GPU (CUDA) 路径。ROCm (AMD) 和 XPU (Intel) 有各自的 backend 实现。
- 实际运行时选择哪个 attention backend 取决于 `VLLM_ATTENTION_BACKEND` 环境变量和硬件检测。
- FP8 量化模式下，Linear 层会使用量化 GEMM（cutlass/deep_gemm），而非标准 cuBLAS。
- DeepSeek V3 相比 V2 新增了 `topk_softplus_sqrt` 和 `dsv3_router_gemm` 等专用算子。
