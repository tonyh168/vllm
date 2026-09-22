# 第 10 章 量化与 MoE

> **本章回答**：并行线性层怎么切？量化是怎么接进 `LinearBase` 的？`FusedMoE` 被拆成了哪几件套？MoE 的性能瓶颈在哪、kernel 怎么选？
> 涉及文件：`vllm/model_executor/layers/{linear,layernorm,activation,vocab_parallel_embedding,logits_processor}.py`、`layers/quantization/`、`layers/fused_moe/`

## 10.1 量化为什么是最有效的优化

回顾第 7 章的结论：**decode 是显存带宽瓶颈**。而 decode 每步要读：

```text
全部模型权重（w bytes）
+ 全部 KV cache（k bytes）
```

量化直接削减前者：

| 精度 | 权重带宽 | 相对 fp16 |
| --- | --- | --- |
| bf16/fp16 | 2 B/param | 1× |
| fp8 | 1 B/param | **2×** |
| int4 / nvfp4 / mxfp4 | 0.5 B/param | **4×** |

**（近似）线性的 decode 加速**，同时权重显存减半到 1/4 —— 省下的显存全部可以给 KV cache，并发上限也跟着涨。

> 💡 **性能视角**：量化是**唯一一个同时改善"显存容量"和"显存带宽"的手段**。
> 这也是为什么 vLLM 把量化支持做得如此之广（20 多种方法）。**其他所有优化都是在一个固定预算内做分配；量化是直接扩大预算。**

代价：
1. **精度损失**（因模型/方法而异，通常 fp8 几乎无损，int4 需要看具体模型）；
2. **反量化开销**（尤其是 weight-only 量化，decode 时要做 dequant）；
3. **kernel 覆盖度**（不是所有量化格式在所有硬件上都有高效 kernel）。

## 10.2 并行线性层家族

`vllm/model_executor/layers/linear.py`。

### `LinearBase`：量化的接入点

```python
# linear.py:228
class LinearBase(PluggableLayer):
    def __init__(self, input_size, output_size, bias=False, skip_bias_add=False,
                 params_dtype=None, quant_config=None, prefix="", ...):
        ...
        # ★ 量化方法的唯一分发点
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
            self.quant_method = quant_method
        else:
            raise ValueError("All linear layers should support quant method.")
        ...
        self.tp_rank = tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
        self.tp_size = tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
```

**`get_quant_method(self, prefix=prefix)` 是关键设计**：量化配置**按层名前缀**决定用哪种量化方法。这让"非均匀量化"成为可能 —— 例如某些层用 fp8、某些层保持 bf16（敏感层）。

`apply` 是统一的调用接口：

```python
output = self.quant_method.apply(self, input_, bias)
```

**所有量化方法都实现 `create_weights` / `apply` / `process_weights_after_loading` 三件套**（`LinearMethodBase`，`:125`）。

`update_param_tp_status`（`:291`）的注释解释了一个微妙的 bug 场景：`process_weights_after_loading` 可能替换掉 `Parameter` 对象，此时 TP 状态需要重新 reconcile，否则后续的 `load_weights` 会按错误的 rank 切片而溢出。

### 六个线性层实现

| 类 | 切什么 | 通信 | 典型用途 |
| --- | --- | --- | --- |
| `ReplicatedLinear`（`:309`） | 不切 | 无 | router gate、小投影 |
| `ColumnParallelLinear`（`:414`） | **输出维** | 无（`gather_output=False` 时） | QKV、gate/up |
| `RowParallelLinear`（`:1606`） | **输入维** | **all-reduce** | O projection、down |
| `MergedColumnParallelLinear`（`:652`） | 输出维，**多层合并** | 无 | MLP 的 gate_proj + up_proj |
| `QKVParallelLinear`（`:978`） | 输出维，**Q/K/V 合并** | 无 | attention 的 QKV |
| `DCPGroupColumnParallelLinear`（`:611`） | decode context parallel 版 | — | DCP |

### 为什么"合并"很重要

`ColumnParallelLinear.forward`（`:582`）：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
if self.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel        # ★ 常见情况：无通信
return output, output_bias
```

**列并行的输出天然是"每 rank 一份分片"，如果不需要 gather 就是零通信。** 这是"列并行在前、行并行在后"这个经典排布的原因：

```text
MLP:  x → [gate(k), up(k) 合并的列并行] → SiLU(gate)*up → [down 的行并行 + all-reduce]
                ↑ 无通信                                   ↑ 一次 all-reduce
```

**一个 Transformer 层只需两次 all-reduce**（attention 的 O projection 和 MLP 的 down projection 各一次）。

`MergedColumnParallelLinear`（`:652`）和 `QKVParallelLinear`（`:978`）的价值：**一次 GEMM kernel 代替两三次**。

- `MergedColumnParallelLinear`：`output_sizes` 是一个列表（`gate_proj` 和 `up_proj` 的尺寸），`weight_loader` 按 `shard_id` 决定往哪一段写。
- `QKVParallelLinear`：`_get_shard_offset_mapping`（`:1067`）/ `_get_shard_size_mapping`（`:1077`）把 `"q"/"k"/"v"` 映射到偏移。**GQA/MQA 下 K/V 的头数少于 Q，切分需要特殊处理**（不能整除时 padding）。

> 💡 **性能视角**：融合的收益有两个层面：
> 1. **计算层面**：一次大 GEMM 比三次小 GEMM 的算力利用率更高（尤其 decode 时 batch 小、GEMM 本就"瘦"）；
> 2. **launch 层面**：少两次 kernel launch（在 CUDA Graph 下这个收益变小）。

### `VocabParallelEmbedding` 与 `ParallelLMHead`

`vocab_parallel_embedding.py`：

- `VocabParallelEmbeddingShardIndices`（`:110`）：**专门处理词表不能被 TP 整除**的情况（padding + 边界），属性有 `num_org_elements` / `num_added_elements` / `num_org_vocab_padding`。
- `VocabParallelEmbedding.forward`（`:490`）：用 masked index 避免越界。
- `ParallelLMHead`（`:532`）：输出投影。
- **`tie_weights`（`:586`）**：权重绑定 —— 输入 embedding 和输出 LM head 共享同一份权重，**省一半的词表显存**（词表 15 万 × hidden 4096 × 2B = 1.2 GB）。

`LogitsProcessor`（`logits_processor.py:58`）只对 `logits_indices` 指定的位置算 LM head（第 5 章讲过）。

## 10.3 `process_weights_after_loading`：把开销挪到启动期

`vllm/model_executor/model_loader/utils.py:97`。它遍历所有层调用 `process_weights_after_loading`。

**这一步做的是"权重重排/打包"**：

| 量化方法 | 重排内容 |
| --- | --- |
| `*_marlin`（GPTQ/AWQ Marlin） | 把权重 swizzle 成 Marlin kernel 需要的 layout |
| `fp8` | 计算并布局 scale（per-tensor / per-channel / per-block） |
| `compressed-tensors` | 按 scheme 解包 + 重排 |
| `bitsandbytes` | 反量化或保持 NF4 布局 |

**核心理念：这些都是"一次性的启动成本，换运行期零开销"。** 如果放到 forward 里做，就是每步每层都付一次。

> 💡 **性能视角**：Marlin 是个好例子 —— 它把 int4 权重重排成一种"反量化极快"的布局，**decode 时能跑到接近 fp16 的速度**（同时带宽只有 1/4）。
> 代价是启动时要多做一次全权重遍历（几十秒），以及只支持特定的 group size / 对称性组合。

## 10.4 量化方法清单

`vllm/model_executor/layers/quantization/__init__.py:12` 的 `QuantizationMethods`：

**离线量化格式（有 checkpoint）**：

| 方法 | 类型 | 特点 |
| --- | --- | --- |
| `fp8` | W8A8 / W8A16 | 精度损失极小，**H100+ 有原生支持** |
| `fbgemm_fp8` | CPU fp8 | — |
| `fp_quant` | 浮点量化 | — |
| `awq` / `auto_awq` | W4A16 | 激活感知的权重量化 |
| `awq_marlin` | W4A16 + Marlin | **decoder 服务的常用选择** |
| `gptq` / `auto_gptq` | W4A16 | — |
| `gptq_marlin` | W4A16 + Marlin | — |
| `marlin` | 通用 Marlin | — |
| `compressed-tensors` | 多种 | llm-compressor 的格式，覆盖面最广 |
| `modelopt` / `modelopt_fp4` / `modelopt_mxfp8` / `modelopt_mixed` | NVIDIA ModelOpt | Blackwell 上的 FP4/MXFP8 |
| `mxfp4` / `gpt_oss_mxfp4` | MXFP4 | GPT-OSS 等 |
| `deepseek_v4_fp8` | — | 模型专用 |
| `experts_int8` | W8A8（MoE 专家） | — |
| `moe_wna16` | MoE W4A16 | — |
| `torchao` | PyTorch AO | — |
| `quark` | AMD Quark | — |
| `inc` | Intel Neural Compressor | — |
| `humming` | — | fork/实验 |

**在线量化（运行时量化，无需量化 checkpoint）**：

| shorthand | 含义 |
| --- | --- |
| `fp8_per_tensor` / `fp8_per_block` / `fp8_per_channel` | FP8 的不同 scale 粒度 |
| `int8_per_channel_weight_only` | 仅权重 int8 |
| `nvfp4_per_token` | NVFP4 |
| `mxfp8` | MXFP8 |

**用法**：

```bash
vllm serve <model> --quantization fp8                    # 加载已有 fp8 checkpoint
vllm serve <model> --quantization fp8_per_block          # 在线量化（从 bf16 checkpoint）
```

**scale 粒度的权衡**：

| 粒度 | 精度 | kernel 复杂度 | 显存开销 |
| --- | --- | --- | --- |
| per-tensor | 最低 | 最简单 | 可忽略 |
| per-channel | 中 | 中 | 小 |
| per-block（128×128） | 高 | 复杂 | 中 |
| per-token / per-head | 最高 | 复杂 | 大 |

## 10.5 MoE 的四件套

⚠️ **重大结构变化**：本版本中全局的 `FusedMoE` 类**已经不存在**，被重构为 `FusedMoEFactory` + 四件套。如果你看到老文档/老博客讲 `FusedMoE`，那是旧版本。

`vllm/model_executor/layers/fused_moe/layer.py:88`：

```python
def FusedMoEFactory(...):
    # 工厂函数，按参数装配一个 MoE 层，内部构造：
    #   RoutedExperts   —— 专家权重
    #   FusedMoERouter  —— 路由（gate）
    #   MoERunner       —— 编排
    #   SharedExperts   —— 共享专家
```

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `FusedMoEFactory` | `layer.py:88` | 装配 |
| `RoutedExperts` | `routed_experts.py:45` | 持有专家权重；`weight_loader`（三个 overload）、`build_expert_params_mapping`（`:1033`，**按 EP/TP 生成参数名映射**）、`forward_modular` / `forward_monolithic` |
| `MoERunner` | `runner/moe_runner.py:227` | **编排层**：`_select_forward`、`_maybe_dispatch` / `_maybe_combine`、`_apply_quant_method`、`_maybe_apply_shared_experts`、`_maybe_add_zero_expert_output`、`_maybe_pad_hidden_states`、`is_monolithic` |
| `FusedMoERouter` | `router/fused_moe_router.py` | 路由：`create_fused_moe_router`（`router_factory.py`）；变体含 `grouped_topk_router`（DeepSeek 式分组 top-k）、`fused_topk_router`、`fused_topk_bias_router`、`zero_expert_router`、`dsv4_topk`、`custom_routing_router`、`bf16x3_router_gemm_cutedsl`（**用 3×bf16 模拟 fp32 的 router GEMM**） |

### 两条执行路径

`MoERunner.is_monolithic`（`:948`）区分：

| 路径 | 说明 |
| --- | --- |
| **monolithic** | 一个融合 kernel 完成 dispatch + 专家计算 + combine（`forward_monolithic`） |
| **modular**（modular kernel） | dispatch / 专家计算 / combine 分成可组合的三段（`modular_kernel.py` 的 `FusedMoEExpertsModular` / `FusedMoEPrepareAndFinalizeModular`） |

**modular 的意义**：让"专家 kernel"和"通信实现"自由组合。例如 `deep_gemm_moe` 的专家 kernel + `deepep_ht` 的通信。这就是 `prepare_finalize/` 目录下那些文件的作用（见第 9 章 9.5）。

配置侧：`FusedMoEConfig` / `FusedMoEParallelConfig`（`config.py:1040`）/ `FusedMoEQuantConfig` / `RoutingMethodType`。

## 10.6 MoE 的性能瓶颈在哪

```text
MoE 层的一次前向：
1. gate GEMM + top-k 选择        ← 小 GEMM，latency-bound
2. dispatch（all-to-all）        ← 通信
3. permute / align              ← 数据重排
4. 专家 GEMM（grouped GEMM）     ← kernel 效率是关键
5. unpermute                   ← 数据重排
6. combine（all-to-all）        ← 通信
7. 加权求和                     ← elementwise
```

### 三个瓶颈点

**① Grouped GEMM 的效率**

每个专家处理的 token 数不同（受路由分布影响）。朴素做法是"每个专家一次 GEMM"，但 token 少时 GEMM 很"瘦"、算力利用率低。

现代实现用 **grouped GEMM**（一个 kernel 处理所有专家，用 `m_indices` 分组）或 **batched GEMM**。相关的 `moe_align_block_size.py`（把 token 数对齐到 block）、`moe_permute_unpermute.py`（重排）就是为此服务的。

**② 通信（第 9 章的 all2all）**

**③ 数据重排的显存往返**

permute / unpermute 是纯 memory-bound 的 elementwise 操作。`moe_fused_mul_sum.py`、`topk_weight_and_reduce.py` 就是在融合这些。

### 专家 kernel 清单（35 个文件）

`vllm/model_executor/layers/fused_moe/experts/`：

| 类别 | 文件 |
| --- | --- |
| 通用兜底 | `triton_moe.py`、`fallback.py` |
| CUTLASS 系 | `cutlass_moe.py`、`triton_cutlass_moe.py` |
| DeepGEMM 系 | `deep_gemm_moe.py`、`batched_deep_gemm_moe.py`、`triton_deep_gemm_moe.py` |
| Marlin | `marlin_moe.py` |
| TRT-LLM 系 | `trtllm_fp8_moe.py`、`trtllm_bf16_moe.py`、`trtllm_nvfp4_moe.py`、`trtllm_mxfp4_moe.py`、`trtllm_mxint4_moe.py` |
| FlashInfer 系 | `flashinfer_cutlass_moe.py`、`flashinfer_cutedsl_moe.py`、`flashinfer_cutedsl_batched_moe.py`、`flashinfer_b12x_moe.py` |
| 模拟量化 | `mxfp8_native_moe.py`、`mxfp8_emulation_moe.py`、`nvfp4_emulation_moe.py`、`ocp_mx_emulation_moe.py`、`int4_emulation_moe.py` |
| ROCm/AITER | `rocm_aiter_moe.py`、`aiter_mxfp4_w4a8_moe.py`、`aiter_mxfp8_moe.py` |
| CPU | `cpu_moe.py`、`cpu_int4_moe.py` |
| 其他 | `xpu_moe.py`、`fused_humming_moe.py`、`gpt_oss_triton_kernels_moe.py`、`fused_batched_moe.py` |

**"emulation" 是什么？** 在没有原生低精度支持的硬件上，用 bf16 计算模拟低精度量化 —— **保留量化 checkpoint 的兼容性，但拿不到带宽收益**。这是"能跑起来"和"跑得快"之间的过渡方案。

> 💡 **性能视角**：MoE 的性能**极度依赖 kernel 选择**。同一个模型在不同 kernel 上的差异可以是 2~3 倍。
> 选择顺序通常由 vLLM 自动决定（结合硬件、量化格式、专家数），但可以用 `--kernel-config` / `MoEBackend` 覆盖。
> **调优 MoE 模型时，第一件事是确认用的是哪个专家 kernel**（启动日志里会打印）。

## 10.7 层归一化与激活的融合

`layernorm.py`：

| 类 | 说明 |
| --- | --- |
| `RMSNorm`（`:37`） | **最常被融合的算子** |
| `GemmaRMSNorm`（`:132`） | Gemma 的变体 |
| `RMSNormGated`（`:172`） | 带门控 |
| `LayerNorm`（`:310`） | 标准 LayerNorm |
| `poly_norm`（`:19`） | 多项式归一化 |

**围绕 RMSNorm 的融合 pass 最多**（见第 6 章 6.9）：

```text
add_rms_fusion          → residual add 融进来
rms_quant_fusion        → 输出量化融进来
allreduce_rms_fusion    → TP 的 all-reduce 融进来  ★ TP 场景收益最大
fused_allreduce_gemma_rms_norm.py  → Gemma 专用
fused_embed_norm.py     → embedding + norm
```

`allreduce_rms_fusion` 尤其值得单独理解：它把 all-reduce 塞进 RMSNorm kernel 的**尾部**，省掉一次完整的显存往返（写 all-reduce 输入 → 读结果 → 写 norm 输出）。

`activation.py` 里的 `*AndMul` 家族：`SiluAndMul`（`:112`）、`GeluAndMul`（`:417`）、`SwigluOAIAndMul`（`:480`）、`FatreluAndMul`（`:73`）、`SituAndMul`（`:156`）、`MulAndSilu`（`:261`）等。

**它们的共同点是把"门控激活"的两次逐元素运算 + 一次乘法合成一个 kernel**。MLP 里这段是纯 memory-bound 的，融合直接削减显存流量。

`get_act_fn`（`:821`）/ `get_act_and_mul_fn`（`:849`）是工厂函数。

## 10.8 Rotary Embedding：18 种变体

`rotary_embedding/` 的基类是 `RotaryEmbeddingBase(CustomOp)`（`:15`）和 `RotaryEmbedding`（`:139`）。

**关键设计：`CustomOp` 基类让每个算子提供两条路径**：

```python
def forward_native(self, ...):   # torch 实现，可被 compile 融合
def forward_cuda(self, ...):     # 自定义 kernel
```

**按编译/图捕获状态自动选**。这是 vLLM 里一个通用模式（`layernorm.py`、`activation.py` 等都用）：**在能融合的场景用 torch 实现（让编译器优化），在不能融合的场景用自定义 kernel**。

18 个变体文件：`common.py`、`deepseek_scaling_rope.py`、`dual_chunk_rope.py`、`dynamic_ntk_alpha_rope.py`、`dynamic_ntk_scaling_rope.py`、`ernie45_vl_rope.py`、`fope.py`、`gemma4_rope.py`、`linear_scaling_rope.py`、`llama3_rope.py`、`llama4_vision_rope.py`、`mrope.py`、`mrope_interleaved.py`、`ntk_scaling_rope.py`、`phi3_long_rope_scaled_rope.py`、`telechat3_scaling_rope.py`、`xdrope.py`、`yarn_scaling_rope.py`。

**长上下文扩展方案（NTK / YaRN / Linear / Dynamic）本质上是不同频率缩放策略**，它们决定了模型能不能外推到训练长度之外。

## 10.9 本章小结

| 机制 | 收益 | 代价 |
| --- | --- | --- |
| 量化（fp8/int4/mxfp4） | **带宽 + 容量双降**（2~4×） | 精度、kernel 覆盖度 |
| `process_weights_after_loading` | 把重排开销挪到启动期 | 启动时间 |
| 列并行 / 行并行排布 | 一个 Transformer 层只两次 all-reduce | — |
| Merged/QKV 融合线性 | 一次 GEMM 代替多次 | 权重加载逻辑复杂（shard 映射） |
| `tie_weights` | 词表权重省一半 | 只适用于绑定权重的模型 |
| `get_quant_method(prefix)` | 支持非均匀量化 | 配置复杂 |
| MoE modular kernel | 专家 kernel 与通信自由组合 | 配置空间大 |
| grouped GEMM / align | 解决专家 token 数不均 | 需要重排（显存往返） |
| `*AndMul` 融合激活 | 削减 elementwise 显存流量 | — |
| `allreduce_rms_fusion` | 省一次显存往返 | 需要 TP |
| `CustomOp` 双路径 | 编译期融合 vs 运行期 kernel | 两条路径都要维护 |

⚠️ **易错点**
- `FusedMoE` 类已重构为 `FusedMoEFactory` + 四件套，老文档会误导你。
- "emulation"（模拟）量化能跑但不快 —— 别把它当成真正的加速。
- 量化模型的 decode 加速来自带宽，**prefill 的加速有限**（prefill 是 compute-bound，量化后反量化还有额外开销）。

📖 官方文档：`docs/design/fused_moe_modular_kernel.md`、`docs/design/moe_kernel_features.md`

下一章 → [第 11 章 模型加载与启动时间优化](11-模型加载与启动时间优化.md)
