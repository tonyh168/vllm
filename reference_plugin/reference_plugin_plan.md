# vLLM Reference Plugin 实现计划

## 1. 目标

创建一个标准的 **reference 插件** (`vllm-reference`)，以插件形式承载 vLLM 推理过程中所有硬件相关的逻辑。最终目标是用纯 PyTorch / Triton 实现所有硬件特定算子，但分阶段推进。

**用途：**
- 作为多硬件插件开发的参考实现和模板
- 验证 vLLM 插件架构对硬件逻辑的拆分能力
- 为新硬件适配提供对照基准

---

## 2. 插件项目结构

```
vllm-reference/
├── pyproject.toml                    # 包定义 + entry points
├── README.md
├── vllm_reference/
│   ├── __init__.py                   # register(), register_ops() 入口函数
│   ├── platform.py                   # ReferencePlatform(Platform)
│   ├── attention/
│   │   ├── __init__.py
│   │   └── reference_attn.py        # Attention backend
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── activation.py            # 激活函数
│   │   ├── layernorm.py             # RMSNorm 系列
│   │   ├── rotary_embedding.py      # RotaryEmb
│   │   ├── fused_moe.py             # FusedMoE
│   │   ├── mamba.py                 # Mamba 相关
│   │   ├── conv.py                  # ConvLayer
│   │   └── sparse_attn.py           # SparseAttnIndexer
│   ├── communicator/
│   │   ├── __init__.py
│   │   └── reference_communicator.py # 通信实现
│   └── layers/
│       ├── __init__.py
│       ├── linear.py                 # LinearBase
│       ├── fused_moe.py              # FusedMoE PluggableLayer
│       ├── mla.py                    # MLA PluggableLayer
│       └── vocab_embedding.py        # VocabParallelEmbedding
```

---

## 3. Entry Points 注册

```toml
[project]
name = "vllm-reference"
version = "0.1.0"
dependencies = ["vllm>=0.20.0"]

[project.entry-points."vllm.platform_plugins"]
reference = "vllm_reference:register"

[project.entry-points."vllm.general_plugins"]
reference_ops = "vllm_reference:register_ops"
```

---

## 4. 施工步骤

### 阶段一：逻辑摘取（不写新代码，只做拆分验证）

目标：把 vLLM 中 NVIDIA CUDA platform 已有的逻辑，以插件形式组织起来，直接复用 vLLM 内部的现有实现。验证插件拆分的架构正确性。

**这个阶段不编写任何新算子，所有实现都是对 vLLM 已有代码的引用/继承/委托。**

#### Step 1: 项目骨架
- 创建 `pyproject.toml`、`vllm_reference/__init__.py`
- 实现 `register()` 函数（返回 Platform 类的 qualname）
- 实现 `register_ops()` 函数（注册 CustomOp / PluggableLayer）

#### Step 2: Platform 摘取
- 创建 `ReferencePlatform`，继承 `CudaPlatformBase`（或直接继承 `Platform`）
- `device_type = "cuda"`，`dispatch_key = "CUDA"`
- `get_attn_backend_cls()` → 直接指向 vLLM 已有的 Triton attention backend (`vllm.v1.attention.backends.triton_attn.TritonAttentionBackend`)
- `get_device_communicator_cls()` → 指向 vLLM 已有的 `CudaCommunicator`
- 其他方法（`get_device_capability`, `get_device_name`, `set_device` 等）直接委托 torch.cuda

#### Step 3: CustomOp 摘取
- 对所有 CustomOp 子类注册 OOT 版本
- `forward_oot()` 内部直接调用 `forward_cuda()`（因为阶段一仍跑在 NVIDIA GPU 上）
- 覆盖列表：
  - 激活函数：SiluAndMul, GeluAndMul, GELU, NewGELU, FastGELU, QuickGELU, 等全部
  - 归一化：RMSNorm, GemmaRMSNorm, RMSNormGated
  - RotaryEmb：ApplyRotaryEmb, RotaryEmbeddingBase, DualChunkRotaryEmbedding
  - Conv：ConvLayerBase
  - MoE：UnquantizedFusedMoEMethod, FusedMoEModularMethod, GroupedTopk
  - Mamba：Mixer2RMSNormGated, ShortConv, ChunkGatedDeltaRule
  - Attention：MMEncoderAttention
  - 暂不支持（直接抛异常）：QuantFP8、SparseAttnIndexer

#### Step 4: PluggableLayer 摘取
- 对核心 PluggableLayer 注册 OOT 版本，直接继承原类不做任何修改
- 覆盖列表：
  - LinearBase
  - FusedMoE
  - VocabParallelEmbedding
  - LogitsProcessor

#### Step 5: 通信层摘取
- `ReferenceCommunicator` 直接继承 `CudaCommunicator`，不 override 任何方法
- 验证通信层可以通过插件的 `get_device_communicator_cls()` 正确加载

#### Step 6: 验证
- `pip install -e .` 安装插件
- 设置 `VLLM_PLUGINS=reference` 激活插件
- 跑通一个 LLaMA/Qwen2 非量化模型的推理
- 确认输出与不装插件时完全一致（bit-exact）

**阶段一完成标准：** 插件安装后，vLLM 在 NVIDIA GPU 上的推理行为和结果与不装插件完全一致。证明插件的逻辑拆分架构是正确的。

---

### 阶段二：替换 NVIDIA 特定实现为通用实现

目标：将阶段一中直接委托 `forward_cuda` / 直接继承 CUDA 实现的部分，逐步替换为硬件无关的 PyTorch / Triton 实现。

#### Step 7: 激活函数 → PyTorch
- 将所有激活函数的 `forward_oot()` 从委托 `forward_cuda()` 改为委托 `forward_native()`
- 这些算子全部已有 `forward_native()` 实现，改动量极小

#### Step 8: 归一化 → PyTorch
- RMSNorm 系列的 `forward_oot()` 改为委托 `forward_native()`
- 已有 native 实现

#### Step 9: RotaryEmbedding → PyTorch
- 确认 `forward_native()` 可用，改为委托
- 如无 native 实现，用 PyTorch 编写（rotary 计算本身很简单）

#### Step 10: 通信层 → torch.distributed
- `ReferenceCommunicator` 改为继承 `DeviceCommunicatorBase`（纯 torch.distributed）
- 不再走 CudaCommunicator 的 PyNCCL / custom allreduce / FlashInfer allreduce 等加速路径
- 实现 `batch_isend_irecv` 等 base class 未覆盖的方法

#### Step 11: Attention Backend → Triton
- 复用 vLLM 已有的 `TritonAttentionBackend`（`vllm/v1/attention/backends/triton_attn.py`）
- 如果 Triton backend 有 NVIDIA 特定依赖，进行最小化修改使其通用化
- Triton 本身跨 NVIDIA/AMD/Intel 可移植，符合 reference 定位

#### Step 12: FusedMoE → PyTorch/Triton
- 实现一个 naive 版本的 MoE：topk routing + 循环 expert matmul
- 或者复用 vLLM 的 Triton FusedMoE kernel（已有 Triton 实现）

#### Step 13: LinearBase → PyTorch
- 替换为 `torch.nn.functional.linear` + tensor parallel allreduce/allgather

#### Step 14: VocabParallelEmbedding → PyTorch
- 替换为 `torch.nn.functional.embedding` + allreduce

#### Step 15: Conv / Mamba → PyTorch/Triton
- Conv：`forward_native()` 已有
- Mamba selective scan：用 Triton 或纯 PyTorch 实现

#### Step 16: MLA → Triton
- Multi-Head Latent Attention 的 reference 实现

**阶段二验证：**
- 跑通 LLaMA/Qwen2 → 验证基础路径
- 跑通 Mixtral → 验证 MoE 路径
- 跑通 DeepSeek-V3 → 验证 MLA + MoE
- 对比数值精度，允许浮点误差（非 bit-exact）

---

## 5. 暂不支持（直接抛异常）

以下算子/功能暂不实现，调用时统一抛 `NotImplementedError`：
- **量化**：QuantFP8 及所有量化算子。Platform 设置 `supported_quantization = []`，`verify_quantization()` 直接报错
- **Sparse Attention Indexer**：依赖 C++ CUDA kernel（`torch.ops._C.top_k_per_row_*`, `torch.ops._C.persistent_topk`），暂无 Triton/PyTorch 替代

---

## 6. 关键设计决策

### 阶段一为什么要委托 forward_cuda 而不是 forward_native？
因为阶段一的目标纯粹是验证**插件架构的正确性**，不改变任何计算逻辑。直接委托 forward_cuda 可以保证 bit-exact 输出，任何输出差异都说明插件拆分本身有 bug。

### 阶段二中 PyTorch vs Triton 的选择标准
- 简单逐元素算子（激活、归一化、rotary）→ 纯 PyTorch
- 复杂算子（attention、Mamba selective scan）→ 复用 vLLM 已有的 Triton 实现
- Sparse Attention Indexer → 暂不支持，C++ CUDA kernel 无现成替代
- Triton 跨 NVIDIA/AMD/Intel 均有编译器支持，是合理的通用层

### 性能预期
阶段一：与原始 CUDA 路径性能一致（只是多了一层函数调用）
阶段二：比优化路径慢 5-50x，可接受，目标是正确性
