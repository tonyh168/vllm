# vLLM 模型权重加载机制详解

本文档介绍 vLLM 如何从 HuggingFace 格式的 checkpoint（本地磁盘或远程 Hub）加载模型权重，并将其映射到 vLLM 内部模型类的参数上。

## 整体流程概览

```
get_model(vllm_config)
  → get_model_loader(load_config)          # 选择 Loader（默认 DefaultModelLoader）
  → loader.load_model(vllm_config)
       1. initialize_model()               # 解析架构 → 实例化空模型
       2. loader.load_weights(model)       # 读取文件 → 逐 tensor 加载
       3. process_weights_after_loading()  # 量化后处理等
```

---

## 阶段 1：模型类解析（Registry）

### 入口

文件：`vllm/model_executor/model_loader/__init__.py`

```python
def get_model(*, vllm_config: VllmConfig, ...) -> nn.Module:
    loader = get_model_loader(load_config)
    return loader.load_model(vllm_config=vllm_config, ...)
```

### 架构 → 类 的映射

文件：`vllm/model_executor/models/registry.py`

vLLM 读取模型目录下 `config.json` 的 `architectures` 字段（如 `"DeepseekV3ForCausalLM"`），在全局映射表 `_VLLM_MODELS` 中查找：

```python
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    ...
}
```

格式为 `"HF架构名": ("模块文件名", "vLLM类名")`。

解析流程（`model_loader/utils.py` 中的 `_get_model_architecture()`）：
1. 读取 `model_config.hf_config.architectures`
2. 调用 `ModelRegistry.resolve_model_cls()` 查找匹配项
3. 通过 `_LazyRegisteredModel` 懒加载：`importlib.import_module` + `getattr`
4. 若没有原生实现，回退到 Transformers 后端

---

## 阶段 2：发现权重文件

文件：`vllm/model_executor/model_loader/default_loader.py`

### `_prepare_weights()` 方法

```python
def _prepare_weights(self, model_name_or_path, ...):
    is_local = os.path.isdir(model_name_or_path)

    if is_local:
        hf_folder = model_name_or_path   # 直接使用本地路径
    else:
        hf_folder = download_weights_from_hf(...)  # 从 HuggingFace Hub 下载

    # 按优先级扫描文件
    # load_format="hf" → ["*.safetensors", "*.bin"]
    # load_format="safetensors" → ["*.safetensors"]
    # load_format="pt" → ["*.pt"]
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
```

对于 safetensors 格式，还会读取 `model.safetensors.index.json` 过滤重复分片文件。

---

## 阶段 3：迭代读取权重张量

文件：`vllm/model_executor/model_loader/weight_utils.py`

### `safetensors_weights_iterator()`

```python
def safetensors_weights_iterator(hf_weights_files, ...):
    for st_file in sorted(hf_weights_files):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                yield (name, tensor)
```

特点：
- **流式读取**：逐个 tensor yield，不需要一次性把所有权重加载到内存
- **支持 EP 过滤**：开启 Expert Parallelism 时可跳过非本地 expert 的权重
- **支持多线程**：可配置 `enable_multithread_load` 并行读取多个文件

### `_get_weights_iterator()` 根据格式选择迭代器

| 格式 | 迭代器 |
|------|--------|
| safetensors | `safetensors_weights_iterator` |
| fastsafetensors | `fastsafetensors_weights_iterator` |
| instanttensor | `instanttensor_weights_iterator` |
| pt/bin | `pt_weights_iterator`（使用 `torch.load`） |
| npcache | `np_cache_weights_iterator` |

---

## 阶段 4：权重名称映射与加载

这是最核心的部分。`DefaultModelLoader.load_weights()` 将迭代器传给模型自身：

```python
def load_weights(self, model, model_config):
    loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

### 为什么需要名称映射？

HuggingFace checkpoint 中的权重是**分离**的（如 `q_proj`、`k_proj`、`v_proj` 各自独立），而 vLLM 为了性能会将它们**融合**成单个大参数：

```
HuggingFace checkpoint                    vLLM 模型参数
─────────────────────                    ──────────────
layers.0.self_attn.q_proj.weight    ──┐
layers.0.self_attn.k_proj.weight    ──┼──► layers.0.self_attn.qkv_proj.weight
layers.0.self_attn.v_proj.weight    ──┘

layers.0.mlp.gate_proj.weight       ──┐
layers.0.mlp.up_proj.weight         ──┴──► layers.0.mlp.gate_up_proj.weight

layers.0.mlp.down_proj.weight       ────► layers.0.mlp.down_proj.weight（直接对应）
```

**融合的好处**：用一次 GEMM 代替多次小 GEMM，提升 GPU 计算利用率。

---

## 两种 load_weights 实现风格

### 风格 A：手动 `stacked_params_mapping`（旧风格）

以 `DeepseekV2Model.load_weights()` 为例（`deepseek_v2.py`）：

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (vllm参数名, HF权重名, shard_id)
        ("gate_up_proj", "gate_proj", 0),   # gate_proj → gate_up_proj 的第0片
        ("gate_up_proj", "up_proj", 1),     # up_proj → gate_up_proj 的第1片
    ]

    params_dict = dict(self.named_parameters())

    for name, loaded_weight in weights:
        # 尝试匹配 stacked mapping
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            param.weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # 不需要特殊映射的，直接按名字匹配
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
```

关键点：
- `param.weight_loader` 是挂在参数上的回调函数（由 `ColumnParallelLinear`、`QKVParallelLinear` 等层注册）
- 它知道如何将 shard 数据写入融合参数的正确偏移位置
- 同时自动处理 Tensor Parallel 切分

### 风格 B：`AutoWeightsLoader` + `WeightsMapper`（新风格）

文件：`vllm/model_executor/models/utils.py`

```python
# llama.py — 顶层调用非常简洁
class LlamaForCausalLM(nn.Module):
    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
```

`AutoWeightsLoader` 递归遍历模块树：
1. 按 `.` 分隔前缀，找到对应的子模块
2. 如果子模块有自己的 `load_weights()` 方法，委托给它
3. 如果参数有 `weight_loader` 回调，调用它
4. 否则使用 `default_weight_loader`（简单 copy）

`WeightsMapper` 支持四种名称变换：
- `orig_to_new_regex`：正则替换
- `orig_to_new_substr`：子串替换
- `orig_to_new_prefix`：前缀替换
- `orig_to_new_suffix`：后缀替换

---

## MoE Expert 权重的特殊处理

DeepSeek V2/V3 的 MoE 层有大量 expert 权重（如 `mlp.experts.0.gate_proj`），需要额外的映射逻辑：

```python
expert_params_mapping = fused_moe_make_expert_params_mapping(
    self,
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=config.n_routed_experts,
)
# 生成映射：(param_name, weight_name, expert_id, shard_id)
```

所有 expert 的权重被融合到一个大的 `FusedMoE` 参数中（维度为 `[num_experts, intermediate_size, hidden_size]`），加载时通过 `expert_id` 索引到正确位置。

---

## `packed_modules_mapping` 的作用

模型类上的类属性，告知量化和 LoRA 系统哪些 HF 权重被融合了：

```python
class DeepseekV2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
```

量化方法需要知道这个信息，以便正确处理 scale/zero point。

---

## 使用方式

```python
from vllm import LLM

# 传本地路径，vLLM 自动完成上述所有步骤
llm = LLM(model="/path/to/your/local/model")

# 传 HuggingFace repo id，自动下载后加载
llm = LLM(model="deepseek-ai/DeepSeek-V3-Base")
```

---

## 关键文件速查

| 作用 | 文件路径 |
|------|----------|
| 顶层入口 | `vllm/model_executor/model_loader/__init__.py` |
| 加载编排 | `vllm/model_executor/model_loader/base_loader.py` |
| 文件发现 + 迭代 | `vllm/model_executor/model_loader/default_loader.py` |
| safetensors 读取 | `vllm/model_executor/model_loader/weight_utils.py` |
| 模型注册表 | `vllm/model_executor/models/registry.py` |
| AutoWeightsLoader | `vllm/model_executor/models/utils.py` |
| DeepSeek V2 权重加载 | `vllm/model_executor/models/deepseek_v2.py` |
| Llama 权重加载 | `vllm/model_executor/models/llama.py` |

---

## 流程图总结

```
用户传入 model="/local/deepseek-v3"
    │
    ▼
读取 config.json → architectures: ["DeepseekV3ForCausalLM"]
    │
    ▼
Registry 查找 → ("deepseek_v2", "DeepseekV3ForCausalLM")
    │
    ▼
import vllm.model_executor.models.deepseek_v2
实例化 DeepseekV3ForCausalLM（此时参数为空/随机）
    │
    ▼
DefaultModelLoader._prepare_weights()
    → glob("*.safetensors") 发现权重文件
    │
    ▼
safetensors_weights_iterator()
    → 逐文件逐 tensor yield (name, tensor)
    │
    ▼
model.load_weights(iterator)
    → stacked_params_mapping 做名称替换
    → param.weight_loader(param, tensor, shard_id) 写入融合参数
    → expert_params_mapping 处理 MoE expert 权重
    │
    ▼
process_weights_after_loading()
    → 量化后处理、权重变换等
    │
    ▼
模型就绪，可以开始推理
```
