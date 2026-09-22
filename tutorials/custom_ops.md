# vLLM Custom Op 机制详解

vLLM 拥有多层 custom op 体系，从底层 C++ kernel 注册到上层 Python 平台调度，各层职责分明、协同工作。

---

## 整体架构

```
用户模型代码 (nn.Module)
    │
    ▼
CustomOp (nn.Module，平台 dispatch)
    │
    ├─ forward_native() ──→ torch.compile / Inductor 可 fuse
    │
    └─ forward_cuda() ──→ _custom_ops.py wrapper
                              │
                              ▼
                         torch.ops._C.xxx()  (C++ CUDA kernel)
```

---

## 第一层：C++ Kernel 注册（最底层）

**核心文件：**
- `csrc/torch_bindings.cpp` — C++ 侧定义和注册
- `vllm/_custom_ops.py` — Python 封装 + fake impl

用 PyTorch 的 `TORCH_LIBRARY` 宏把 C++/CUDA kernel 注册到 `torch.ops._C` 命名空间：

```cpp
// csrc/torch_bindings.cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("paged_attention_v1(...)  -> ()");
    ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);
}
```

Python 端通过 `_custom_ops.py` 提供薄封装，并用 `@register_fake` 注册 meta/fake 实现（给 torch.compile 做 shape inference）：

```python
# vllm/_custom_ops.py
from torch.library import register_fake

if hasattr(torch.ops._C, "awq_dequantize"):
    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(qweight, scales, zeros, split_k_iters, thx, thy):
        in_c = qweight.size(0)
        out_c = qweight.size(1) * 8
        return torch.empty((in_c, out_c), dtype=scales.dtype, device=scales.device)
```

---

## 第二层：`direct_register_custom_op`（轻量 Python 注册）

**核心文件：** `vllm/utils/torch_utils.py:899`

绕过 `torch.library.custom_op` 装饰器的调度开销，直接用 `Library.define()` + `Library.impl()` 注册到 `torch.ops.vllm.*` 命名空间：

```python
# vllm/utils/torch_utils.py
vllm_lib = Library("vllm", "FRAGMENT")

def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    if mutates_args is None:
        mutates_args = []
    if dispatch_key is None:
        from vllm.platforms import current_platform
        dispatch_key = current_platform.dispatch_key

    schema_str = infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
```

### 使用示例

```python
# vllm/model_executor/layers/fused_moe/fused_moe.py
def outplace_fused_experts(...) -> torch.Tensor:
    # 实际计算逻辑
    ...

def outplace_fused_experts_fake(...) -> torch.Tensor:
    return torch.empty_like(hidden_states)  # 只返回正确 shape/dtype

direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    fake_impl=outplace_fused_experts_fake,
)
```

对于 in-place 操作，fake impl 直接是 `pass`：

```python
def inplace_fused_experts_fake(...) -> None:
    pass
```

### 为什么不直接用 `torch.library.custom_op`？

`torch.library.custom_op` 装饰器内部需要处理复杂的调度逻辑，有不可忽略的注册开销。`direct_register_custom_op` 跳过这些中间层，直接调用底层 API，注册速度更快。

---

## 第三层：`CustomOp` 基类（平台调度层）

**核心文件：** `vllm/model_executor/custom_op.py`

这是一个 `nn.Module` 子类，核心功能是**按硬件平台自动 dispatch forward 方法**。

### 基本用法

```python
@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    def forward_native(self, x):   # PyTorch 原生实现（fallback / torch.compile 用）
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x):     # 调用 C++ CUDA kernel
        from vllm import _custom_ops as ops
        out = torch.empty(...)
        ops.silu_and_mul(out, x)
        return out

    def forward_xpu(self, x):      # Intel XPU 实现
        ...
```

### 运行时调度逻辑

`dispatch_forward()` 在 `__init__` 时确定要调用哪个 forward 方法：

1. 检查该 op 是否通过 `--compilation_config.custom_ops` 被启用/禁用
2. **启用** → 根据 `current_platform` 选择 `forward_cuda` / `forward_hip` / `forward_xpu` / `forward_cpu` / `forward_tpu`
3. **禁用** → fallback 到 `forward_native()`，让 Inductor 可以 fuse

```python
def dispatch_forward(self, compile_native: bool):
    enabled = self._enforce_enable or self.enabled()
    if not enabled:
        return self.maybe_compile(self.forward_native, enable=compile_native)
    if current_platform.is_rocm():
        return self.forward_hip
    elif current_platform.is_cpu():
        return self.forward_cpu
    elif current_platform.is_cuda():
        return self.forward_cuda
    # ...
```

### 启用/禁用控制

通过 `--compilation_config.custom_ops` 参数控制：

| 配置值 | 含义 |
|--------|------|
| `["all"]` | 启用所有已注册的 CustomOp |
| `["none"]` | 禁用所有，全部走 `forward_native` |
| `["+silu_and_mul"]` | 额外启用指定 op |
| `["-rms_norm"]` | 额外禁用指定 op |

当 Inductor 作为编译后端时，默认值为 `"none"`，以最大化编译器融合能力。

### 已注册的 CustomOp 示例

| Op 名称 | 文件位置 |
|---------|---------|
| `silu_and_mul`, `gelu_and_mul`, `quick_gelu` | `layers/activation.py` |
| `rms_norm`, `gemma_rms_norm` | `layers/layernorm.py` |
| `apply_rotary_emb` | `layers/rotary_embedding/common.py` |
| `unquantized_fused_moe`, `modular_fused_moe` | `layers/fused_moe/` |
| `short_conv`, `mixer2_gated_rms_norm` | `layers/mamba/` |
| `conv2d`, `conv3d` | `layers/conv.py` |

---

## 第四层：IR Op（编译管线层，最新）

**核心文件：** `vllm/ir/op.py`

面向编译优化管线的更高级抽象，支持多 provider 实现 + 优先级调度：

```python
@vllm.ir.register_op
def my_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y  # native 实现（默认 fallback）

@my_op.register_impl("triton_provider", supported=torch.cuda.is_available())
def triton_impl(x, y):
    ...  # Triton 优化实现
```

特性：
- 一个 native 默认实现 + 多个命名 provider 实现
- `supported` 参数标识当前环境是否可用
- `set_priority()` 设置各 provider 的优先级
- 可选 `torch.compile` 包装（通过 `enable_torch_wrap` 控制）
- 通过 `vllm/compilation/passes/ir/lowering_pass.py` 集成到 Inductor lowering

---

## 第五层：`@torch.library.custom_op`（PyTorch 原生装饰器）

少数场景直接使用 PyTorch 的高级 API：

```python
# vllm/utils/flashinfer.py
@torch.library.custom_op("vllm::flashinfer_mm_fp4", mutates_args=[], device_types="cuda")
def flashinfer_mm_fp4(A, B, A_scale, B_scale, g_scale, dtype, ...) -> torch.Tensor:
    return flashinfer_mm_fp4_(A, B, A_scale, B_scale, g_scale, dtype, ...)

@torch.library.register_fake("vllm::flashinfer_mm_fp4")
def flashinfer_mm_fp4_fake(A, B, A_scale, B_scale, g_scale, dtype, ...) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)
```

---

## OOT（Out-of-Tree）插件机制

硬件厂商（华为 Ascend、Intel Gaudi 等）可以通过 OOT 机制无侵入替换任意 op 实现。

### 注册方式

```python
# 在 OOT 插件中（如 vllm-ascend）
@CustomOp.register_oot("SiluAndMul")
class AscendSiluAndMul(CustomOp):
    def forward_oot(self, x):
        # Ascend NPU 专用实现
        ...
```

### 工作原理

`CustomOp.__new__` 在实例化时检查 `op_registry_oot` 字典：

```python
class CustomOp(nn.Module):
    def __new__(cls, *args, **kwargs):
        op_name = cls.__name__
        if op_name not in op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = op_registry_oot[op_name]
        return super().__new__(op_cls_to_instantiate)
```

模型代码无需任何修改，OOT 插件加载后自动生效。

### PluggableLayer

类似 CustomOp 的 OOT 替换能力，但不提供 `forward_*` 平台调度。适合需要自定义初始化和子模块组合的复杂层。

---

## 命名空间总览

| 命名空间 | Library 对象 | 用途 |
|---------|-------------|------|
| `torch.ops._C.*` | C++ extension | C++ CUDA kernel |
| `torch.ops._moe_C.*` | C++ MoE extension | MoE 专用 C++ kernel |
| `torch.ops.vllm.*` | `Library("vllm", "FRAGMENT")` | Python 端 `direct_register_custom_op` 注册的 op |
| `torch.ops.vllm_helion.*` | `Library("vllm_helion", "FRAGMENT")` | Helion kernel（Triton 变体） |
| `torch.ops.oink.*` | 外部插件 | Blackwell 等专用 op |

---

## 与 torch.compile 的交互总结

| 注册机制 | 编译行为 |
|---------|---------|
| `direct_register_custom_op()` | 图中的不透明节点（graph boundary），fake_impl 提供 shape 信息 |
| `@register_fake("_C::op")` | 同上，C++ op 的 fake 实现 |
| `@torch.library.custom_op` | 同上，但用 PyTorch 高级 API（开销略大） |
| `CustomOp` 类（enabled） | 不透明，走平台 kernel |
| `CustomOp` 类（disabled） | 透明，暴露 `forward_native` 让 Inductor fuse |

**核心设计哲学：** 在"编译器自动融合优化"和"手写 kernel 极致性能"之间提供灵活切换能力，同时通过 OOT 机制让第三方硬件厂商零侵入接入。

---

## 相关文件索引

| 文件 | 说明 |
|------|------|
| `csrc/torch_bindings.cpp` | C++ kernel 注册入口 |
| `vllm/_custom_ops.py` | C++ op 的 Python 封装 + fake impl |
| `vllm/utils/torch_utils.py` | `direct_register_custom_op` 定义 |
| `vllm/model_executor/custom_op.py` | `CustomOp` 基类 + 平台调度 |
| `vllm/ir/op.py` | IR Op 多 provider 调度 |
| `vllm/compilation/passes/ir/lowering_pass.py` | IR Op 的 Inductor lowering |
| `vllm/utils/flashinfer.py` | FlashInfer op 注册示例 |
| `docs/design/custom_op.md` | 官方设计文档 |
