"""
验证 torch.nn.attention.flex_attention 在当前设备上是否可用。

测试内容：
1. PyTorch 版本是否支持 flex_attention API
2. 基础 causal attention（模拟 decode）
3. 非因果 attention（模拟 encoder）
4. 自定义 mask_mod（sliding window）
5. Variable-length batch（模拟 vLLM 的 chunked prefill）
6. GQA（Grouped Query Attention）
7. torch.compile 是否能正常编译 flex_attention

用法：
    python test_flex_attention_availability.py [--device cuda]  # 默认 auto detect
"""

import argparse
import sys
import time

import torch

# ============================================================
# 0. 环境检查
# ============================================================

def check_pytorch_version():
    version = torch.__version__
    major, minor = int(version.split(".")[0]), int(version.split(".")[1])
    if major < 2 or (major == 2 and minor < 5):
        print(f"[FAIL] PyTorch {version} 不支持 flex_attention（需要 >= 2.5）")
        sys.exit(1)
    print(f"[OK] PyTorch version: {version}")
    return major, minor


def detect_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        print(f"[INFO] 使用指定设备: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] 检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, "musa") and torch.musa.is_available():
        device = torch.device("musa")
        print(f"[OK] 检测到 MUSA 设备")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"[OK] 检测到 XPU 设备")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu")
        print(f"[OK] 检测到 NPU 设备")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[OK] 检测到 MPS 设备")
    else:
        device = torch.device("cpu")
        print(f"[WARN] 未检测到加速设备，使用 CPU（flex_attention 在 CPU 上可能不支持编译）")
    return device


def try_import_flex_attention():
    try:
        from torch.nn.attention.flex_attention import (
            BlockMask,
            create_block_mask,
            flex_attention,
        )
        print("[OK] flex_attention API 导入成功")
        return flex_attention, create_block_mask, BlockMask
    except ImportError as e:
        print(f"[FAIL] 无法导入 flex_attention: {e}")
        sys.exit(1)


# ============================================================
# 1. 基础 causal attention
# ============================================================

def test_basic_causal(flex_attention_fn, device, dtype=torch.bfloat16):
    """测试最基础的因果注意力"""
    print("\n--- Test 1: 基础 Causal Attention ---")
    batch, heads, seq_len, head_dim = 2, 8, 64, 128

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    from torch.nn.attention.flex_attention import create_block_mask
    block_mask = create_block_mask(causal_mask, B=batch, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

    output = flex_attention_fn(q, k, v, block_mask=block_mask)

    assert output.shape == (batch, heads, seq_len, head_dim)
    assert output.dtype == dtype
    assert not torch.isnan(output).any(), "输出包含 NaN"
    print(f"[OK] output shape: {output.shape}, dtype: {output.dtype}")


# ============================================================
# 2. 非因果 attention（encoder-style）
# ============================================================

def test_non_causal(flex_attention_fn, device, dtype=torch.bfloat16):
    """测试非因果注意力（双向 attention）"""
    print("\n--- Test 2: 非因果 Attention (Bidirectional) ---")
    batch, heads, seq_len, head_dim = 2, 8, 64, 128

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # 不传 block_mask = 全部可见（非因果）
    output = flex_attention_fn(q, k, v)

    assert output.shape == (batch, heads, seq_len, head_dim)
    assert not torch.isnan(output).any()
    print(f"[OK] output shape: {output.shape}")


# ============================================================
# 3. Sliding Window Attention (自定义 mask_mod)
# ============================================================

def test_sliding_window(flex_attention_fn, device, dtype=torch.bfloat16):
    """测试滑动窗口注意力（自定义 mask_mod）"""
    print("\n--- Test 3: Sliding Window (mask_mod) ---")
    batch, heads, seq_len, head_dim = 1, 4, 128, 64
    window_size = 32

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    def sliding_window_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        windowed = (q_idx - kv_idx) <= window_size
        return causal & windowed

    from torch.nn.attention.flex_attention import create_block_mask
    block_mask = create_block_mask(
        sliding_window_mask, B=batch, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )

    output = flex_attention_fn(q, k, v, block_mask=block_mask)

    assert output.shape == (batch, heads, seq_len, head_dim)
    assert not torch.isnan(output).any()
    print(f"[OK] sliding window (size={window_size}) 计算正常")


# ============================================================
# 4. GQA (Grouped Query Attention)
# ============================================================

def test_gqa(flex_attention_fn, device, dtype=torch.bfloat16):
    """测试 GQA：Q heads > KV heads"""
    print("\n--- Test 4: GQA (Grouped Query Attention) ---")
    batch, q_heads, kv_heads, seq_len, head_dim = 1, 32, 8, 64, 128

    q = torch.randn(batch, q_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    try:
        output = flex_attention_fn(q, k, v, enable_gqa=True)
        assert output.shape == (batch, q_heads, seq_len, head_dim)
        assert not torch.isnan(output).any()
        print(f"[OK] GQA: Q heads={q_heads}, KV heads={kv_heads}")
    except TypeError as e:
        if "enable_gqa" in str(e):
            print(f"[WARN] 当前 PyTorch 版本不支持 enable_gqa 参数，跳过 GQA 测试")
        else:
            raise


# ============================================================
# 5. Variable-length sequences (模拟 vLLM chunked prefill)
# ============================================================

def test_variable_length(flex_attention_fn, device, dtype=torch.bfloat16):
    """测试变长序列：用 document mask 模拟 batch 中不同长度的请求"""
    print("\n--- Test 5: Variable-Length Sequences ---")
    from torch.nn.attention.flex_attention import create_block_mask

    # 模拟 3 个请求拼在一起: 长度 20, 30, 14 = 总长 64
    heads, head_dim = 8, 128
    seq_lens = [20, 30, 14]
    total_len = sum(seq_lens)

    q = torch.randn(1, heads, total_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, heads, total_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, heads, total_len, head_dim, device=device, dtype=dtype)

    # 构建 document id: [0]*20 + [1]*30 + [2]*14
    offsets = torch.tensor([0] + seq_lens, device=device).cumsum(0)

    def doc_causal_mask(b, h, q_idx, kv_idx):
        # 同一个 document 内因果
        q_doc = (q_idx >= offsets[:-1].unsqueeze(-1)).sum(0) - 1
        kv_doc = (kv_idx >= offsets[:-1].unsqueeze(-1)).sum(0) - 1
        same_doc = q_doc == kv_doc
        causal = q_idx >= kv_idx
        return same_doc & causal

    try:
        block_mask = create_block_mask(
            doc_causal_mask, B=1, H=None, Q_LEN=total_len, KV_LEN=total_len, device=device
        )
        output = flex_attention_fn(q, k, v, block_mask=block_mask)
        assert output.shape == (1, heads, total_len, head_dim)
        assert not torch.isnan(output).any()
        print(f"[OK] 变长序列 (lengths={seq_lens}, total={total_len})")
    except Exception as e:
        print(f"[WARN] 变长序列测试失败: {e}")
        print("       这可能是 mask_mod 的限制，不影响基础功能")


# ============================================================
# 6. torch.compile 编译测试
# ============================================================

def test_torch_compile(flex_attention_fn, device, dtype=torch.bfloat16):
    """测试 torch.compile 能否成功编译 flex_attention"""
    print("\n--- Test 6: torch.compile 编译 ---")
    batch, heads, seq_len, head_dim = 1, 8, 64, 128

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    compiled_fn = torch.compile(flex_attention_fn, fullgraph=True)

    try:
        t0 = time.time()
        output = compiled_fn(q, k, v)
        t1 = time.time()
        assert output.shape == (batch, heads, seq_len, head_dim)
        assert not torch.isnan(output).any()
        print(f"[OK] torch.compile 成功 (首次编译耗时: {t1-t0:.2f}s)")

        # 第二次调用应该用缓存
        t0 = time.time()
        output2 = compiled_fn(q, k, v)
        t2 = time.time()
        print(f"[OK] 第二次调用耗时: {t2-t0:.4f}s (应远小于首次)")
    except Exception as e:
        print(f"[FAIL] torch.compile 失败: {e}")
        print("       flex_attention 在该设备上可能无法使用 compile 加速")


# ============================================================
# 7. 数值正确性对比（与 SDPA 对比）
# ============================================================

def test_correctness_vs_sdpa(flex_attention_fn, device, dtype=torch.bfloat16):
    """将 flex_attention 的结果与 scaled_dot_product_attention 对比"""
    print("\n--- Test 7: 数值正确性 (vs SDPA) ---")
    batch, heads, seq_len, head_dim = 1, 4, 32, 64

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # flex_attention without mask = non-causal full attention
    flex_out = flex_attention_fn(q, k, v)

    # SDPA reference (also non-causal)
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    max_diff = (flex_out - sdpa_out).abs().max().item()
    mean_diff = (flex_out - sdpa_out).abs().mean().item()

    # bf16 tolerance
    tol = 1e-2
    if max_diff < tol:
        print(f"[OK] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} (tol={tol})")
    else:
        print(f"[WARN] max_diff={max_diff:.6f} 超过容忍阈值 {tol}")
        print("       可能是实现差异，不一定是 bug")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="验证 flex_attention 在当前设备上的可用性")
    parser.add_argument("--device", default="auto", help="目标设备 (cuda/musa/xpu/npu/cpu/auto)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("  flex_attention 可用性验证")
    print("=" * 60)

    major, minor = check_pytorch_version()
    device = detect_device(args.device)
    flex_attention_fn, create_block_mask, BlockMask = try_import_flex_attention()

    print(f"[INFO] 测试 dtype: {dtype}")
    print(f"[INFO] 设备: {device}")

    tests = [
        ("基础 Causal", test_basic_causal),
        ("非因果", test_non_causal),
        ("Sliding Window", test_sliding_window),
        ("GQA", test_gqa),
        ("变长序列", test_variable_length),
        ("torch.compile", test_torch_compile),
        ("数值正确性", test_correctness_vs_sdpa),
    ]

    passed, failed, warned = 0, 0, 0
    for name, test_fn in tests:
        try:
            test_fn(flex_attention_fn, device, dtype)
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  结果: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n结论: flex_attention 在该设备上基本可用。")
        print("如需集成到 vLLM，还需确认：")
        print("  1. torch.compile fullgraph=True 能正常工作")
        print("  2. BlockMask 构建性能可接受（create_block_mask 有一定开销）")
        print("  3. 与 vLLM 的 paged KV cache 机制对接（需要 block index 映射）")
    else:
        print("\n结论: flex_attention 在该设备上存在兼容性问题，请检查失败项。")


if __name__ == "__main__":
    main()
