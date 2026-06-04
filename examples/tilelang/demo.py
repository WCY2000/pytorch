import os
import sys
import importlib.util
import torch
import torch_npu  # noqa: F401  — 注册 npu device
# ---------------------------------------------------------------------------
# Step 4: 测试用例
# ---------------------------------------------------------------------------
DEVICE = "npu"
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "tilelang"

def test_elementwise_add():
    print("\n--- test_elementwise_add ---")

    def fn(x, y):
        return x + y

    compiled = torch.compile(fn, backend="inductor")
    x = torch.randn(1024, device=DEVICE)
    y = torch.randn(1024, device=DEVICE)
    ref = fn(x, y)
    try:
        out = compiled(x, y)
    except Exception as e:
        print("FULL ERROR:", str(e))
        raise
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("  PASS  max_err =", (out - ref).abs().max().item())


def test_elementwise_exp():
    print("\n--- test_elementwise_exp ---")

    def fn(x):
        return torch.exp(x)

    compiled = torch.compile(fn, backend="inductor")
    x = torch.randn(2048, device=DEVICE)
    ref = fn(x)
    out = compiled(x)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("  PASS  max_err =", (out - ref).abs().max().item())


def test_elementwise_relu():
    print("\n--- test_elementwise_relu ---")

    def fn(x):
        return torch.relu(x)

    compiled = torch.compile(fn, backend="inductor")
    x = torch.randn(4096, device=DEVICE)
    ref = fn(x)
    out = compiled(x)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("  PASS  max_err =", (out - ref).abs().max().item())


def test_fused_chain():
    print("\n--- test_fused_chain  x * 2 + 1 ---")

    def fn(x):
        return x * 2.0 + 1.0

    compiled = torch.compile(fn, backend="inductor")
    x = torch.randn(512, device=DEVICE)
    ref = fn(x)
    out = compiled(x)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("  PASS  max_err =", (out - ref).abs().max().item())


# ---------------------------------------------------------------------------
# Step 5: 运行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_elementwise_add()
    test_elementwise_exp()
    test_elementwise_relu()
    test_fused_chain()
    print("\nAll tests passed!")
