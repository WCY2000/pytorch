"""
TileLang inductor backend — pointwise & mixed kernel demos.

Run:
    TORCHINDUCTOR_NPU_BACKEND=tilelang python demo.py

Kernel classification:
  [TileLang]  pointwise ops → T.vadd / T.vmul / T.vexp / T.vsigmoid / ...
  [Triton]    reductions (max, sum) → fall back to Triton automatically
  [Mixed]     fused kernel split by inductor: pointwise part via TileLang,
              reduction part via Triton
"""
import os
import torch
import torch_npu  # noqa: F401

DEVICE = "npu"
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "tilelang"

rtol, atol = 1e-2, 1e-2   # fp32 on NPU; loosen slightly for transcendentals


def _run(tag, fn, *inputs):
    compiled = torch.compile(fn, backend="inductor")
    ref = fn(*inputs)
    out = compiled(*inputs)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    err = (out - ref).abs().max().item()
    print(f"  PASS  max_err = {err:.6f}  [{tag}]")


# ---------------------------------------------------------------------------
# 1. 纯 pointwise（单算子）
# ---------------------------------------------------------------------------

def test_elementwise_add():
    print("\n--- test_elementwise_add  [TileLang] ---")
    def fn(x, y): return x + y
    x = torch.randn(1024, device=DEVICE)
    y = torch.randn(1024, device=DEVICE)
    _run("T.vadd", fn, x, y)


def test_elementwise_exp():
    print("\n--- test_elementwise_exp  [TileLang] ---")
    def fn(x): return torch.exp(x)
    x = torch.randn(2048, device=DEVICE)
    _run("T.vexp", fn, x)


def test_elementwise_relu():
    print("\n--- test_elementwise_relu  [TileLang] ---")
    def fn(x): return torch.relu(x)
    x = torch.randn(4096, device=DEVICE)
    _run("T.vrelu", fn, x)


def test_elementwise_sigmoid():
    print("\n--- test_elementwise_sigmoid  [TileLang] ---")
    def fn(x): return torch.sigmoid(x)
    x = torch.randn(2048, device=DEVICE)
    _run("T.vsigmoid", fn, x)


def test_elementwise_tanh():
    print("\n--- test_elementwise_tanh  [TileLang] ---")
    def fn(x): return torch.tanh(x)
    x = torch.randn(2048, device=DEVICE)
    _run("T.vtanh", fn, x)


def test_elementwise_sqrt():
    print("\n--- test_elementwise_sqrt  [TileLang] ---")
    def fn(x): return torch.sqrt(x)
    x = torch.rand(2048, device=DEVICE) + 0.1   # positive
    _run("T.vsqrt", fn, x)


def test_elementwise_abs():
    print("\n--- test_elementwise_abs  [TileLang] ---")
    def fn(x): return torch.abs(x)
    x = torch.randn(2048, device=DEVICE)
    _run("T.vabs", fn, x)


# ---------------------------------------------------------------------------
# 2. fused pointwise 链
# ---------------------------------------------------------------------------

def test_fused_chain():
    print("\n--- test_fused_chain  x * 2 + 1  [TileLang] ---")
    def fn(x): return x * 2.0 + 1.0
    x = torch.randn(512, device=DEVICE)
    _run("T.vmul+T.vadd", fn, x)


def test_silu():
    """SiLU / Swish: x * sigmoid(x)  — 两个 TileLang kernel fused"""
    print("\n--- test_silu  x * sigmoid(x)  [TileLang] ---")
    def fn(x): return x * torch.sigmoid(x)
    x = torch.randn(4096, device=DEVICE)
    _run("T.vmul+T.vsigmoid", fn, x)


def test_gelu_tanh():
    """GELU (tanh approximation): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))"""
    print("\n--- test_gelu_tanh  [TileLang] ---")
    def fn(x):
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    x = torch.randn(4096, device=DEVICE)
    _run("fused pointwise", fn, x)


def test_fused_scale_shift():
    """逐元素 scale + shift：常见于 LayerNorm/BatchNorm 的 affine 步骤"""
    print("\n--- test_fused_scale_shift  x * w + b  [TileLang] ---")
    def fn(x, w, b): return x * w + b
    N = 2048
    x = torch.randn(N, device=DEVICE)
    w = torch.randn(N, device=DEVICE)
    b = torch.randn(N, device=DEVICE)
    _run("T.vmul+T.vadd", fn, x, w, b)


def test_2d_add():
    """2D 连续 tensor — TileLang 把整个 tensor 展平处理"""
    print("\n--- test_2d_add  [B,H] + [B,H]  [TileLang] ---")
    def fn(x, y): return x + y
    x = torch.randn(32, 128, device=DEVICE)
    y = torch.randn(32, 128, device=DEVICE)
    _run("T.vadd 2D", fn, x, y)


# ---------------------------------------------------------------------------
# 3. 含 reduction 的多 kernel 算子（TileLang + Triton 混合）
# ---------------------------------------------------------------------------

def test_softmax():
    """
    Softmax 被 inductor 分解为多个 kernel：
      kernel-1: x / temperature         [TileLang pointwise]
      kernel-2: x.max(dim=-1)           [Triton reduction]
      kernel-3: x - x_max, exp(x)       [TileLang pointwise]
      kernel-4: exp_x.sum(dim=-1)       [Triton reduction]
      kernel-5: exp_x / sum             [TileLang pointwise]
    """
    print("\n--- test_softmax  [Mixed: TileLang + Triton] ---")
    def my_softmax(x, temperature=1.0):
        x = x / temperature
        x_max = x.max(dim=-1, keepdim=True)[0]
        x = x - x_max
        exp_x = x.exp()
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    x = torch.randn(128, 512, device=DEVICE)
    compiled = torch.compile(my_softmax, backend="inductor")
    ref = my_softmax(x)
    out = compiled(x)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print(f"  PASS  max_err = {(out - ref).abs().max().item():.6f}  [Mixed]")


def test_softmax_temperature():
    print("\n--- test_softmax_temperature=0.5  [Mixed: TileLang + Triton] ---")
    def my_softmax(x, temperature=1.0):
        x = x / temperature
        x_max = x.max(dim=-1, keepdim=True)[0]
        x = x - x_max
        exp_x = x.exp()
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    x = torch.randn(64, 256, device=DEVICE)
    compiled = torch.compile(my_softmax, backend="inductor")
    ref = my_softmax(x, temperature=0.5)
    out = compiled(x, temperature=0.5)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print(f"  PASS  max_err = {(out - ref).abs().max().item():.6f}  [Mixed]")


def test_layer_norm_affine():
    """
    手动 LayerNorm（不含 torch.nn.LayerNorm，便于观察 kernel 分解）:
      mean = x.mean(dim=-1)    [Triton reduction]
      var  = x.var(dim=-1)     [Triton reduction]
      x_norm = (x - mean) / sqrt(var + eps)   [TileLang pointwise]
      out = x_norm * w + b     [TileLang pointwise]
    """
    print("\n--- test_layer_norm_affine  [Mixed: TileLang + Triton] ---")
    def fn(x, w, b, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + eps)
        return x_norm * w + b

    B, H = 32, 256
    x = torch.randn(B, H, device=DEVICE)
    w = torch.ones(H, device=DEVICE)
    b = torch.zeros(H, device=DEVICE)
    compiled = torch.compile(fn, backend="inductor")
    ref = fn(x, w, b)
    out = compiled(x, w, b)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print(f"  PASS  max_err = {(out - ref).abs().max().item():.6f}  [Mixed]")


def test_rms_norm():
    """
    RMSNorm: x / sqrt(mean(x^2) + eps) * w
      x^2                          [TileLang pointwise]
      mean(x^2, dim=-1)            [Triton reduction]
      x / sqrt(mean + eps) * w     [TileLang pointwise]
    """
    print("\n--- test_rms_norm  [Mixed: TileLang + Triton] ---")
    def fn(x, w, eps=1e-6):
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
        return x / rms * w

    B, H = 32, 256
    x = torch.randn(B, H, device=DEVICE)
    w = torch.ones(H, device=DEVICE)
    compiled = torch.compile(fn, backend="inductor")
    ref = fn(x, w)
    out = compiled(x, w)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print(f"  PASS  max_err = {(out - ref).abs().max().item():.6f}  [Mixed]")


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 纯 pointwise (TileLang) ---
    test_elementwise_add()
    test_elementwise_exp()
    test_elementwise_relu()
    test_elementwise_sigmoid()
    test_elementwise_tanh()
    test_elementwise_sqrt()
    test_elementwise_abs()

    # --- fused pointwise 链 (TileLang) ---
    test_fused_chain()
    test_silu()
    test_gelu_tanh()
    test_fused_scale_shift()
    test_2d_add()

    # --- 含 reduction (TileLang + Triton 混合) ---
    test_softmax()
    test_softmax_temperature()
    test_layer_norm_affine()
    test_rms_norm()

    print("\nAll tests passed!")
