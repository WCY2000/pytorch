"""
TileLang inductor backend — exhaustive op coverage demo.

Tests every op that has a T.v* mapping in tilelang.py:

  _BINARY_VEC_OPS:  add vsub vmul vdiv vmax vmin vpow  (+ scalar-const variants)
  _UNARY_VEC_OPS:   vexp vln vexp2 vlog2 vrelu vsigmoid vsqrt vrsqrt vabs
                    vcos vsin verf vtanh
  special:          neg (→ vmul x -1.0)

Note: bitwise (vand/vor/vxor) require integer tensors which hit the
_assert_ub_dtype guard and fall back to Triton; excluded here.

Run:
    TORCHINDUCTOR_NPU_BACKEND=tilelang python demo_ops.py
"""
import os
import torch
import torch_npu  # noqa: F401

DEVICE = "npu"
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "tilelang"

rtol, atol = 1e-2, 1e-2

PASS_COUNT = 0
FAIL_COUNT = 0


def _check(name, fn, *inputs):
    global PASS_COUNT, FAIL_COUNT
    compiled = torch.compile(fn, backend="inductor")
    ref = fn(*inputs)
    out = compiled(*inputs)
    flat_ref = ref.flatten()
    flat_out = out.flatten()
    vals_ref = [f"{v:.4f}" for v in flat_ref[:4].tolist()]
    vals_out = [f"{v:.4f}" for v in flat_out[:4].tolist()]
    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        err = (out - ref).abs().max().item()
        print(f"  [PASS] {name:<30}  ref={vals_ref}  out={vals_out}  max_err={err:.6f}")
        PASS_COUNT += 1
    except AssertionError as e:
        print(f"  [FAIL] {name:<30}  ref={vals_ref}  out={vals_out}")
        print(f"         {e}")
        FAIL_COUNT += 1


N = 1024
x  = torch.randn(N, device=DEVICE)
xp = torch.rand(N, device=DEVICE) + 0.1        # strictly positive
y  = torch.randn(N, device=DEVICE)
yp = torch.rand(N, device=DEVICE) + 0.1        # strictly positive, for div


# ===========================================================================
# Binary ops:  T.vXXX(A, B, C)  with tensor B
# ===========================================================================
print("\n========== Binary ops (tensor OP tensor) ==========")

_check("add      x+y",      lambda x, y: x + y,                  x, y)
_check("sub      x-y",      lambda x, y: x - y,                  x, y)
_check("mul      x*y",      lambda x, y: x * y,                  x, y)
_check("truediv  x/yp",     lambda x, y: x / y,                  x, yp)
_check("maximum  max(x,y)", lambda x, y: torch.maximum(x, y),    x, y)
_check("minimum  min(x,y)", lambda x, y: torch.minimum(x, y),    x, y)
_check("pow      xp**yp",   lambda x, y: torch.pow(x, y),        xp, yp)


# ===========================================================================
# Binary ops:  T.vXXX(A, scalar, C)  — constant B via _var_consts
# ===========================================================================
print("\n========== Binary ops (tensor OP scalar constant) ==========")

_check("add      x+2.0",    lambda x: x + 2.0,                   x)
_check("sub      x-0.5",    lambda x: x - 0.5,                   x)
_check("mul      x*3.0",    lambda x: x * 3.0,                   x)
_check("truediv  x/4.0",    lambda x: x / 4.0,                   x)
_check("pow      xp**2.0",  lambda x: torch.pow(x, 2.0),         xp)
_check("pow      xp**0.5",  lambda x: torch.pow(x, 0.5),         xp)


# ===========================================================================
# Unary ops:  T.vXXX(A, B)
# ===========================================================================
print("\n========== Unary ops ==========")

_check("exp      exp(x)",    lambda x: torch.exp(x),              x)
_check("log      log(xp)",   lambda x: torch.log(x),              xp)
_check("exp2     exp2(x)",   lambda x: torch.exp2(x),             x)
_check("log2     log2(xp)",  lambda x: torch.log2(x),             xp)
_check("relu     relu(x)",   lambda x: torch.relu(x),             x)
_check("sigmoid  sig(x)",    lambda x: torch.sigmoid(x),          x)
_check("sqrt     sqrt(xp)",  lambda x: torch.sqrt(x),             xp)
_check("rsqrt    rsqrt(xp)", lambda x: torch.rsqrt(x),            xp)
_check("abs      abs(x)",    lambda x: torch.abs(x),              x)
_check("cos      cos(x)",    lambda x: torch.cos(x),              x)
_check("sin      sin(x)",    lambda x: torch.sin(x),              x)
_check("erf      erf(x)",    lambda x: torch.erf(x),              x)
_check("tanh     tanh(x)",   lambda x: torch.tanh(x),             x)
_check("neg      -x",        lambda x: -x,                        x)


# ===========================================================================
# Fused chains — multiple ops in one kernel
# ===========================================================================
print("\n========== Fused chains ==========")

_check("silu     x*sig(x)",
       lambda x: x * torch.sigmoid(x),                            x)

_check("gelu_tanh",
       lambda x: x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
       x)

_check("scale_bias  x*w+b",
       lambda x, y, z: x * y + z,                                x, y, xp)

_check("exp_then_scale  exp(x)*2-1",
       lambda x: torch.exp(x) * 2.0 - 1.0,                       x)

_check("sqrt_relu  relu(sqrt(xp))",
       lambda x: torch.relu(torch.sqrt(x)),                       xp)

_check("log_sigmoid  log(sig(x))",
       lambda x: torch.log(torch.sigmoid(x)),                     x)

_check("abs_sqrt  sqrt(abs(x)+eps)",
       lambda x: torch.sqrt(torch.abs(x) + 1e-6),                x)

_check("neg_exp  exp(-x)",
       lambda x: torch.exp(-x),                                   x)

_check("cos_sin_add  cos(x)+sin(x)",
       lambda x: torch.cos(x) + torch.sin(x),                    x)

_check("tanh_scale  tanh(x*0.5)*2",
       lambda x: torch.tanh(x * 0.5) * 2.0,                      x)


# ===========================================================================
# 2-D tensors — TileLang treats as flat contiguous
# ===========================================================================
print("\n========== 2-D tensors ==========")

x2 = torch.randn(32, 128, device=DEVICE)
y2 = torch.randn(32, 128, device=DEVICE)

_check("add 2D",     lambda x, y: x + y,             x2, y2)
_check("sigmoid 2D", lambda x: torch.sigmoid(x),     x2)
_check("silu 2D",    lambda x: x * torch.sigmoid(x), x2)


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*50}")
print(f"  PASSED: {PASS_COUNT}   FAILED: {FAIL_COUNT}")
print(f"{'='*50}")
