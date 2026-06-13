"""
TileLang inductor backend — T.gemm (matmul) codegen demo.

Tests the TileLang T.gemm path for aten.mm / aten.matmul / torch.nn.Linear.
T.gemm lowers to hivm.hir.mmadL1 on Ascend NPU.

Supported input dtypes : fp16 (accum fp32), int8 (accum int32)
Tile sizes              : block_M=128, block_N=128, block_K=64

Two test modes are exercised:
  1. Direct  — tilelang.compile() called manually on the generated prim_func.
  2. Inductor — torch.compile(fn, backend="inductor") goes through the full
               TileLang scheduling path (add_tilelang_gemm_choices →
               TileLangScheduling.codegen_template).

Run:
    TORCHINDUCTOR_NPU_BACKEND=tilelang python demo_gemm.py
"""
import os

import tilelang
import tilelang.language as T   # noqa: F401 — needed inside exec'd factory

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "tilelang"

DEVICE = "npu"
rtol, atol = 1e-2, 1e-2

PASS_COUNT = 0
FAIL_COUNT = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(name, fn, *inputs, rtol=rtol, atol=atol):
    """Compile fn with inductor, run both compiled and eager, compare."""
    global PASS_COUNT, FAIL_COUNT
    compiled = torch.compile(fn, backend="inductor")
    ref = fn(*inputs)
    out = compiled(*inputs)
    # cast ref to same dtype as out for fair comparison (mm returns fp32 accum)
    if ref.dtype != out.dtype:
        ref = ref.to(out.dtype)
    flat_ref = ref.flatten()
    flat_out = out.flatten()
    vals_ref = [f"{v:.4f}" for v in flat_ref[:4].tolist()]
    vals_out = [f"{v:.4f}" for v in flat_out[:4].tolist()]
    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        err = (out - ref).abs().max().item()
        print(f"  [PASS] {name:<42}  ref={vals_ref}  out={vals_out}  max_err={err:.2e}")
        PASS_COUNT += 1
    except AssertionError as e:
        print(f"  [FAIL] {name:<42}  ref={vals_ref}  out={vals_out}")
        print(f"         {e}")
        FAIL_COUNT += 1


def _check_direct(name, compiled_fn, A, B, C_out, ref_fn):
    """Run a pre-compiled tilelang kernel, compare to ref."""
    global PASS_COUNT, FAIL_COUNT
    C_out.zero_()
    compiled_fn(A, B, C_out)
    torch.npu.synchronize()
    ref = ref_fn(A, B)
    flat_ref = ref.flatten()
    flat_out = C_out.flatten()
    vals_ref = [f"{v:.4f}" for v in flat_ref[:4].tolist()]
    vals_out = [f"{v:.4f}" for v in flat_out[:4].tolist()]
    try:
        torch.testing.assert_close(C_out, ref, rtol=rtol, atol=atol)
        err = (C_out - ref).abs().max().item()
        print(f"  [PASS] {name:<42}  ref={vals_ref}  out={vals_out}  max_err={err:.2e}")
        PASS_COUNT += 1
    except AssertionError as e:
        print(f"  [FAIL] {name:<42}  ref={vals_ref}  out={vals_out}")
        print(f"         {e}")
        FAIL_COUNT += 1


# ---------------------------------------------------------------------------
# T.gemm prim_func generator  (mirrors codegen_tilelang_mm_src in tilelang.py)
# ---------------------------------------------------------------------------

_TORCH_TO_TL = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int8:    "int8",
    torch.int32:   "int32",
}

_GEMM_ACCUM = {
    torch.float16: torch.float32,
    torch.int8:    torch.int32,
}


def _tl(dt):
    return _TORCH_TO_TL[dt]


def make_gemm_prim_fn(M, N, K, *, block_M=128, block_N=128, block_K=64,
                      dtype=torch.float16):
    """
    Build and compile a TileLang @T.prim_func for C = A @ B.

      A : (M, K)  dtype
      B : (K, N)  dtype
      C : (M, N)  accum_dtype  (fp32 for fp16, int32 for int8)

    Returns the compiled callable.
    """
    accum_dtype = _GEMM_ACCUM[dtype]
    td, ta = _tl(dtype), _tl(accum_dtype)
    name = f"gemm_prim_{M}x{N}x{K}"

    src = (
        "import tilelang.language as T\n"
        f"_block_M = {block_M}\n"
        f"_block_N = {block_N}\n"
        f"_block_K = {block_K}\n"
        "@T.prim_func\n"
        f"def {name}(\n"
        f"    A: T.Tensor(({M}, {K}), '{td}'),\n"
        f"    B: T.Tensor(({K}, {N}), '{td}'),\n"
        f"    C: T.Tensor(({M}, {N}), '{ta}'),\n"
        "):\n"
        "    with T.Kernel(\n"
        "        T.ceildiv(_N, _block_N) * T.ceildiv(_M, _block_M),\n"
        "        is_npu=True) as (cid, _):\n"
        "        by = cid // T.ceildiv(_N, _block_N)\n"
        "        bx = cid % T.ceildiv(_N, _block_N)\n"
        "\n"
        f"        A_shared = T.alloc_shared((_block_M, _block_K), '{td}')\n"
        f"        B_shared = T.alloc_shared((_block_K, _block_N), '{td}')\n"
        f"        C_local  = T.alloc_fragment((_block_M, _block_N), '{ta}')\n"
        "\n"
        "        for k in T.Pipelined(T.ceildiv(_K, _block_K), num_stages=2):\n"
        "            T.copy(A[by * _block_M, k * _block_K], A_shared)\n"
        "            T.copy(B[k * _block_K, bx * _block_N], B_shared)\n"
        "            T.gemm(A_shared, B_shared, C_local, initC=(k == 0))\n"
        "\n"
        "        T.copy(C_local, C[by * _block_M, bx * _block_N])\n"
    )
    # Replace symbolic dims
    src = src.replace("_M", str(M)).replace("_N", str(N)).replace("_K", str(K))

    ns = {}
    exec(src, ns)
    prim_fn = ns[name]

    print(f"\n  --- Generated @T.prim_func for {M}×{K} @ {K}×{N} ---")
    print(src)

    return tilelang.compile(prim_fn, target="npuir")


# ===========================================================================
# Section 1 — Direct tilelang.compile tests
# ===========================================================================
print("\n========== Direct T.gemm (tilelang.compile) fp16 ==========")

for M, N, K in [(512, 512, 256), (1024, 1024, 512), (256, 512, 128)]:
    compiled = make_gemm_prim_fn(M, N, K, dtype=torch.float16)
    A = torch.randn(M, K, dtype=torch.float16).npu()
    B = torch.randn(K, N, dtype=torch.float16).npu()
    C = torch.zeros(M, N, dtype=torch.float32).npu()
    _check_direct(
        f"fp16  {M}x{K} @ {K}x{N} → fp32",
        compiled, A, B, C,
        lambda a, b: torch.mm(a.float(), b.float()),
    )


# ===========================================================================
# Section 2 — torch.compile / inductor  (TileLangScheduling.codegen_template)
# ===========================================================================
print("\n========== torch.compile inductor — aten.mm (fp16) ==========")

# Square shapes
for sz in [256, 512, 1024]:
    A = torch.randn(sz, sz, dtype=torch.float16, device=DEVICE)
    B = torch.randn(sz, sz, dtype=torch.float16, device=DEVICE)
    _check(f"mm   fp16  {sz}x{sz} @ {sz}x{sz}",
           lambda a, b: torch.mm(a, b), A, B)

# Non-square shapes
for M, N, K in [(256, 512, 128), (128, 1024, 256), (512, 128, 256)]:
    A = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    B = torch.randn(K, N, dtype=torch.float16, device=DEVICE)
    _check(f"mm   fp16  {M}x{K} @ {K}x{N}",
           lambda a, b: torch.mm(a, b), A, B)


print("\n========== torch.compile inductor — aten.matmul (fp16) ==========")

for M, N, K in [(512, 512, 256), (256, 1024, 512)]:
    A = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    B = torch.randn(K, N, dtype=torch.float16, device=DEVICE)
    _check(f"matmul fp16  {M}x{K} @ {K}x{N}",
           lambda a, b: torch.matmul(a, b), A, B)


print("\n========== torch.compile inductor — torch.nn.Linear (fp16) ==========")
# nn.Linear uses aten.addmm (bias present) or aten.mm (no bias)

for in_f, out_f, batch in [(256, 512, 128), (512, 1024, 64)]:
    linear_no_bias = torch.nn.Linear(in_f, out_f, bias=False).half().to(DEVICE)
    linear_bias    = torch.nn.Linear(in_f, out_f, bias=True).half().to(DEVICE)
    x = torch.randn(batch, in_f, dtype=torch.float16, device=DEVICE)

    _check(f"Linear no-bias  {batch}x{in_f} → {batch}x{out_f}",
           linear_no_bias, x)
    _check(f"Linear bias     {batch}x{in_f} → {batch}x{out_f}",
           linear_bias, x)


print("\n========== torch.compile inductor — fused mm+activation (fp16) ==========")
# Common patterns: mm → relu, mm → gelu, mm → sigmoid

M, N, K = 512, 512, 256
A = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
B = torch.randn(K, N, dtype=torch.float16, device=DEVICE)

_check("mm + relu    fp16",
       lambda a, b: torch.relu(torch.mm(a, b)), A, B)

_check("mm + sigmoid fp16",
       lambda a, b: torch.sigmoid(torch.mm(a, b)), A, B)

_check("mm + tanh    fp16",
       lambda a, b: torch.tanh(torch.mm(a, b)), A, B)

_check("mm + scale+bias  fp16",
       lambda a, b: torch.mm(a, b) * 0.5 + 1.0, A, B)


print("\n========== torch.compile inductor — batched matmul (fp16) ==========")
# aten.bmm — inductor may fuse to addmm or keep as bmm

for batch, M, N, K in [(4, 256, 256, 128), (8, 128, 512, 256)]:
    A3 = torch.randn(batch, M, K, dtype=torch.float16, device=DEVICE)
    B3 = torch.randn(batch, K, N, dtype=torch.float16, device=DEVICE)
    _check(f"bmm  fp16  {batch}x{M}x{K} @ {batch}x{K}x{N}",
           lambda a, b: torch.bmm(a, b), A3, B3)


# ===========================================================================
# Summary
# ===========================================================================
total = PASS_COUNT + FAIL_COUNT
print(f"\n{'='*60}")
print(f"  PASSED: {PASS_COUNT}/{total}    FAILED: {FAIL_COUNT}/{total}")
print(f"{'='*60}")
