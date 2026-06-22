"""
Benchmark test_softmax from demo.py using torch_npu profiler.

Run:
    TORCHINDUCTOR_NPU_BACKEND=tilelang python demo_softmax.py
"""
import os
import shutil
import sys
import torch
import torch_npu  # noqa: F401

# ---- env setup ----
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "tilelang"
os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/tilelang_cache")

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "../../../tilelang-mlir-ascend"))
from tilelang.profiler.bench import do_bench_npu

DEVICE = "npu"
rtol, atol = 1e-2, 1e-2

_PROF_DIR = os.path.join(os.environ["TILELANG_CACHE_DIR"], "bench_torch_tmp")


def do_bench_torch(fn, warmup=5, rep=30, prof_dir=_PROF_DIR, timeout=120, keep_res=False):
    """
    Benchmark a single callable using torch_npu.profiler.profile.

    Unlike do_bench_npu, this handles ops that produce multiple NPU kernels
    per call (e.g. eager max/sum reductions): all kernel durations within
    each active iteration are summed, then averaged over `rep` iterations.

    Returns: average latency in milliseconds.
    """
    import time
    import pandas as pd

    # warmup outside profiler
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    shutil.rmtree(prof_dir, ignore_errors=True)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ):
        for _ in range(warmup + rep):
            fn()
        torch.npu.synchronize()

    # profiler parses traces asynchronously after the with-block exits;
    # poll until kernel_details.csv appears (or timeout)
    deadline = time.time() + timeout
    kernel_details_file = None
    while time.time() < deadline:
        for root, _, files in os.walk(prof_dir):
            for f in files:
                if f == "kernel_details.csv":
                    kernel_details_file = os.path.join(root, f)
                    break
        if kernel_details_file:
            break
        time.sleep(1)

    if kernel_details_file is None:
        print(f"[do_bench_torch] WARNING: kernel_details.csv not found in {prof_dir} after {timeout}s")
        shutil.rmtree(prof_dir, ignore_errors=True)
        return float("inf")

    df = pd.read_csv(kernel_details_file)
    total_calls = warmup + rep

    # auto-detect how many kernel rows one fn() call produces
    kernels_per_call = max(1, len(df) // total_calls)

    # sum all kernel durations for each active iteration, then average
    active_start = warmup * kernels_per_call
    active_df = df.iloc[active_start: active_start + rep * kernels_per_call]
    avg_us = active_df["Duration(us)"].sum() / rep

    if not keep_res:
        shutil.rmtree(prof_dir, ignore_errors=True)
    return avg_us / 1e3  # ms


def my_softmax(x, temperature=1.0):
    x = x / temperature
    x_max = x.max(dim=-1, keepdim=True)[0]
    x = x - x_max
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def bench_eager_per_op(x, temperature=1.0):
    x_scaled  = x / temperature
    x_max     = x_scaled.max(dim=-1, keepdim=True)[0]
    x_shifted = x_scaled - x_max
    exp_x     = x_shifted.exp()
    exp_sum   = exp_x.sum(dim=-1, keepdim=True)

    ops = [
        ("div (x/temperature)",  lambda: x.div(temperature)),
        ("max (dim=-1)",         lambda: x_scaled.max(dim=-1, keepdim=True)),
        ("sub (x - x_max)",      lambda: x_scaled.sub(x_max)),
        ("exp",                  lambda: x_shifted.exp()),
        ("sum (dim=-1)",         lambda: exp_x.sum(dim=-1, keepdim=True)),
        ("div (exp_x / sum)",    lambda: exp_x.div(exp_sum)),
    ]

    print("\n--- Eager per-op latency (torch_npu profiler) ---")
    total = 0.0
    for name, fn in ops:
        t = do_bench_torch(fn)
        total += t
        print(f"  {name:<25} {t:.4f} ms")
    print(f"  {'total (sum of parts)':<25} {total:.4f} ms")


def main():
    x = torch.randn(128, 512, device=DEVICE)

    compiled = torch.compile(my_softmax, backend="inductor")

    # correctness check
    ref = my_softmax(x)
    out = compiled(x)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print(f"Correctness PASS  max_err={(out - ref).abs().max().item():.6f}")

    # per-op breakdown (eager, each op profiled individually)
    bench_eager_per_op(x)

    # end-to-end
    latency_eager    = do_bench_torch(lambda: my_softmax(x))
    latency_compiled = do_bench_npu([lambda: compiled(x)])

    print(f"\n--- End-to-end latency ---")
    print(f"  Eager    (end-to-end): {latency_eager:.4f} ms")
    print(f"  Compiled (end-to-end): {latency_compiled:.4f} ms")
    print(f"  Speedup: {latency_eager / latency_compiled:.2f}x")


if __name__ == "__main__":
    main()
