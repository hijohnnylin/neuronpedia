#!/usr/bin/env python3
"""
Microbenchmark for:
1. Device/dtype check overhead
2. torch.compile on slow path loop

Usage:
    uv run python scripts/bench_checks_and_compile.py
"""
import torch
import time
from dataclasses import dataclass


@dataclass
class BenchConfig:
    batch_size: int = 32
    seq_len: int = 128
    hidden_size: int = 4096
    num_unique_vecs: int = 32  # Worst case: all different
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


def setup_tensors(cfg: BenchConfig):
    """Create hidden states and steering vectors."""
    total_tokens = cfg.batch_size * cfg.seq_len
    hidden = torch.randn(total_tokens, cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)

    # Create steering vectors - already on correct device/dtype
    vectors = [
        torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        for _ in range(cfg.num_unique_vecs)
    ]

    # Assign vectors to requests
    request_vecs = [vectors[i % cfg.num_unique_vecs] for i in range(cfg.batch_size)]

    # Token ranges per request
    ranges = [(i * cfg.seq_len, (i + 1) * cfg.seq_len) for i in range(cfg.batch_size)]

    return hidden, request_vecs, ranges


# ============ APPROACH 1: With device/dtype checks (current) ============
def slow_path_with_checks(hidden, request_vecs, ranges):
    """Current implementation with device/dtype checks."""
    h = hidden.clone()
    for vec, (start, end) in zip(request_vecs, ranges):
        # The check that happens in runtime.py
        if vec.device != h.device or vec.dtype != h.dtype:
            vec = vec.to(device=h.device, dtype=h.dtype)
        h[start:end] += vec
    return h


# ============ APPROACH 2: Without checks (vectors pre-validated) ============
def slow_path_no_checks(hidden, request_vecs, ranges):
    """No device/dtype checks - assume vectors are correct."""
    h = hidden.clone()
    for vec, (start, end) in zip(request_vecs, ranges):
        h[start:end] += vec
    return h


# ============ APPROACH 3: torch.compile the loop ============
@torch.compile(dynamic=True)
def _compiled_apply_steering(hidden, vecs_stacked, starts, ends):
    """Compiled steering application."""
    h = hidden.clone()
    for i in range(len(starts)):
        h[starts[i]:ends[i]] += vecs_stacked[i]
    return h


def slow_path_compiled(hidden, request_vecs, ranges):
    """torch.compile version."""
    # Stack vectors and ranges for compiled function
    vecs_stacked = torch.stack(request_vecs)
    starts = torch.tensor([r[0] for r in ranges], device=hidden.device)
    ends = torch.tensor([r[1] for r in ranges], device=hidden.device)
    return _compiled_apply_steering(hidden, vecs_stacked, starts, ends)


# ============ APPROACH 4: Fully unrolled compiled ============
def make_compiled_unrolled(num_requests):
    """Create a compiled function with unrolled loop."""
    # Generate code for unrolled loop
    code = f"""
def _apply(hidden, vecs, ranges):
    h = hidden.clone()
"""
    for i in range(num_requests):
        code += f"    h[ranges[{i}][0]:ranges[{i}][1]] += vecs[{i}]\n"
    code += "    return h\n"

    local_vars = {}
    exec(code, {}, local_vars)
    return torch.compile(local_vars['_apply'], dynamic=True)


# ============ BENCHMARK FUNCTIONS ============
def bench_function(fn, hidden, request_vecs, ranges, num_iters=100, warmup=20):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        fn(hidden, request_vecs, ranges)
    torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(num_iters):
        fn(hidden, request_vecs, ranges)
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6  # μs


def run_check_overhead_benchmark():
    """Measure device/dtype check overhead."""
    print("=" * 80)
    print("Device/Dtype Check Overhead Benchmark")
    print("=" * 80)

    configs = [
        BenchConfig(batch_size=8, num_unique_vecs=8),
        BenchConfig(batch_size=32, num_unique_vecs=32),
        BenchConfig(batch_size=64, num_unique_vecs=64),
        BenchConfig(batch_size=128, num_unique_vecs=128),
    ]

    print(f"\n{'Batch Size':<12} {'With Checks':>15} {'No Checks':>15} {'Overhead':>15} {'% Overhead':>12}")
    print("-" * 70)

    for cfg in configs:
        hidden, request_vecs, ranges = setup_tensors(cfg)

        t_with = bench_function(slow_path_with_checks, hidden, request_vecs, ranges)
        t_without = bench_function(slow_path_no_checks, hidden, request_vecs, ranges)

        overhead = t_with - t_without
        pct = (overhead / t_without) * 100 if t_without > 0 else 0

        print(f"{cfg.batch_size:<12} {t_with:>12.1f} μs {t_without:>12.1f} μs {overhead:>+12.1f} μs {pct:>+10.1f}%")


def run_compile_benchmark():
    """Measure torch.compile benefit."""
    print("\n" + "=" * 80)
    print("torch.compile Benchmark")
    print("=" * 80)

    configs = [
        BenchConfig(batch_size=8, num_unique_vecs=8),
        BenchConfig(batch_size=32, num_unique_vecs=32),
        BenchConfig(batch_size=64, num_unique_vecs=64),
    ]

    print(f"\n{'Batch Size':<12} {'No Compile':>15} {'Compiled':>15} {'Speedup':>12}")
    print("-" * 55)

    for cfg in configs:
        hidden, request_vecs, ranges = setup_tensors(cfg)

        t_no_compile = bench_function(slow_path_no_checks, hidden, request_vecs, ranges)

        # Warmup compiled version (includes compilation time)
        print(f"  Compiling for batch_size={cfg.batch_size}...", end=" ", flush=True)
        for _ in range(5):
            slow_path_compiled(hidden, request_vecs, ranges)
        torch.cuda.synchronize()
        print("done")

        t_compiled = bench_function(slow_path_compiled, hidden, request_vecs, ranges)

        speedup = t_no_compile / t_compiled if t_compiled > 0 else 0

        print(f"{cfg.batch_size:<12} {t_no_compile:>12.1f} μs {t_compiled:>12.1f} μs {speedup:>10.2f}x")


def run_pure_python_overhead():
    """Measure pure Python loop overhead vs CUDA time."""
    print("\n" + "=" * 80)
    print("Python Loop Overhead Analysis")
    print("=" * 80)

    cfg = BenchConfig(batch_size=32, num_unique_vecs=32)
    hidden, request_vecs, ranges = setup_tensors(cfg)

    # Time just the Python loop with no-ops
    def python_loop_only(hidden, request_vecs, ranges):
        h = hidden
        total = 0
        for vec, (start, end) in zip(request_vecs, ranges):
            # Simulate check overhead
            _ = vec.device
            _ = vec.dtype
            _ = h.device
            _ = h.dtype
            total += start + end
        return total

    # Time check expressions only
    def check_expressions(hidden, request_vecs, ranges):
        h = hidden
        for vec, (start, end) in zip(request_vecs, ranges):
            _ = (vec.device != h.device or vec.dtype != h.dtype)
        return None

    # Warmup
    for _ in range(100):
        python_loop_only(hidden, request_vecs, ranges)
        check_expressions(hidden, request_vecs, ranges)

    # Time Python overhead
    num_iters = 10000
    t0 = time.perf_counter()
    for _ in range(num_iters):
        python_loop_only(hidden, request_vecs, ranges)
    t_loop = (time.perf_counter() - t0) / num_iters * 1e6

    t0 = time.perf_counter()
    for _ in range(num_iters):
        check_expressions(hidden, request_vecs, ranges)
    t_checks = (time.perf_counter() - t0) / num_iters * 1e6

    # Time actual CUDA work
    t_cuda = bench_function(slow_path_no_checks, hidden, request_vecs, ranges)

    print(f"\nFor batch_size={cfg.batch_size}, num_vecs={cfg.num_unique_vecs}:")
    print(f"  Python loop overhead (no CUDA):    {t_loop:.2f} μs")
    print(f"  Device/dtype check expressions:    {t_checks:.2f} μs")
    print(f"  Full CUDA steering application:    {t_cuda:.1f} μs")
    print(f"\n  Check overhead as % of total:      {t_checks/t_cuda*100:.2f}%")


def run_attribute_access_microbench():
    """Micro-benchmark tensor attribute access."""
    print("\n" + "=" * 80)
    print("Tensor Attribute Access Microbenchmark")
    print("=" * 80)

    t = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    num_iters = 100000

    # Warmup
    for _ in range(1000):
        _ = t.device
        _ = t.dtype

    # Time device access
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = t.device
    t_device = (time.perf_counter() - t0) / num_iters * 1e9  # ns

    # Time dtype access
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = t.dtype
    t_dtype = (time.perf_counter() - t0) / num_iters * 1e9  # ns

    # Time comparison
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = t.device != t.device
    t_cmp_device = (time.perf_counter() - t0) / num_iters * 1e9  # ns

    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = t.dtype != t.dtype
    t_cmp_dtype = (time.perf_counter() - t0) / num_iters * 1e9  # ns

    print(f"\n  tensor.device access:              {t_device:.1f} ns")
    print(f"  tensor.dtype access:               {t_dtype:.1f} ns")
    print(f"  device != device comparison:       {t_cmp_device:.1f} ns")
    print(f"  dtype != dtype comparison:         {t_cmp_dtype:.1f} ns")
    print(f"\n  Full check (2 accesses + 2 cmps):  ~{2*t_device + 2*t_dtype + t_cmp_device + t_cmp_dtype:.1f} ns")
    print(f"  Per 32 requests:                   ~{32*(2*t_device + 2*t_dtype + t_cmp_device + t_cmp_dtype)/1000:.2f} μs")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    run_attribute_access_microbench()
    run_pure_python_overhead()
    run_check_overhead_benchmark()
    run_compile_benchmark()
