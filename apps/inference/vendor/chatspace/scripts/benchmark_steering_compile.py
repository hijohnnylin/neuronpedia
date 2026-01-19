"""Benchmark torch.compile for steering operations.

Compares:
1. Compiled vs uncompiled steering ops
2. Fast path (uniform batch) vs slow path (heterogeneous)
3. steerllm vs chatspace runtime implementations

Usage:
    uv run python scripts/benchmark_steering_compile.py
"""

import os
import sys
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable

# Add parent dir to path for steerllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Import steerllm runtime
from steerllm.backends.vllm import runtime as steer_rt

# ============================================================================
# Benchmark utilities
# ============================================================================

@dataclass
class BenchResult:
    name: str
    mean_us: float
    std_us: float
    iterations: int


def bench_fn(fn: Callable, warmup: int = 10, iters: int = 100) -> BenchResult:
    """Benchmark a function, return mean/std in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # microseconds

    return BenchResult(
        name="",
        mean_us=np.mean(times),
        std_us=np.std(times),
        iterations=iters,
    )


def print_table(results: list[BenchResult], title: str):
    """Print results as a table."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Name':<40} | {'Mean (μs)':<12} | {'Std (μs)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<40} | {r.mean_us:<12.2f} | {r.std_us:<10.2f}")
    print()


# ============================================================================
# Test individual op performance (baseline, no compilation)
# ============================================================================

def bench_individual_ops():
    """Benchmark individual steering operations (uncompiled baseline)."""
    print("\n[1] Benchmarking individual steering ops (uncompiled baseline)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Create test tensors
    hidden_size = 4096
    results = []

    # Test at different sequence lengths
    for seq_len in [256, 1024, 4096]:
        hidden = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
        vec = torch.randn(hidden_size, device=device, dtype=dtype)
        vec = vec / vec.norm()

        # --- Additive steering ---
        def bench_add():
            return hidden + vec

        r = bench_fn(bench_add)
        r.name = f"Add (seq_len={seq_len})"
        results.append(r)

        # --- Projection cap ---
        cap_config = steer_rt._ProjectionCapConfig(unit_vector=vec, min=-0.5, max=0.5)

        def bench_cap():
            return steer_rt._apply_projection_cap(hidden, cap_config)

        r = bench_fn(bench_cap)
        r.name = f"Projection cap (seq_len={seq_len})"
        results.append(r)

        # --- Ablation ---
        ablation_config = steer_rt._AblationConfig(unit_vector=vec, scale=0.5)

        def bench_ablation():
            return steer_rt._apply_ablation(hidden, ablation_config)

        r = bench_fn(bench_ablation)
        r.name = f"Ablation (seq_len={seq_len})"
        results.append(r)

    print_table(results, "Individual Op Performance (Uncompiled)")
    return results


# ============================================================================
# Test fast path vs slow path
# ============================================================================

def bench_fast_vs_slow_path():
    """Benchmark uniform (fast) vs heterogeneous (slow) batch steering."""
    print("\n[2] Benchmarking fast path vs slow path (with loop-level compilation)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    hidden_size = 4096  # Larger to match real models
    num_requests = 32
    seq_len_per_request = 64
    total_tokens = num_requests * seq_len_per_request

    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    vec = torch.randn(hidden_size, device=device, dtype=dtype)
    vec = vec / vec.norm()

    # Create mock state
    state = steer_rt._SteeringState(
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )

    # Create layer spec
    class LayerSpec:
        pass

    layer_spec = LayerSpec()
    layer_spec.operations = [("cap", vec, (-0.5, 0.5))]  # Use cap for more realistic benchmark

    # Create identical specs for all requests (fast path)
    class Spec:
        pass

    uniform_spec = Spec()
    uniform_spec.layers = {0: layer_spec}

    # Register same spec for all requests (fast path - uses identity check)
    request_ids = [f"req_{i}" for i in range(num_requests)]
    seq_lens = [seq_len_per_request] * num_requests

    for req_id in request_ids:
        state.request_steering_specs[req_id] = uniform_spec  # Same object!

    results = []

    # Fast path benchmark (uniform batch - single batched op)
    def bench_fast_path():
        h = hidden.clone()
        # Check uniformity
        first_layer_spec = None
        all_same = True
        for req_id in request_ids:
            spec = state.request_steering_specs.get(req_id)
            if spec and 0 in spec.layers:
                ls = spec.layers[0]
                if first_layer_spec is None:
                    first_layer_spec = ls
                elif ls is not first_layer_spec:
                    all_same = False
                    break

        if all_same and first_layer_spec:
            total = sum(seq_lens)
            h[:total] = steer_rt._apply_layer_steering_to_hidden(h[:total], first_layer_spec, state)
        return h

    r = bench_fn(bench_fast_path)
    r.name = f"Fast path (uniform, {num_requests} reqs)"
    results.append(r)

    # Slow path benchmark - create separate spec objects (heterogeneous)
    layer_specs_list = []
    for i, req_id in enumerate(request_ids):
        separate_spec = Spec()
        separate_layer_spec = LayerSpec()
        separate_layer_spec.operations = [("cap", vec, (-0.5, 0.5))]
        separate_spec.layers = {0: separate_layer_spec}
        state.request_steering_specs[req_id] = separate_spec  # Different objects
        layer_specs_list.append(separate_layer_spec)

    # Test slow path (loop-based) uncompiled
    steer_rt._COMPILE_STEERING = False

    def bench_slow_path_uncompiled():
        h = hidden.clone()
        return steer_rt._slow_path_loop_impl(h, seq_lens, layer_specs_list)

    r = bench_fn(bench_slow_path_uncompiled)
    r.name = f"Slow path loop (uncompiled, {num_requests} reqs)"
    results.append(r)

    # Test slow path (loop-based) compiled
    steer_rt._COMPILE_STEERING = True
    steer_rt._compiled_slow_path = None  # Force recompilation

    def bench_slow_path_compiled():
        h = hidden.clone()
        slow_path_fn = steer_rt._get_compiled_slow_path()
        return slow_path_fn(h, seq_lens, layer_specs_list)

    r = bench_fn(bench_slow_path_compiled)
    r.name = f"Slow path loop (compiled, {num_requests} reqs)"
    results.append(r)

    # Test gather/scatter uncompiled
    steer_rt._COMPILE_STEERING = False

    def bench_gather_scatter_uncompiled():
        h = hidden.clone()
        return steer_rt._gather_scatter_steering(h, seq_lens, layer_specs_list)

    r = bench_fn(bench_gather_scatter_uncompiled)
    r.name = f"Gather/scatter (uncompiled, {num_requests} reqs)"
    results.append(r)

    # Test gather/scatter compiled
    steer_rt._COMPILE_STEERING = True
    steer_rt._compiled_gather_scatter = None  # Force recompilation

    def bench_gather_scatter_compiled():
        h = hidden.clone()
        gather_fn = steer_rt._get_compiled_gather_scatter()
        return gather_fn(h, seq_lens, layer_specs_list)

    r = bench_fn(bench_gather_scatter_compiled)
    r.name = f"Gather/scatter (compiled, {num_requests} reqs)"
    results.append(r)

    print_table(results, "Fast Path vs Slow Path vs Gather/Scatter")

    # Calculate speedups
    fast_time = results[0].mean_us
    slow_uncompiled = results[1].mean_us
    slow_compiled = results[2].mean_us
    gather_uncompiled = results[3].mean_us
    gather_compiled = results[4].mean_us

    print(f"  Fast path vs slow uncompiled: {slow_uncompiled / fast_time:.1f}x faster")
    print(f"  Slow path compiled vs uncompiled: {slow_uncompiled / slow_compiled:.2f}x faster")
    print(f"  Gather/scatter uncompiled vs slow uncompiled: {slow_uncompiled / gather_uncompiled:.2f}x faster")
    print(f"  Gather/scatter compiled vs slow compiled: {slow_compiled / gather_compiled:.2f}x faster")
    print(f"  Gather/scatter compiled vs slow uncompiled: {slow_uncompiled / gather_compiled:.2f}x faster")

    return results


# ============================================================================
# Compare steerllm vs chatspace runtimes
# ============================================================================

def bench_steerllm_vs_chatspace():
    """Compare steerllm and chatspace runtime implementations."""
    print("\n[3] Benchmarking steerllm vs chatspace runtime")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    hidden_size = 4096
    seq_len = 512
    hidden = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    vec = torch.randn(hidden_size, device=device, dtype=dtype)
    vec = vec / vec.norm()

    results = []

    # steerllm projection cap (uncompiled - individual ops aren't compiled anymore)
    steer_config = steer_rt._ProjectionCapConfig(unit_vector=vec, min=-0.5, max=0.5)

    def bench_steerllm_cap():
        return steer_rt._apply_projection_cap(hidden, steer_config)

    r = bench_fn(bench_steerllm_cap)
    r.name = "steerllm: projection cap"
    results.append(r)

    # chatspace projection cap
    try:
        from chatspace.vllm_steering import runtime as chat_rt

        chat_config = chat_rt._ProjectionCapConfig(unit_vector=vec, min=-0.5, max=0.5)

        def bench_chatspace_cap():
            return chat_rt._apply_projection_cap(hidden, chat_config)

        r = bench_fn(bench_chatspace_cap)
        r.name = "chatspace: projection cap"
        results.append(r)
    except ImportError:
        print("  [SKIP] chatspace runtime not available")

    # steerllm ablation
    steer_ablation_config = steer_rt._AblationConfig(unit_vector=vec, scale=0.5)

    def bench_steerllm_ablation():
        return steer_rt._apply_ablation(hidden, steer_ablation_config)

    r = bench_fn(bench_steerllm_ablation)
    r.name = "steerllm: ablation"
    results.append(r)

    # chatspace ablation
    try:
        from chatspace.vllm_steering import runtime as chat_rt

        chat_ablation_config = chat_rt._AblationConfig(unit_vector=vec, scale=0.5)

        def bench_chatspace_ablation():
            return chat_rt._apply_ablation(hidden, chat_ablation_config)

        r = bench_fn(bench_chatspace_ablation)
        r.name = "chatspace: ablation"
        results.append(r)
    except ImportError:
        pass

    print_table(results, "steerllm vs chatspace Runtime")
    return results


# ============================================================================
# Loop-level compilation warmup overhead
# ============================================================================

def bench_compilation_overhead():
    """Measure first-call compilation overhead for loop-level compilation."""
    print("\n[4] Measuring loop-level compilation warmup overhead")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    hidden_size = 4096
    num_requests = 32
    seq_len_per = 64
    total_tokens = num_requests * seq_len_per

    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    vec = torch.randn(hidden_size, device=device, dtype=dtype)
    vec = vec / vec.norm()

    seq_lens = [seq_len_per] * num_requests

    # Create layer specs
    class LayerSpec:
        pass

    layer_specs = []
    for _ in range(num_requests):
        ls = LayerSpec()
        ls.operations = [("cap", vec, (-0.5, 0.5))]
        layer_specs.append(ls)

    # Clear cache to force recompilation
    steer_rt._compiled_slow_path = None
    steer_rt._COMPILE_STEERING = True

    # First call (includes compilation)
    torch.cuda.synchronize()
    start = time.perf_counter()
    slow_path_fn = steer_rt._get_compiled_slow_path()
    h = hidden.clone()
    _ = slow_path_fn(h, seq_lens, layer_specs)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000

    # Subsequent calls
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        h = hidden.clone()
        start = time.perf_counter()
        _ = slow_path_fn(h, seq_lens, layer_specs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_subsequent_ms = np.mean(times)

    print(f"  First call (incl. compilation): {first_call_ms:.2f} ms")
    print(f"  Subsequent calls (avg):         {avg_subsequent_ms:.4f} ms")
    print(f"  Compilation overhead:           {first_call_ms - avg_subsequent_ms:.2f} ms")
    print(f"  Amortization point:             {int(first_call_ms / avg_subsequent_ms)} calls")

    return {
        "first_call_ms": first_call_ms,
        "avg_subsequent_ms": avg_subsequent_ms,
        "compilation_overhead_ms": first_call_ms - avg_subsequent_ms,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print(" Steering Loop-Level Compilation Benchmark")
    print("="*60)

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, benchmarks may not be representative")

    print(f"PyTorch version: {torch.__version__}")

    # Run benchmarks
    bench_individual_ops()
    bench_fast_vs_slow_path()
    bench_steerllm_vs_chatspace()
    bench_compilation_overhead()

    print("\n" + "="*60)
    print(" Benchmark Complete")
    print("="*60)


if __name__ == "__main__":
    main()
