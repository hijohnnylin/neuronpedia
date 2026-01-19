#!/usr/bin/env python3
"""
End-to-end benchmark for all steering operation types.

Tests add, cap, ablation with:
1. Uniform batch (all same spec) - fast path
2. Heterogeneous batch (all unique specs) - optimal GPU-native path
3. Mixed operations (add + cap + ablation)

Usage:
    PYTHONPATH=/root/chatspace uv run python scripts/bench_all_ops_e2e.py
"""
import torch
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchConfig:
    batch_size: int = 32
    seq_len: int = 128
    hidden_size: int = 4096
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


@dataclass
class MockLayerSpec:
    """Mock layer spec matching runtime.py's expected structure."""
    operations: list[tuple[str, torch.Tensor, Any]] = field(default_factory=list)


def setup_scenario(cfg: BenchConfig, op_type: str, scenario: str):
    """Set up tensors and specs for a scenario."""

    # Determine seq_lens based on scenario
    if scenario.endswith("_nonuniform"):
        # Non-uniform: vary seq_lens from 64 to 192 (centered around 128)
        import random
        random.seed(42)
        seq_lens = [random.randint(64, 192) for _ in range(cfg.batch_size)]
        total_tokens = sum(seq_lens)
        base_scenario = scenario.replace("_nonuniform", "")
    else:
        seq_lens = [cfg.seq_len] * cfg.batch_size
        total_tokens = cfg.batch_size * cfg.seq_len
        base_scenario = scenario

    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)

    request_layer_specs = []

    if base_scenario == "uniform":
        # All requests use the same vector
        vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        vec = vec / vec.norm()

        if op_type == "add":
            ops = [("add", vec, None)]
        elif op_type == "cap":
            ops = [("cap", vec, (-1.0, 1.0))]
        elif op_type == "ablation":
            ops = [("ablation", vec, 0.5)]
        else:
            raise ValueError(f"Unknown op_type: {op_type}")

        layer_spec = MockLayerSpec(operations=ops)
        request_layer_specs = [layer_spec] * cfg.batch_size

    elif base_scenario == "heterogeneous":
        # All requests have different vectors
        for _ in range(cfg.batch_size):
            vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
            vec = vec / vec.norm()

            if op_type == "add":
                ops = [("add", vec, None)]
            elif op_type == "cap":
                ops = [("cap", vec, (-1.0, 1.0))]
            elif op_type == "ablation":
                ops = [("ablation", vec, 0.5)]
            else:
                raise ValueError(f"Unknown op_type: {op_type}")

            request_layer_specs.append(MockLayerSpec(operations=ops))
    else:
        raise ValueError(f"Unknown scenario: {base_scenario}")

    return hidden, seq_lens, request_layer_specs


def setup_mixed_scenario(cfg: BenchConfig):
    """Set up a scenario with mixed operations (add + cap + ablation)."""
    total_tokens = cfg.batch_size * cfg.seq_len
    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)
    seq_lens = [cfg.seq_len] * cfg.batch_size

    request_layer_specs = []
    for i in range(cfg.batch_size):
        vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        vec = vec / vec.norm()

        # Alternate between add, cap, ablation
        if i % 3 == 0:
            ops = [("add", vec, None)]
        elif i % 3 == 1:
            ops = [("cap", vec, (-1.0, 1.0))]
        else:
            ops = [("ablation", vec, 0.5)]

        request_layer_specs.append(MockLayerSpec(operations=ops))

    return hidden, seq_lens, request_layer_specs


# Import the actual steering functions
from steerllm.backends.vllm.runtime import (
    _slow_path_loop_impl,
    _fused_gather_steering,
    _fused_add_gpu_native as _fused_add_raw,
    _fused_cap_gpu_native as _fused_cap_raw,
    _fused_ablation_gpu_native as _fused_ablation_raw,
    _gather_scatter_steering,
)


# Wrapper functions that compute uniform_seq_lens for the benchmark
def _fused_add_gpu_native(hidden, seq_lens, specs):
    uniform = len(set(seq_lens)) == 1
    return _fused_add_raw(hidden, seq_lens, specs, uniform)


def _fused_cap_gpu_native(hidden, seq_lens, specs):
    uniform = len(set(seq_lens)) == 1
    return _fused_cap_raw(hidden, seq_lens, specs, uniform)


def _fused_ablation_gpu_native(hidden, seq_lens, specs):
    uniform = len(set(seq_lens)) == 1
    return _fused_ablation_raw(hidden, seq_lens, specs, uniform)


def bench_func(fn, hidden, seq_lens, specs, num_layers=4, num_iters=50):
    """Benchmark a steering function."""
    # Warmup
    for _ in range(5):
        for _ in range(num_layers):
            h = hidden.clone()
            fn(h, seq_lens, specs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for _ in range(num_layers):
            h = fn(h, seq_lens, specs)
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


def verify_correctness(hidden, seq_lens, specs, op_type):
    """Verify GPU-native produces same result as slow path."""
    h_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs)

    if op_type == "add":
        h_fused = _fused_add_gpu_native(hidden.clone(), seq_lens, specs)
    elif op_type == "cap":
        h_fused = _fused_cap_gpu_native(hidden.clone(), seq_lens, specs)
    elif op_type == "ablation":
        h_fused = _fused_ablation_gpu_native(hidden.clone(), seq_lens, specs)
    elif op_type == "mixed":
        h_fused = _fused_gather_steering(hidden.clone(), seq_lens, specs)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    # bf16 has ~3 decimal digits of precision, so allow 0.02 tolerance
    # Slow path uses @ (matmul), fused uses element-wise * + sum
    # These can differ slightly due to FMA and accumulation order
    if not torch.allclose(h_slow, h_fused, atol=0.02, rtol=0.01):
        diff = (h_slow - h_fused).abs()
        print(f"  WARNING: Results differ! Max diff: {diff.max().item():.6f}")
        return False
    return True


def run_single_op_benchmark():
    """Benchmark each operation type individually."""
    print("=" * 90)
    print("Single Operation Type Benchmark (4 layers)")
    print("=" * 90)

    cfg = BenchConfig(batch_size=32)
    num_layers = 4

    for op_type in ["add", "cap", "ablation"]:
        print(f"\n{'='*50}")
        print(f"Operation: {op_type.upper()}")
        print(f"{'='*50}")

        # Verify correctness
        for scenario in ["uniform", "heterogeneous"]:
            hidden, seq_lens, specs = setup_scenario(cfg, op_type, scenario)
            if not verify_correctness(hidden, seq_lens, specs, op_type):
                print(f"  {scenario}: FAILED")
                return
            print(f"  {scenario}: OK")

        print(f"\n{'Scenario':<20} {'SlowLoop':>12} {'Fused':>12} {'Speedup':>12}")
        print("-" * 60)

        for scenario in ["uniform", "heterogeneous"]:
            hidden, seq_lens, specs = setup_scenario(cfg, op_type, scenario)

            t_slow = bench_func(_slow_path_loop_impl, hidden, seq_lens, specs, num_layers)

            if op_type == "add":
                t_fused = bench_func(_fused_add_gpu_native, hidden, seq_lens, specs, num_layers)
            elif op_type == "cap":
                t_fused = bench_func(_fused_cap_gpu_native, hidden, seq_lens, specs, num_layers)
            elif op_type == "ablation":
                t_fused = bench_func(_fused_ablation_gpu_native, hidden, seq_lens, specs, num_layers)

            speedup = t_slow / t_fused
            print(f"{scenario:<20} {t_slow:>12.1f} {t_fused:>12.1f} {speedup:>11.2f}x")


def run_mixed_ops_benchmark():
    """Benchmark mixed operations."""
    print("\n" + "=" * 90)
    print("Mixed Operations Benchmark (add + cap + ablation)")
    print("=" * 90)

    cfg = BenchConfig(batch_size=32)
    num_layers = 4

    hidden, seq_lens, specs = setup_mixed_scenario(cfg)

    # Verify correctness
    if not verify_correctness(hidden, seq_lens, specs, "mixed"):
        print("Mixed ops: FAILED")
        return
    print("Mixed ops correctness: OK")

    t_slow = bench_func(_slow_path_loop_impl, hidden, seq_lens, specs, num_layers)
    t_fused = bench_func(_fused_gather_steering, hidden, seq_lens, specs, num_layers)

    print(f"\n{'Method':<30} {'Time (μs)':>12} {'Speedup':>12}")
    print("-" * 55)
    print(f"{'Slow loop':<30} {t_slow:>12.1f} {'1.00x':>12}")
    print(f"{'Fused GPU-native':<30} {t_fused:>12.1f} {t_slow/t_fused:>11.2f}x")


def run_scaling_analysis():
    """Analyze scaling with batch size for cap and ablation."""
    print("\n" + "=" * 90)
    print("Scaling Analysis: Cap and Ablation")
    print("=" * 90)

    batch_sizes = [8, 16, 32, 64, 128]
    num_layers = 4

    for op_type in ["cap", "ablation"]:
        print(f"\n--- {op_type.upper()} (heterogeneous, {num_layers} layers) ---")
        print(f"\n{'Batch':<10} {'SlowLoop':>12} {'Fused':>12} {'Speedup':>12}")
        print("-" * 50)

        for batch_size in batch_sizes:
            cfg = BenchConfig(batch_size=batch_size)
            hidden, seq_lens, specs = setup_scenario(cfg, op_type, "heterogeneous")

            t_slow = bench_func(_slow_path_loop_impl, hidden, seq_lens, specs, num_layers)

            if op_type == "cap":
                t_fused = bench_func(_fused_cap_gpu_native, hidden, seq_lens, specs, num_layers)
            else:
                t_fused = bench_func(_fused_ablation_gpu_native, hidden, seq_lens, specs, num_layers)

            speedup = t_slow / t_fused
            print(f"{batch_size:<10} {t_slow:>12.1f} {t_fused:>12.1f} {speedup:>11.2f}x")


def setup_shared_nonuniform(cfg: BenchConfig, op_type: str, num_unique_vecs: int = 4):
    """Set up shared vectors with non-uniform seq_lens."""
    import random
    random.seed(42)
    seq_lens = [random.randint(64, 192) for _ in range(cfg.batch_size)]
    total_tokens = sum(seq_lens)

    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)

    # Create a few shared vectors
    shared_vecs = []
    for _ in range(num_unique_vecs):
        vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        vec = vec / vec.norm()
        shared_vecs.append(vec)

    request_layer_specs = []
    for i in range(cfg.batch_size):
        vec = shared_vecs[i % num_unique_vecs]

        if op_type == "add":
            ops = [("add", vec, None)]
        elif op_type == "cap":
            ops = [("cap", vec, (-1.0, 1.0))]
        elif op_type == "ablation":
            ops = [("ablation", vec, 0.5)]
        else:
            raise ValueError(f"Unknown op_type: {op_type}")

        request_layer_specs.append(MockLayerSpec(operations=ops))

    return hidden, seq_lens, request_layer_specs


def run_nonuniform_benchmark():
    """Benchmark non-uniform seq_lens."""
    print("\n" + "=" * 90)
    print("Non-Uniform Seq_Lens Benchmark")
    print("=" * 90)
    print("seq_lens vary from 64-192 per request (avg ~128)")

    cfg = BenchConfig(batch_size=32)
    num_layers = 4

    # Test both heterogeneous (32 unique) and shared (4 unique) vectors
    for vector_scenario in ["heterogeneous", "shared"]:
        print(f"\n{'='*60}")
        print(f"Vectors: {vector_scenario.upper()} ({'32 unique' if vector_scenario == 'heterogeneous' else '4 shared'})")
        print(f"{'='*60}")

        for op_type in ["add", "cap", "ablation"]:
            print(f"\n--- {op_type.upper()} ---")

            # Set up scenario
            if vector_scenario == "heterogeneous":
                hidden, seq_lens, specs = setup_scenario(cfg, op_type, "heterogeneous_nonuniform")
            else:
                hidden, seq_lens, specs = setup_shared_nonuniform(cfg, op_type, num_unique_vecs=4)

            # Verify correctness
            if not verify_correctness(hidden, seq_lens, specs, op_type):
                print(f"  {op_type} non-uniform: FAILED")
                continue
            print(f"  Correctness: OK (seq_lens range: {min(seq_lens)}-{max(seq_lens)})")

            t_slow = bench_func(_slow_path_loop_impl, hidden, seq_lens, specs, num_layers)
            t_gather = bench_func(_gather_scatter_steering, hidden, seq_lens, specs, num_layers)

            if op_type == "add":
                t_fused = bench_func(_fused_add_gpu_native, hidden, seq_lens, specs, num_layers)
            elif op_type == "cap":
                t_fused = bench_func(_fused_cap_gpu_native, hidden, seq_lens, specs, num_layers)
            else:
                t_fused = bench_func(_fused_ablation_gpu_native, hidden, seq_lens, specs, num_layers)

            best = min(t_slow, t_gather, t_fused)
            best_name = "Slow" if best == t_slow else ("Gather" if best == t_gather else "Fused")

            print(f"\n  {'Method':<25} {'Time (μs)':>12} {'vs Slow':>12}")
            print(f"  {'-'*50}")
            print(f"  {'Slow loop':<25} {t_slow:>12.1f} {'1.00x':>12}")
            print(f"  {'Gather/scatter':<25} {t_gather:>12.1f} {t_slow/t_gather:>11.2f}x")
            print(f"  {'Fused GPU-native':<25} {t_fused:>12.1f} {t_slow/t_fused:>11.2f}x")
            print(f"\n  Best: {best_name} ({best:.1f}μs)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    run_single_op_benchmark()
    run_mixed_ops_benchmark()
    run_scaling_analysis()
    run_nonuniform_benchmark()
