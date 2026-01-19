#!/usr/bin/env python3
"""
End-to-end benchmark for steering paths across multiple layers.

Tests:
1. Uniform batch fast path
2. Fused gather (new)
3. Slow loop (fallback)
4. Old gather/scatter (for comparison)

Across 1, 2, 4, 8 layers.

Usage:
    uv run python scripts/bench_steering_e2e.py
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
    num_unique_vecs: int = 4
    num_layers: int = 4
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


@dataclass
class MockLayerSpec:
    """Mock layer spec matching runtime.py's expected structure."""
    operations: list[tuple[str, torch.Tensor, Any]] = field(default_factory=list)


def setup_scenario(cfg: BenchConfig, scenario: str):
    """Set up tensors and specs for a scenario."""
    total_tokens = cfg.batch_size * cfg.seq_len
    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)
    seq_lens = [cfg.seq_len] * cfg.batch_size

    # Create steering vectors
    if scenario == "uniform":
        # All requests use the same vector (fast path)
        vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        vec = vec / vec.norm()  # Normalize
        layer_spec = MockLayerSpec(operations=[("add", vec, None)])
        request_layer_specs = [layer_spec] * cfg.batch_size
    elif scenario == "shared":
        # 4 unique vectors shared across requests
        vecs = [torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
                for _ in range(4)]
        vecs = [v / v.norm() for v in vecs]
        request_layer_specs = []
        for i in range(cfg.batch_size):
            vec = vecs[i % 4]
            request_layer_specs.append(MockLayerSpec(operations=[("add", vec, None)]))
    elif scenario == "heterogeneous":
        # All different vectors
        request_layer_specs = []
        for _ in range(cfg.batch_size):
            vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
            vec = vec / vec.norm()
            request_layer_specs.append(MockLayerSpec(operations=[("add", vec, None)]))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return hidden, seq_lens, request_layer_specs


# Import the actual steering functions
from steerllm.backends.vllm.runtime import (
    _slow_path_loop_impl,
    _gather_scatter_steering,
    _fused_gather_steering,
    _apply_layer_steering_to_hidden,
)


def bench_uniform_fast_path(hidden, seq_lens, request_layer_specs, num_layers, num_iters=50):
    """Benchmark uniform batch fast path (single batched op)."""
    layer_spec = request_layer_specs[0]  # All same
    total_tokens = sum(seq_lens)

    # Warmup
    for _ in range(5):
        for _ in range(num_layers):
            h = hidden.clone()
            h[:total_tokens] = _apply_layer_steering_to_hidden(hidden[:total_tokens], layer_spec, None)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for _ in range(num_layers):
            h[:total_tokens] = _apply_layer_steering_to_hidden(h[:total_tokens], layer_spec, None)
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


def bench_fused(hidden, seq_lens, request_layer_specs, num_layers, num_iters=50):
    """Benchmark fused gather path."""
    # Warmup
    for _ in range(5):
        for _ in range(num_layers):
            h = hidden.clone()
            h = _fused_gather_steering(h, seq_lens, request_layer_specs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for _ in range(num_layers):
            h = _fused_gather_steering(h, seq_lens, request_layer_specs)
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


def bench_gather_scatter(hidden, seq_lens, request_layer_specs, num_layers, num_iters=50):
    """Benchmark old gather/scatter path."""
    # Warmup
    for _ in range(5):
        for _ in range(num_layers):
            h = hidden.clone()
            h = _gather_scatter_steering(h, seq_lens, request_layer_specs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for _ in range(num_layers):
            h = _gather_scatter_steering(h, seq_lens, request_layer_specs)
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


def bench_slow_loop(hidden, seq_lens, request_layer_specs, num_layers, num_iters=50):
    """Benchmark slow loop path."""
    # Warmup
    for _ in range(5):
        for _ in range(num_layers):
            h = hidden.clone()
            h = _slow_path_loop_impl(h, seq_lens, request_layer_specs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for _ in range(num_layers):
            h = _slow_path_loop_impl(h, seq_lens, request_layer_specs)
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


def verify_correctness(hidden, seq_lens, request_layer_specs):
    """Verify all paths produce the same result."""
    h_fused = _fused_gather_steering(hidden.clone(), seq_lens, request_layer_specs)
    h_gather = _gather_scatter_steering(hidden.clone(), seq_lens, request_layer_specs)
    h_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, request_layer_specs)

    fused_vs_gather = torch.allclose(h_fused, h_gather, atol=1e-5, rtol=1e-5)
    fused_vs_slow = torch.allclose(h_fused, h_slow, atol=1e-5, rtol=1e-5)

    if not fused_vs_gather or not fused_vs_slow:
        print(f"WARNING: Results differ!")
        print(f"  Fused vs Gather: {fused_vs_gather}")
        print(f"  Fused vs Slow: {fused_vs_slow}")
        if not fused_vs_gather:
            diff = (h_fused - h_gather).abs()
            print(f"  Max diff (fused vs gather): {diff.max().item():.6f}")
        if not fused_vs_slow:
            diff = (h_fused - h_slow).abs()
            print(f"  Max diff (fused vs slow): {diff.max().item():.6f}")
        return False
    return True


def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 100)
    print("End-to-End Steering Benchmark (Multi-Layer)")
    print("=" * 100)

    # Verify correctness first
    print("\nVerifying correctness...")
    cfg = BenchConfig(batch_size=32, num_unique_vecs=8)
    for scenario in ["uniform", "shared", "heterogeneous"]:
        hidden, seq_lens, specs = setup_scenario(cfg, scenario)
        if not verify_correctness(hidden, seq_lens, specs):
            print(f"  {scenario}: FAILED")
            return
        print(f"  {scenario}: OK")

    print("\n" + "=" * 100)

    scenarios = ["uniform", "shared", "heterogeneous"]
    num_layers_list = [1, 2, 4, 8]

    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario.upper()}")
        print(f"{'='*50}")

        if scenario == "uniform":
            cfg = BenchConfig(batch_size=32, num_unique_vecs=1)
        elif scenario == "shared":
            cfg = BenchConfig(batch_size=32, num_unique_vecs=4)
        else:
            cfg = BenchConfig(batch_size=32, num_unique_vecs=32)

        hidden, seq_lens, specs = setup_scenario(cfg, scenario)

        print(f"\n{'Layers':<10} {'FastPath':>12} {'Fused':>12} {'Gather':>12} {'SlowLoop':>12} {'Best':>10} {'Fused vs Best':>15}")
        print("-" * 85)

        for num_layers in num_layers_list:
            if scenario == "uniform":
                t_fast = bench_uniform_fast_path(hidden, seq_lens, specs, num_layers)
            else:
                t_fast = float('inf')  # Not applicable

            t_fused = bench_fused(hidden, seq_lens, specs, num_layers)
            t_gather = bench_gather_scatter(hidden, seq_lens, specs, num_layers)
            t_slow = bench_slow_loop(hidden, seq_lens, specs, num_layers)

            times = {'Fast': t_fast, 'Fused': t_fused, 'Gather': t_gather, 'Slow': t_slow}
            best_name = min(times, key=times.get)
            best_time = times[best_name]

            fused_vs_best = t_fused / best_time if best_time > 0 else 0

            fast_str = f"{t_fast:.1f}" if t_fast < float('inf') else "N/A"
            print(f"{num_layers:<10} {fast_str:>12} {t_fused:>12.1f} {t_gather:>12.1f} {t_slow:>12.1f} {best_name:>10} {fused_vs_best:>14.2f}x")

    print("\n" + "=" * 100)
    print("Summary:")
    print("  - Fast path: Only for uniform batches (all requests use same spec)")
    print("  - Fused: New approach - single gather/scatter for all ops")
    print("  - Gather: Old approach - one gather/scatter per unique op")
    print("  - Slow: Per-request slice operations")
    print("\nFused vs Best = 1.0x means fused IS the best, >1.0x means fused is slower than best")


def run_scaling_analysis():
    """Analyze how performance scales with batch size."""
    print("\n" + "=" * 100)
    print("Scaling Analysis: Fused vs Others")
    print("=" * 100)

    batch_sizes = [8, 16, 32, 64, 128]
    num_layers = 4

    print(f"\nHeterogeneous scenario (all unique vectors), {num_layers} layers")
    print(f"\n{'Batch':<10} {'Fused':>12} {'Gather':>12} {'SlowLoop':>12} {'Fused Speedup':>15}")
    print("-" * 65)

    for batch_size in batch_sizes:
        cfg = BenchConfig(batch_size=batch_size, num_unique_vecs=batch_size)
        hidden, seq_lens, specs = setup_scenario(cfg, "heterogeneous")

        t_fused = bench_fused(hidden, seq_lens, specs, num_layers)
        t_gather = bench_gather_scatter(hidden, seq_lens, specs, num_layers)
        t_slow = bench_slow_loop(hidden, seq_lens, specs, num_layers)

        best_other = min(t_gather, t_slow)
        speedup = best_other / t_fused if t_fused > 0 else 0

        print(f"{batch_size:<10} {t_fused:>12.1f} {t_gather:>12.1f} {t_slow:>12.1f} {speedup:>14.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    run_benchmark()
    run_scaling_analysis()
