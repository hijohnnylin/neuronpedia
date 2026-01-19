#!/usr/bin/env python3
"""
Benchmark for clock scheduler multi-op steering.

Tests:
1. Single-op (baseline - should use fast path)
2. Multi-op uniform (all requests have same ops)
3. Multi-op heterogeneous (requests have different ops)
4. Add merging benefit

Usage:
    uv run python scripts/bench_clock_scheduler.py
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


# Import the actual steering functions
from steerllm.backends.vllm.runtime import (
    _slow_path_loop_impl,
    _fused_gather_steering,
    _build_clock_schedule,
    _execute_clock_schedule,
)


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


def setup_single_op(cfg: BenchConfig, op_type: str = "add"):
    """Setup: each request has exactly 1 op."""
    total_tokens = cfg.batch_size * cfg.seq_len
    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)
    seq_lens = [cfg.seq_len] * cfg.batch_size

    specs = []
    for _ in range(cfg.batch_size):
        vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        vec = vec / vec.norm()
        if op_type == "add":
            ops = [("add", vec, None)]
        elif op_type == "cap":
            ops = [("cap", vec, (-0.5, 0.5))]
        elif op_type == "ablation":
            ops = [("ablation", vec, 0.5)]
        specs.append(MockLayerSpec(operations=ops))

    return hidden, seq_lens, specs


def setup_multi_op_uniform(cfg: BenchConfig):
    """Setup: all requests have the same [add, cap] sequence."""
    total_tokens = cfg.batch_size * cfg.seq_len
    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)
    seq_lens = [cfg.seq_len] * cfg.batch_size

    add_vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
    cap_vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
    cap_vec = cap_vec / cap_vec.norm()

    specs = []
    for _ in range(cfg.batch_size):
        ops = [("add", add_vec, None), ("cap", cap_vec, (-0.5, 0.5))]
        specs.append(MockLayerSpec(operations=ops))

    return hidden, seq_lens, specs


def setup_multi_op_heterogeneous(cfg: BenchConfig):
    """Setup: requests have different op sequences."""
    total_tokens = cfg.batch_size * cfg.seq_len
    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)
    seq_lens = [cfg.seq_len] * cfg.batch_size

    specs = []
    for i in range(cfg.batch_size):
        vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        vec = vec / vec.norm()

        # Vary the ops per request
        if i % 4 == 0:
            ops = [("add", vec, None), ("cap", vec, (-0.5, 0.5))]
        elif i % 4 == 1:
            ops = [("cap", vec, (-0.5, 0.5)), ("add", vec, None)]
        elif i % 4 == 2:
            ops = [("add", vec, None), ("ablation", vec, 0.5)]
        else:
            ops = [("ablation", vec, 0.5), ("cap", vec, (-0.5, 0.5)), ("add", vec, None)]

        specs.append(MockLayerSpec(operations=ops))

    return hidden, seq_lens, specs


def setup_consecutive_adds(cfg: BenchConfig, num_adds: int = 3):
    """Setup: each request has multiple consecutive adds (should be merged)."""
    total_tokens = cfg.batch_size * cfg.seq_len
    hidden = torch.randn(total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)
    seq_lens = [cfg.seq_len] * cfg.batch_size

    specs = []
    for _ in range(cfg.batch_size):
        ops = []
        for _ in range(num_adds):
            vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
            ops.append(("add", vec, None))
        specs.append(MockLayerSpec(operations=ops))

    return hidden, seq_lens, specs


def verify_correctness(hidden, seq_lens, specs):
    """Verify fused matches slow path.

    Note: bf16 has ~3 decimal digits of precision. Multi-op sequences
    accumulate numerical error differently between slow path (sequential)
    and fused path (merged adds). Use relaxed tolerance.
    """
    h_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs)
    h_fused = _fused_gather_steering(hidden.clone(), seq_lens, specs)

    # bf16 tolerance: 0.15 allows for 5+ ops with accumulated error
    if not torch.allclose(h_slow, h_fused, atol=0.15, rtol=0.05):
        diff = (h_slow - h_fused).abs()
        print(f"  WARNING: Max diff: {diff.max().item():.6f}")
        return False
    return True


def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 90)
    print("Clock Scheduler Benchmark (Multi-Op Steering)")
    print("=" * 90)

    cfg = BenchConfig(batch_size=32)
    num_layers = 4

    scenarios = [
        ("Single-op (add)", lambda: setup_single_op(cfg, "add")),
        ("Single-op (cap)", lambda: setup_single_op(cfg, "cap")),
        ("Multi-op uniform [add,cap]", lambda: setup_multi_op_uniform(cfg)),
        ("Multi-op heterogeneous", lambda: setup_multi_op_heterogeneous(cfg)),
        ("Consecutive adds (x3)", lambda: setup_consecutive_adds(cfg, 3)),
        ("Consecutive adds (x5)", lambda: setup_consecutive_adds(cfg, 5)),
    ]

    print(f"\nBatch={cfg.batch_size}, SeqLen={cfg.seq_len}, Layers={num_layers}")
    print(f"\n{'Scenario':<35} {'SlowPath':>12} {'Fused':>12} {'Speedup':>12}")
    print("-" * 75)

    for name, setup_fn in scenarios:
        hidden, seq_lens, specs = setup_fn()

        # Verify correctness
        if not verify_correctness(hidden, seq_lens, specs):
            print(f"{name:<35} FAILED (correctness)")
            continue

        t_slow = bench_func(_slow_path_loop_impl, hidden, seq_lens, specs, num_layers)
        t_fused = bench_func(_fused_gather_steering, hidden, seq_lens, specs, num_layers)
        speedup = t_slow / t_fused

        print(f"{name:<35} {t_slow:>12.1f} {t_fused:>12.1f} {speedup:>11.2f}x")


def run_scaling_analysis():
    """Analyze how performance scales with number of ops per request."""
    print("\n" + "=" * 90)
    print("Scaling Analysis: Ops per Request")
    print("=" * 90)

    cfg = BenchConfig(batch_size=32)
    num_layers = 4

    print(f"\n{'Ops/Request':<15} {'SlowPath':>12} {'Fused':>12} {'Speedup':>12}")
    print("-" * 55)

    for num_ops in [1, 2, 3, 4, 5]:
        # Create specs with varying op counts
        total_tokens = cfg.batch_size * cfg.seq_len
        hidden = torch.randn(total_tokens, cfg.hidden_size,
                             device=cfg.device, dtype=cfg.dtype)
        seq_lens = [cfg.seq_len] * cfg.batch_size

        specs = []
        for i in range(cfg.batch_size):
            ops = []
            for j in range(num_ops):
                vec = torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
                vec = vec / vec.norm()
                # Alternate op types
                if j % 3 == 0:
                    ops.append(("add", vec, None))
                elif j % 3 == 1:
                    ops.append(("cap", vec, (-0.5, 0.5)))
                else:
                    ops.append(("ablation", vec, 0.5))
            specs.append(MockLayerSpec(operations=ops))

        if not verify_correctness(hidden, seq_lens, specs):
            print(f"{num_ops:<15} FAILED")
            continue

        t_slow = bench_func(_slow_path_loop_impl, hidden, seq_lens, specs, num_layers)
        t_fused = bench_func(_fused_gather_steering, hidden, seq_lens, specs, num_layers)
        speedup = t_slow / t_fused

        print(f"{num_ops:<15} {t_slow:>12.1f} {t_fused:>12.1f} {speedup:>11.2f}x")


def run_batch_size_scaling():
    """Analyze how performance scales with batch size."""
    print("\n" + "=" * 90)
    print("Scaling Analysis: Batch Size (multi-op)")
    print("=" * 90)

    batch_sizes = [8, 16, 32, 64, 128]
    num_layers = 4

    print(f"\n{'Batch':<10} {'SlowPath':>12} {'Fused':>12} {'Speedup':>12}")
    print("-" * 50)

    for batch_size in batch_sizes:
        cfg = BenchConfig(batch_size=batch_size)
        hidden, seq_lens, specs = setup_multi_op_heterogeneous(cfg)

        if not verify_correctness(hidden, seq_lens, specs):
            print(f"{batch_size:<10} FAILED")
            continue

        t_slow = bench_func(_slow_path_loop_impl, hidden, seq_lens, specs, num_layers)
        t_fused = bench_func(_fused_gather_steering, hidden, seq_lens, specs, num_layers)
        speedup = t_slow / t_fused

        print(f"{batch_size:<10} {t_slow:>12.1f} {t_fused:>12.1f} {speedup:>11.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    run_benchmark()
    run_scaling_analysis()
    run_batch_size_scaling()
