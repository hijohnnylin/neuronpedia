# steerllm torch.compile Benchmark

**Date**: 2025-11-30
**Hardware**: NVIDIA H200
**PyTorch**: 2.8.0+cu128
**CUDA**: 12.8

## Objective

Evaluate torch.compile with dynamic shapes for steerllm steering operations on the slow path (heterogeneous batches).

## Key Insight: Compile at Loop Level, Not Op Level

Individual op compilation adds dispatch overhead (~30-50μs per call) that dominates small tensor ops.
**Loop-level compilation** amortizes dispatch overhead across all iterations, giving 21% speedup on heterogeneous batches.

## Findings

### 1. Individual op compilation adds overhead

Using `mode="reduce-overhead"` (initial attempt):
| Op | Compiled (μs) | Uncompiled (μs) | Ratio |
|---|---|---|---|
| Add | 82 | 15 | 5.7x slower |
| Projection cap | 92 | 60 | 1.5x slower |
| Ablation | 86 | 40 | 2.1x slower |

**Problem**: Per-call dispatch overhead dominates for small ops.

### 2. Loop-level compilation gives 21% speedup

| Path | Mean (μs) | vs Fast Path |
|---|---|---|
| Fast path (uniform, 32 reqs) | 96 | 1x |
| Slow path compiled | 1248 | 13x slower |
| Slow path uncompiled | 1515 | 16x slower |

**Compiled vs uncompiled slow path: 1.21x faster (21% speedup)**

### 3. Fast path remains the dominant optimization

| Path | Time | Speedup |
|---|---|---|
| Fast path (uniform batch) | 96 μs | 15.7x vs slow uncompiled |
| Slow path compiled | 1248 μs | 1.21x vs slow uncompiled |
| Slow path uncompiled | 1515 μs | baseline |

### 4. Memory access patterns are the remaining bottleneck

Even with perfect kernel fusion, 32 separate slice operations are ~10x slower than one contiguous operation:
- One big op (2048 tokens): 71 μs
- 32 slices (compiled): 738 μs
- 32 slices (uncompiled): 1532 μs

The remaining 10x gap is due to memory access patterns, not kernel launch overhead.

### 5. steerllm vs chatspace at parity

| Op | steerllm | chatspace |
|---|---|---|
| Projection cap | 57 μs | 57 μs |
| Ablation | 40 μs | 40 μs |

### 6. Compilation warmup overhead is significant

| Metric | Value |
|---|---|
| First call (incl. compile) | 3043 ms |
| Subsequent calls (avg) | 1.24 ms |
| Amortization point | ~2500 calls |

## Decision

- **Disabled torch.compile by default** (`STEERLLM_COMPILE_STEERING=0`)
- Loop-level compilation gives 21% speedup on heterogeneous batches
- But compilation overhead (3s first call) is significant
- **Fast path is the main optimization** (15.7x speedup)
- Left infrastructure in place for users to opt-in via env var

## Implementation

Changed from individual op compilation to loop-level compilation:

```python
# OLD: Individual ops compiled (doesn't help)
compiled_cap = torch.compile(_projection_cap_impl, dynamic=True)
for req in requests:
    hidden[slice] = compiled_cap(hidden[slice], vec, min, max)  # dispatch overhead each call

# NEW: Entire loop compiled (21% faster)
@torch.compile(dynamic=True)
def _slow_path_loop_impl(hidden, seq_lens, request_layer_specs):
    for i, layer_spec in enumerate(request_layer_specs):
        # All ops fused within single compiled graph
        ...
```

## Configuration

To enable loop-level compilation for heterogeneous batches:
```bash
export STEERLLM_COMPILE_STEERING=1
```

## Test Commands

```bash
uv run python scripts/benchmark_steering_compile.py
```

---

## Update: Gather/Scatter + Vector Interning (6x speedup)

### Problem Identified

The fast path and loop-level compilation only worked for **in-process batching** because they relied on Python object identity (`id(vec)`). Across RPC, each request deserializes its own tensor, breaking identity checks.

### Solution: Two-Part Optimization

#### 1. Gather/Scatter Steering

Instead of N non-contiguous slice operations, group tokens by operation signature and apply in batches:

| Unique Ops | Slow Loop | Gather/Scatter | Speedup |
|------------|-----------|----------------|---------|
| 1 | 1566 μs | 255 μs | **6.1x** |
| 2 | 1564 μs | 298 μs | 5.2x |
| 4 | 1566 μs | 432 μs | 3.6x |
| 8 | 1569 μs | 714 μs | 2.2x |
| 16 | 1583 μs | 1312 μs | 1.2x |
| 32 | 1583 μs | 2391 μs | 0.66x (regression) |

**Adaptive heuristic**: Use gather/scatter when `num_unique_ops < num_requests`, otherwise fall back to loop.

#### 2. Content-Based Vector Interning

Hash vector bytes (SHA256, ~6μs) and cache tensors. Identical vectors across RPC now share the same `id()`:

```
Request 1: vec_bytes -> deserialize -> hash -> cache miss -> store
Request 2: vec_bytes -> deserialize -> hash -> cache HIT -> reuse tensor

Both requests now have same id(vec) -> gather/scatter batches them
```

### End-to-End Results (Simulated RPC)

| Scenario | Time | vs Baseline |
|----------|------|-------------|
| Baseline (slow loop) | 1530 μs | 1x |
| Gather/scatter + interning | 249 μs | **6.1x faster** |

### Path Selection Summary

| Condition | Path | Time |
|-----------|------|------|
| All same spec object | Fast path | ~100 μs |
| Shared vectors (via interning) | Gather/scatter | ~250 μs |
| All unique vectors | Loop | ~1550 μs |

### Implementation

- `_gather_scatter_steering()`: Groups by operation, gather→apply→scatter
- `_intern_vector()`: SHA256 hash + LRU cache (1000 vectors, ~8MB)
- Adaptive heuristic in `_apply_per_request_steering()`
