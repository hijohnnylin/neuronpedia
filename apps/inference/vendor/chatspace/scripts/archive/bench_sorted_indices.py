#!/usr/bin/env python3
"""
Microbenchmark for pre-sorted index tensor approaches.

Hypothesis: Sorting indices for better memory locality could improve gather/scatter.

Approaches:
1. Current gather/scatter (indices in request order)
2. Sorted indices (ascending order for coalesced memory access)
3. Blocked indices (group contiguous ranges)
4. Single fused gather (all ops at once)

Usage:
    uv run python scripts/bench_sorted_indices.py
"""
import torch
import time
from dataclasses import dataclass


@dataclass
class BenchConfig:
    batch_size: int = 32
    seq_len: int = 128
    hidden_size: int = 4096
    num_unique_vecs: int = 4
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    @property
    def total_tokens(self):
        return self.batch_size * self.seq_len


def setup_tensors(cfg: BenchConfig):
    """Create hidden states, vectors, and index structures."""
    hidden = torch.randn(cfg.total_tokens, cfg.hidden_size,
                         device=cfg.device, dtype=cfg.dtype)

    vectors = [
        torch.randn(cfg.hidden_size, device=cfg.device, dtype=cfg.dtype)
        for _ in range(cfg.num_unique_vecs)
    ]

    # Assign vectors to requests (round-robin)
    request_vecs = [vectors[i % cfg.num_unique_vecs] for i in range(cfg.batch_size)]
    ranges = [(i * cfg.seq_len, (i + 1) * cfg.seq_len) for i in range(cfg.batch_size)]

    return hidden, vectors, request_vecs, ranges


def build_index_structures(cfg: BenchConfig, vectors, request_vecs, ranges):
    """Pre-build all index structures for fair comparison."""

    # Group by vector identity
    vec_to_data = {}
    for vec, (start, end) in zip(request_vecs, ranges):
        key = id(vec)
        if key not in vec_to_data:
            vec_to_data[key] = {'vec': vec, 'ranges': [], 'indices': []}
        vec_to_data[key]['ranges'].append((start, end))
        vec_to_data[key]['indices'].extend(range(start, end))

    structures = {}

    # 1. Unsorted indices (current approach)
    structures['unsorted'] = []
    for data in vec_to_data.values():
        idx = torch.tensor(data['indices'], device=cfg.device, dtype=torch.long)
        structures['unsorted'].append((data['vec'], idx))

    # 2. Sorted indices (ascending for coalesced access)
    structures['sorted'] = []
    for data in vec_to_data.values():
        idx = torch.tensor(sorted(data['indices']), device=cfg.device, dtype=torch.long)
        structures['sorted'].append((data['vec'], idx))

    # 3. Blocked ranges (keep track of contiguous blocks)
    structures['blocked'] = []
    for data in vec_to_data.values():
        # Each range is already contiguous, store as (start, length) pairs
        blocks = [(s, e - s) for s, e in data['ranges']]
        structures['blocked'].append((data['vec'], blocks))

    # 4. Single fused structure (all vectors stacked, all indices together)
    all_vecs = []
    all_indices = []
    vec_assignments = []  # Which vector each index uses

    for vec_idx, data in enumerate(vec_to_data.values()):
        all_vecs.append(data['vec'])
        n_indices = len(data['indices'])
        all_indices.extend(data['indices'])
        vec_assignments.extend([vec_idx] * n_indices)

    structures['fused'] = {
        'vecs': torch.stack(all_vecs),  # [num_unique, hidden]
        'indices': torch.tensor(all_indices, device=cfg.device, dtype=torch.long),
        'assignments': torch.tensor(vec_assignments, device=cfg.device, dtype=torch.long),
    }

    # 5. Sorted fused (sort indices, reorder assignments accordingly)
    sorted_order = torch.argsort(structures['fused']['indices'])
    structures['fused_sorted'] = {
        'vecs': structures['fused']['vecs'],
        'indices': structures['fused']['indices'][sorted_order],
        'assignments': structures['fused']['assignments'][sorted_order],
    }

    return structures


# ============ APPROACH 1: Current gather/scatter (unsorted) ============
def bench_unsorted_gather_scatter(hidden, structures, num_iters=100):
    """Current implementation: gather by vector group, unsorted indices."""
    data = structures['unsorted']

    # Warmup
    for _ in range(10):
        h = hidden.clone()
        for vec, idx in data:
            gathered = h[idx]
            gathered = gathered + vec
            h[idx] = gathered
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for vec, idx in data:
            gathered = h[idx]
            gathered = gathered + vec
            h[idx] = gathered
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


# ============ APPROACH 2: Sorted indices ============
def bench_sorted_gather_scatter(hidden, structures, num_iters=100):
    """Sorted indices for potentially better memory coalescing."""
    data = structures['sorted']

    # Warmup
    for _ in range(10):
        h = hidden.clone()
        for vec, idx in data:
            gathered = h[idx]
            gathered = gathered + vec
            h[idx] = gathered
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for vec, idx in data:
            gathered = h[idx]
            gathered = gathered + vec
            h[idx] = gathered
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


# ============ APPROACH 3: Blocked slice operations ============
def bench_blocked_slices(hidden, structures, num_iters=100):
    """Use slice operations on contiguous blocks instead of index tensors."""
    data = structures['blocked']

    # Warmup
    for _ in range(10):
        h = hidden.clone()
        for vec, blocks in data:
            for start, length in blocks:
                h[start:start+length] += vec
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for vec, blocks in data:
            for start, length in blocks:
                h[start:start+length] += vec
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


# ============ APPROACH 4: Single fused gather (advanced indexing) ============
def bench_fused_gather(hidden, structures, num_iters=100):
    """Single gather, apply all vectors at once using advanced indexing."""
    data = structures['fused']
    vecs = data['vecs']           # [num_unique, hidden]
    indices = data['indices']      # [total_selected]
    assignments = data['assignments']  # [total_selected] -> which vec

    # Warmup
    for _ in range(10):
        h = hidden.clone()
        gathered = h[indices]                    # [total_selected, hidden]
        vec_for_each = vecs[assignments]         # [total_selected, hidden]
        gathered = gathered + vec_for_each
        h[indices] = gathered
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        gathered = h[indices]
        vec_for_each = vecs[assignments]
        gathered = gathered + vec_for_each
        h[indices] = gathered
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


# ============ APPROACH 5: Fused gather with sorted indices ============
def bench_fused_sorted(hidden, structures, num_iters=100):
    """Single gather with sorted indices for better memory access."""
    data = structures['fused_sorted']
    vecs = data['vecs']
    indices = data['indices']
    assignments = data['assignments']

    # Warmup
    for _ in range(10):
        h = hidden.clone()
        gathered = h[indices]
        vec_for_each = vecs[assignments]
        gathered = gathered + vec_for_each
        h[indices] = gathered
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        gathered = h[indices]
        vec_for_each = vecs[assignments]
        gathered = gathered + vec_for_each
        h[indices] = gathered
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


# ============ APPROACH 6: index_add_ (atomic, avoids scatter) ============
def bench_index_add(hidden, structures, num_iters=100):
    """Use index_add_ which may have better memory semantics."""
    data = structures['fused_sorted']
    vecs = data['vecs']
    indices = data['indices']
    assignments = data['assignments']

    # Pre-compute the values to add
    vec_for_each = vecs[assignments]  # [total_selected, hidden]

    # Warmup
    for _ in range(10):
        h = hidden.clone()
        h.index_add_(0, indices, vec_for_each)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        h.index_add_(0, indices, vec_for_each)
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


# ============ APPROACH 7: Slow path baseline ============
def bench_slow_path(hidden, request_vecs, ranges, num_iters=100):
    """Baseline: per-request slice operations."""
    # Warmup
    for _ in range(10):
        h = hidden.clone()
        for vec, (start, end) in zip(request_vecs, ranges):
            h[start:end] += vec
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hidden.clone()
        for vec, (start, end) in zip(request_vecs, ranges):
            h[start:end] += vec
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters * 1e6


def run_benchmark():
    """Run full benchmark suite."""
    print("=" * 100)
    print("Pre-Sorted Index Tensor Benchmark")
    print("=" * 100)

    configs = [
        # (batch, seq, hidden, unique_vecs)
        BenchConfig(32, 128, 4096, 1),   # All same vec
        BenchConfig(32, 128, 4096, 4),   # 4 unique
        BenchConfig(32, 128, 4096, 8),   # 8 unique
        BenchConfig(32, 128, 4096, 32),  # All different
        BenchConfig(32, 512, 4096, 4),   # Longer seqs
        BenchConfig(64, 128, 4096, 4),   # More requests
        BenchConfig(128, 128, 4096, 4),  # Even more
    ]

    print(f"\n{'Config':<25} {'Slow':>8} {'Unsort':>8} {'Sorted':>8} {'Block':>8} {'Fused':>8} {'FuseS':>8} {'IdxAdd':>8} {'Best':>8}")
    print("-" * 100)

    for cfg in configs:
        hidden, vectors, request_vecs, ranges = setup_tensors(cfg)
        structures = build_index_structures(cfg, vectors, request_vecs, ranges)

        t_slow = bench_slow_path(hidden, request_vecs, ranges)
        t_unsorted = bench_unsorted_gather_scatter(hidden, structures)
        t_sorted = bench_sorted_gather_scatter(hidden, structures)
        t_blocked = bench_blocked_slices(hidden, structures)
        t_fused = bench_fused_gather(hidden, structures)
        t_fused_sorted = bench_fused_sorted(hidden, structures)
        t_index_add = bench_index_add(hidden, structures)

        times = {
            'Slow': t_slow,
            'Unsort': t_unsorted,
            'Sorted': t_sorted,
            'Block': t_blocked,
            'Fused': t_fused,
            'FuseS': t_fused_sorted,
            'IdxAdd': t_index_add,
        }
        best = min(times, key=times.get)

        desc = f"B={cfg.batch_size},S={cfg.seq_len},V={cfg.num_unique_vecs}"
        print(f"{desc:<25} {t_slow:>7.1f} {t_unsorted:>7.1f} {t_sorted:>7.1f} "
              f"{t_blocked:>7.1f} {t_fused:>7.1f} {t_fused_sorted:>7.1f} {t_index_add:>7.1f} {best:>8}")

    print("-" * 100)
    print("\nLegend:")
    print("  Slow   = Per-request slice ops (baseline)")
    print("  Unsort = Gather/scatter with unsorted indices (current)")
    print("  Sorted = Gather/scatter with sorted indices")
    print("  Block  = Slice ops on contiguous blocks")
    print("  Fused  = Single gather, vec lookup by assignment")
    print("  FuseS  = Fused with sorted indices")
    print("  IdxAdd = index_add_ (atomic accumulate)")


def run_memory_analysis():
    """Analyze memory access patterns."""
    print("\n" + "=" * 80)
    print("Memory Access Pattern Analysis")
    print("=" * 80)

    cfg = BenchConfig(32, 128, 4096, 4)
    hidden, vectors, request_vecs, ranges = setup_tensors(cfg)
    structures = build_index_structures(cfg, vectors, request_vecs, ranges)

    print(f"\nConfig: {cfg.batch_size} requests, {cfg.seq_len} tokens/req, {cfg.num_unique_vecs} unique vecs")
    print(f"Total tokens: {cfg.total_tokens}")

    # Analyze index patterns
    for name, data in [('unsorted', structures['unsorted']), ('sorted', structures['sorted'])]:
        print(f"\n{name.upper()} indices:")
        for i, (vec, idx) in enumerate(data):
            idx_cpu = idx.cpu().numpy()
            # Check for contiguous runs
            diffs = idx_cpu[1:] - idx_cpu[:-1]
            contiguous = (diffs == 1).sum()
            jumps = (diffs != 1).sum()
            print(f"  Vec {i}: {len(idx_cpu)} indices, {contiguous} contiguous steps, {jumps} jumps")
            if len(idx_cpu) <= 20:
                print(f"         First 20: {idx_cpu[:20].tolist()}")
            else:
                print(f"         First 10: {idx_cpu[:10].tolist()}")
                print(f"         Last 10:  {idx_cpu[-10:].tolist()}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    run_benchmark()
    run_memory_analysis()
