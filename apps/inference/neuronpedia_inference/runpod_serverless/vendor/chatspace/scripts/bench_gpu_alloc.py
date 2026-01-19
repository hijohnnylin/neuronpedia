#!/usr/bin/env python3
"""
Benchmark GPU-side tensor allocation vs CPU-side.

Options:
1. torch.tensor(list, device='cuda') - CPU list → GPU tensor (current)
2. Pre-allocated GPU buffer + copy - avoid repeated allocation
3. torch ops on GPU (arange, repeat, etc.) - pure GPU construction

Usage:
    PYTHONPATH=/root/chatspace uv run python scripts/bench_gpu_alloc.py
"""
import torch
import time


def bench(fn, num_iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / num_iters * 1e6


def run_allocation_benchmark():
    print("=" * 80)
    print("Tensor Allocation Benchmark: CPU vs GPU")
    print("=" * 80)

    device = "cuda"
    batch_size = 32
    seq_len = 128
    total_tokens = batch_size * seq_len
    hidden_size = 4096

    # Simulate index building for heterogeneous batch
    # Each request has its own range [start, end)
    python_indices = list(range(total_tokens))  # All tokens
    python_assignments = []
    for i in range(batch_size):
        python_assignments.extend([i] * seq_len)

    print(f"\nScenario: {batch_size} requests, {seq_len} tokens each, {total_tokens} total indices")

    # Method 1: torch.tensor from Python list (current approach)
    def cpu_to_gpu():
        idx = torch.tensor(python_indices, device=device, dtype=torch.long)
        assign = torch.tensor(python_assignments, device=device, dtype=torch.long)
        return idx, assign

    # Method 2: Pre-allocated GPU buffers
    idx_buffer = torch.empty(total_tokens, device=device, dtype=torch.long)
    assign_buffer = torch.empty(total_tokens, device=device, dtype=torch.long)
    idx_cpu = torch.tensor(python_indices, dtype=torch.long)
    assign_cpu = torch.tensor(python_assignments, dtype=torch.long)

    def preallocated_copy():
        idx_buffer.copy_(idx_cpu)
        assign_buffer.copy_(assign_cpu)
        return idx_buffer, assign_buffer

    # Method 3: Pure GPU construction using torch ops
    def gpu_construct():
        # Build indices: 0,1,2,...,total_tokens-1
        idx = torch.arange(total_tokens, device=device, dtype=torch.long)
        # Build assignments: 0,0,0...(seq_len times), 1,1,1..., etc.
        assign = torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(seq_len)
        return idx, assign

    # Method 4: GPU construct with pre-computed patterns
    base_range = torch.arange(seq_len, device=device, dtype=torch.long)

    def gpu_construct_optimized():
        # Use broadcasting/repeat to build indices
        offsets = torch.arange(0, total_tokens, seq_len, device=device, dtype=torch.long)
        idx = (base_range.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        assign = torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(seq_len)
        return idx, assign

    # Method 5: For heterogeneous - build with CUDA tensor directly
    def gpu_from_cpu_tensor():
        # Pre-convert to CPU tensor (faster than list)
        idx = idx_cpu.to(device=device)
        assign = assign_cpu.to(device=device)
        return idx, assign

    t1 = bench(cpu_to_gpu)
    t2 = bench(preallocated_copy)
    t3 = bench(gpu_construct)
    t4 = bench(gpu_construct_optimized)
    t5 = bench(gpu_from_cpu_tensor)

    print(f"\n{'Method':<40} {'Time (μs)':>12} {'vs Current':>12}")
    print("-" * 65)
    print(f"{'1. torch.tensor(list, device=cuda)':<40} {t1:>12.1f} {1.0:>11.2f}x")
    print(f"{'2. Pre-alloc buffer + copy from CPU':<40} {t2:>12.1f} {t1/t2:>11.2f}x")
    print(f"{'3. Pure GPU (arange + repeat_interleave)':<40} {t3:>12.1f} {t1/t3:>11.2f}x")
    print(f"{'4. GPU optimized (broadcast)':<40} {t4:>12.1f} {t1/t4:>11.2f}x")
    print(f"{'5. CPU tensor.to(cuda)':<40} {t5:>12.1f} {t1/t5:>11.2f}x")


def run_realistic_fused_benchmark():
    print("\n" + "=" * 80)
    print("Realistic Fused Gather with GPU-Constructed Indices")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 32
    seq_len = 128
    hidden_size = 4096
    total_tokens = batch_size * seq_len

    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

    # Create unique vector per request
    vecs = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)

    # Current approach: Python list → torch.tensor
    def fused_current():
        h = hidden.clone()
        indices = list(range(total_tokens))
        assignments = []
        for i in range(batch_size):
            assignments.extend([i] * seq_len)

        idx = torch.tensor(indices, device=device, dtype=torch.long)
        assign = torch.tensor(assignments, device=device, dtype=torch.long)

        gathered = h[idx]
        vec_for_each = vecs[assign]
        gathered = gathered + vec_for_each
        h[idx] = gathered
        return h

    # GPU-constructed indices
    def fused_gpu_indices():
        h = hidden.clone()
        idx = torch.arange(total_tokens, device=device, dtype=torch.long)
        assign = torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(seq_len)

        gathered = h[idx]
        vec_for_each = vecs[assign]
        gathered = gathered + vec_for_each
        h[idx] = gathered
        return h

    # Even simpler: if all tokens are contiguous and ordered, we don't need gather
    def fused_no_gather():
        h = hidden.clone()
        # Expand vecs to match hidden shape: [batch, hidden] → [batch*seq_len, hidden]
        vec_expanded = vecs.repeat_interleave(seq_len, dim=0)
        h = h + vec_expanded
        return h

    # Slow loop baseline
    def slow_loop():
        h = hidden.clone()
        for i in range(batch_size):
            start = i * seq_len
            end = start + seq_len
            h[start:end] = h[start:end] + vecs[i]
        return h

    t_current = bench(fused_current)
    t_gpu = bench(fused_gpu_indices)
    t_no_gather = bench(fused_no_gather)
    t_slow = bench(slow_loop)

    print(f"\nBatch={batch_size}, SeqLen={seq_len}, all unique vectors:")
    print(f"\n{'Method':<40} {'Time (μs)':>12} {'vs Slow':>12}")
    print("-" * 65)
    print(f"{'Slow loop (baseline)':<40} {t_slow:>12.1f} {1.0:>11.2f}x")
    print(f"{'Fused (Python list → tensor)':<40} {t_current:>12.1f} {t_slow/t_current:>11.2f}x")
    print(f"{'Fused (GPU-constructed indices)':<40} {t_gpu:>12.1f} {t_slow/t_gpu:>11.2f}x")
    print(f"{'Fused (no gather, expand vecs)':<40} {t_no_gather:>12.1f} {t_slow/t_no_gather:>11.2f}x")

    # Verify correctness
    h1 = slow_loop()
    h2 = fused_gpu_indices()
    h3 = fused_no_gather()
    assert torch.allclose(h1, h2, atol=1e-5), "GPU indices mismatch!"
    assert torch.allclose(h1, h3, atol=1e-5), "No-gather mismatch!"
    print("\n✓ All methods produce identical results")


def run_sparse_scenario():
    """Test when only some requests have steering (sparse)."""
    print("\n" + "=" * 80)
    print("Sparse Scenario: Only Some Requests Have Steering")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 32
    seq_len = 128
    hidden_size = 4096
    total_tokens = batch_size * seq_len

    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

    # Only half the requests have steering
    steered_requests = list(range(0, batch_size, 2))  # Every other request
    num_steered = len(steered_requests)

    vecs = torch.randn(num_steered, hidden_size, device=device, dtype=dtype)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)

    # GPU-constructed indices for sparse case
    def fused_sparse_gpu():
        h = hidden.clone()
        # Build indices only for steered requests
        steered_t = torch.tensor(steered_requests, device=device, dtype=torch.long)
        base_offsets = steered_t * seq_len  # [num_steered]
        within_seq = torch.arange(seq_len, device=device, dtype=torch.long)
        idx = (base_offsets.unsqueeze(1) + within_seq.unsqueeze(0)).flatten()  # [num_steered * seq_len]
        assign = torch.arange(num_steered, device=device, dtype=torch.long).repeat_interleave(seq_len)

        gathered = h[idx]
        vec_for_each = vecs[assign]
        gathered = gathered + vec_for_each
        h[idx] = gathered
        return h

    # Slow loop for sparse
    def slow_sparse():
        h = hidden.clone()
        for vec_idx, req_idx in enumerate(steered_requests):
            start = req_idx * seq_len
            end = start + seq_len
            h[start:end] = h[start:end] + vecs[vec_idx]
        return h

    t_gpu = bench(fused_sparse_gpu)
    t_slow = bench(slow_sparse)

    print(f"\nSteered: {num_steered}/{batch_size} requests")
    print(f"\n{'Method':<40} {'Time (μs)':>12} {'vs Slow':>12}")
    print("-" * 65)
    print(f"{'Slow loop (sparse)':<40} {t_slow:>12.1f} {1.0:>11.2f}x")
    print(f"{'Fused (GPU indices, sparse)':<40} {t_gpu:>12.1f} {t_slow/t_gpu:>11.2f}x")

    # Verify
    assert torch.allclose(slow_sparse(), fused_sparse_gpu(), atol=1e-5)
    print("\n✓ Results match")


if __name__ == "__main__":
    run_allocation_benchmark()
    run_realistic_fused_benchmark()
    run_sparse_scenario()
