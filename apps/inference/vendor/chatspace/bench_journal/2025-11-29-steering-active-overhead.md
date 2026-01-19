# Steering Runtime Active Overhead Benchmark

**Date**: 2025-11-29
**Issue**: #12 (Performance Optimization Epic)
**Branch**: `steerllm-extraction`

## Objective

Measure the throughput overhead of VLLMSteerModel when steering IS active, compared to idle baseline.

## Hardware

- 2x NVIDIA H200 (143GB each)
- Tensor parallel size: 2

## Model

- Qwen/Qwen3-32B
- Eager execution mode (no CUDA graphs)
- Prefix caching enabled

## Workload

- 64 concurrent requests
- 100 max tokens per request
- ~100 prompt tokens per request (10x "The quick brown fox...")
- Temperature: 0.7

## Bug Fixed During Testing

**Issue**: Shape mismatch in `_apply_per_request_steering()` with large batches.
- `transformed_hidden` had shape (1979, hidden_size) - actual tokens
- `residual` had shape (2048, hidden_size) - vLLM's padded tensor

**Fix**: Changed from slice-and-concat to clone-and-modify-in-place:
```python
# Before (broken with padding):
request_slices = [hidden[start:end] for ...]
transformed_hidden = torch.cat(request_slices, dim=0)  # Wrong shape!

# After (works with padding):
transformed_hidden = hidden.clone()
transformed_hidden[start:end] = steered_slice  # Same shape preserved
```

## Results

| Case | Throughput (tok/s) | Overhead vs Idle |
|------|-------------------|------------------|
| baseline_idle | 2161 | - |
| steering_1layer | 1995 | ~8% |
| steering_8layer | 1665 | ~23% |

## Analysis

1. **1-layer steering overhead (~8%)**: Reasonable for per-layer tensor operations
   - Clone hidden state
   - Apply steering operation (vector addition)
   - Reconstruct output with modified delta

2. **8-layer steering overhead (~23%)**: Linear-ish scaling with active layers
   - 8 layers × ~3% per layer ≈ 24%
   - Overhead is dominated by tensor clone and reconstruction

## Conclusions

1. **Steering overhead scales with active layers** - not surprising, more layers = more work

2. **No easy wins from speculative optimizations** - the overhead is real tensor work:
   - Clone: O(seq_len × hidden_size)
   - Steering op: O(seq_len × hidden_size)
   - Reconstruction: O(seq_len × hidden_size)

3. **Potential optimizations for future** (if overhead becomes problematic):
   - Fused kernel for clone+add+reconstruct
   - Skip clone when no steering changes needed (early exit per-slice)
   - Batch steering ops across multiple requests

## Test Gap Identified

Tests only covered TP=1 with small batches. The shape mismatch bug only manifested with:
- Large concurrent batches (64 requests)
- Tensor-parallel execution
- vLLM's internal tensor padding

**Action**: Consider adding TP>1 integration tests for steering.
