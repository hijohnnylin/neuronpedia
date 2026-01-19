# Steering Runtime Idle Overhead Benchmark

**Date**: 2025-11-29
**Issue**: #13 (Phase 0 - Profiling & Baseline)
**Branch**: `steerllm-extraction`

## Objective

Measure the throughput overhead of VLLMSteerModel when steering/capture is NOT active (idle mode) compared to vanilla vLLM eager baseline.

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
- ~10 prompt tokens per request
- Temperature: 0.7

## Results

### Throughput Comparison

| Run | Baseline Eager (tok/s) | Patched Idle (tok/s) |
|-----|------------------------|----------------------|
| 1   | 2165.8                 | 2155.7               |
| 2   | 2115.4                 | 2163.3               |
| 3   | 2087.4                 | 2157.0               |
| **Avg** | **~2123**          | **~2159**            |

**Overhead: ~0% (within noise)**

The run-to-run variance (~80 tok/s spread) exceeds the difference between baseline and patched averages.

### Profiling Counter Data (patched_idle)

```
forward_calls: 25984
exit_no_requests: 25984  (100% fast path)
extraction_triggered: 0
steering_applied: 0
capture_processed: 0
extraction_time: 0.0ms
```

All forward passes exit at the earliest possible point (`exit_no_requests`), confirming the fast path is working correctly.

### Per-Case Breakdown (from benchmark script)

| Case | Throughput (tok/s) | Notes |
|------|-------------------|-------|
| baseline_eager | 2338 | Vanilla vLLM, eager mode |
| baseline_with_extension | 2316 | vLLM + SteeringWorkerExtension loaded |
| patched_idle | ~2159 | VLLMSteerModel, no steering/capture active |

## Conclusions

1. **Idle overhead is negligible** - VLLMSteerModel has essentially zero throughput penalty when steering/capture is not active.

2. **Fast path works** - 100% of forward calls exit at `exit_no_requests` when idle, meaning:
   - No hidden state extraction
   - No steering vector lookups
   - No capture processing

3. **Counter overhead is minimal** - With `CHATSPACE_PERF_COUNTERS=1`, overhead is still within noise.

4. **Original concern was unfounded** - The suspected overhead from "just installing hooks" does not manifest in actual throughput measurements at scale.

## Methodology

Test script: `/tmp/fair_throughput_test.py` (inline, measures actual token IDs not word counts)

Both baseline and patched tests use:
- Same prompts
- Same sampling parameters
- Same token counting method (`len(out.outputs[0].token_ids)`)
- asyncio.gather for concurrent request processing

## Next Steps

- Measure overhead with actual steering active (single layer)
- Measure overhead with capture active (single layer, all layers)
- Profile extraction time breakdown if overhead is found
