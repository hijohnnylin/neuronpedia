# vLLM Health Monitoring & Debugging

This document describes how to monitor your vLLM instance for debugging hanging requests and performance issues.

## Quick Start

### 1. Test Monitor Locally
```bash
cd /root/np-cap/apps/inference/neuronpedia_inference/runpod_serverless/src
python ../scripts/test_monitor.py
```

### 2. Enable Background Monitoring
Set these environment variables before starting the handler:

```bash
export ENABLE_VLLM_MONITOR=1          # Enable periodic health logging
export VLLM_MONITOR_INTERVAL=30       # Log stats every 30 seconds
```

### 3. Get Health Stats via API
Send a request with `"action": "health"`:

```python
response = await client.run({
    "input": {
        "action": "health"
    }
})
# Returns GPU usage, RAM, active requests, threads, etc.
```

---

## Health Stats Output

The monitor provides:

| Metric | Description |
|--------|-------------|
| `num_running_requests` | Requests currently being processed |
| `num_waiting_requests` | Requests queued waiting for processing |
| `num_swapped_requests` | Requests swapped to CPU (memory pressure) |
| `gpu_cache_usage_percent` | KV cache GPU memory usage |
| `cpu_cache_usage_percent` | KV cache CPU memory usage |
| `system_ram_used_mb` | System RAM used |
| `process_ram_mb` | Python process RAM |
| `gpu_stats` | Per-GPU VRAM usage and utilization |
| `active_threads` | Python thread count |
| `pending_tasks` | Asyncio task count |

---

## vLLM Built-in Debugging

### Enable Detailed Logging

```bash
# Increase vLLM logging verbosity
export VLLM_LOGGING_LEVEL=DEBUG

# Log stats more frequently (seconds between stat dumps)
export VLLM_LOG_STATS_INTERVAL=5

# Trace all function calls (VERY verbose, use sparingly)
export VLLM_TRACE_FUNCTION=1
```

### CUDA/NCCL Debugging (for multi-GPU hangs)

```bash
# Enable NCCL debug output
export NCCL_DEBUG=INFO      # or TRACE for more detail

# Force synchronous CUDA operations (helps identify where hangs occur)
export CUDA_LAUNCH_BLOCKING=1
```

---

## Common Causes of Hanging Requests

### 1. GPU Memory Exhaustion
**Symptoms**: Requests hang indefinitely, GPU memory at 100%

**Check**:
```python
stats = await monitor.get_stats()
print(f"GPU: {stats.gpu_stats[0].memory_percent:.1f}%")
```

**Fix**: Reduce `GPU_MEMORY_UTILIZATION` env var (default 0.95):
```bash
export GPU_MEMORY_UTILIZATION=0.85
```

### 2. KV Cache Overflow
**Symptoms**: Requests stuck in waiting/swapped state

**Check**: Look for high `num_waiting_requests` or `num_swapped_requests`

**Fix**: Reduce `max_model_len` or batch size

### 3. Long Prefill Phase
**Symptoms**: First token takes very long on long prompts

**Check**: Timing logs show long gaps before first token

**Fix**: Enable chunked prefill (if not using steering):
```bash
export ENABLE_CHUNKED_PREFILL=1  
```

### 4. CUDA Kernel Stall
**Symptoms**: Completely frozen, no logs

**Check**: Run with `CUDA_LAUNCH_BLOCKING=1` to isolate

**Fix**: Usually driver/hardware issue, try restarting pod

---

## Request Timing Logs

The handler now logs timing for each request:

```
[REQUEST START] action=generate
[REQUEST] Ensuring engine initialized...
[REQUEST] Engine ready (took 0.01s)
[REQUEST] Starting generation: types=['STEERED'], max_tokens=512
[REQUEST COMPLETE] total=3.45s, generation=3.21s, ~tokens=87, tokens/sec=27.1
```

If you see `[REQUEST] Starting generation` but no `[REQUEST COMPLETE]`, the hang is in vLLM generation.

---

## Programmatic Monitoring

```python
from vllm_monitor import VLLMMonitor, get_health_stats

# Option 1: Use global monitor
stats = await get_health_stats(model_manager)
print(stats.summary())

# Option 2: Create your own monitor
monitor = VLLMMonitor(model_manager)

# One-time stats
stats = await monitor.get_stats()

# Background logging
monitor.start_background_logging(interval=10.0)

# Stop background logging
monitor.stop_background_logging()
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_VLLM_MONITOR` | `0` | Enable background health logging |
| `VLLM_MONITOR_INTERVAL` | `30` | Seconds between health logs |
| `VLLM_LOGGING_LEVEL` | `INFO` | vLLM log verbosity |
| `VLLM_LOG_STATS_INTERVAL` | `5` | Seconds between vLLM stat dumps |
| `GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of GPU memory to use |
| `MAX_MODEL_LEN` | varies | Maximum sequence length |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for TP |

---

## Files Added

### RunPod Serverless
- `runpod_serverless/src/vllm_monitor.py` - Health monitoring module
- `runpod_serverless/scripts/test_monitor.py` - Test script
- `runpod_serverless/MONITORING.md` - This document
- `runpod_serverless/src/handler.py` - Updated with request timing logs, health check action, background monitoring

### Non-Serverless (FastAPI)
- `inference_utils/vllm_monitor.py` - Health monitoring module
- `endpoints/steer/completion_chat.py` - Updated with:
  - Request timing logs
  - `/steer/health` endpoint for health checks
  - Optional background monitoring

---

## Non-Serverless Usage

### Health Check Endpoint

```bash
curl http://localhost:8000/steer/health
```

Returns:
```json
{
  "stats": {
    "timestamp": 1234567890.123,
    "gpu_stats": [...],
    "system_ram_used_mb": 1234.5,
    ...
  },
  "summary": "=== vLLM Health Stats @ 12:34:56 ===\n..."
}
```

### Request Timing Logs

The endpoint now logs timing for each request:

```
[REQUEST START] completion-chat
[REQUEST] Starting generation: types=['STEERED'], max_tokens=512
[REQUEST COMPLETE] total=3.45s, generation=3.21s, ~chunks=87
```

