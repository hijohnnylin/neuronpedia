"""Benchmark steering and capture overhead vs vanilla vLLM.

This script measures the performance impact of the steering patches and runtime
hooks. It runs in two modes:
1. Orchestrator: Launches separate subprocesses for each test case to ensure
   clean process state (crucial for comparing unpatched vs patched).
2. Worker: Executes a specific benchmark case and prints metrics as JSON.

Usage:
    uv run python scripts/benchmark_steering_overhead.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import torch
import numpy as np
from vllm import SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

@dataclass
class BenchmarkResult:
    case_name: str
    throughput_tokens_sec: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    total_time_sec: int


def get_dummy_prompts(count: int, token_len: int = 100) -> list[str]:
    """Generate dummy prompts that are roughly `token_len` long."""
    # A simple repeating pattern. Qwen tokenizer is approx 3-4 chars per token.
    # We'll just use a long string.
    base = "The quick brown fox jumps over the lazy dog. "
    repetitions = (token_len * 4) // len(base) + 1
    text = base * repetitions
    return [f"Request {i}: {text}" for i in range(count)]


async def run_baseline_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Run vanilla vLLM benchmark (no steering patches)."""
    from vllm import AsyncLLMEngine, AsyncEngineArgs

    logger.info(f"Starting BASELINE worker for {args.model_name}")
    
    # Handle max_model_len=0 as None (passed from orchestrator)
    max_model_len = args.max_model_len if args.max_model_len and args.max_model_len > 0 else None
    
    is_eager = args.case == "baseline_eager" or args.case == "baseline_with_extension"
    logger.info(f"Starting BASELINE worker for {args.model_name} (Eager={is_eager})")

    # Setup worker extension if requested
    worker_extension_cls = None
    if args.case == "baseline_with_extension":
        worker_extension_cls = "chatspace.vllm_steering.runtime.SteeringWorkerExtension"

    engine_kwargs = {
        "model": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "enforce_eager": is_eager,  # Allow CUDA graphs for fair baseline comparison unless explicit
        "enable_prefix_caching": not args.disable_prefix_caching,
    }
    
    if worker_extension_cls:
        engine_kwargs["worker_extension_cls"] = worker_extension_cls

    engine_args = AsyncEngineArgs(**engine_kwargs)
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    return await _run_workload(engine, args)


async def run_patched_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Run patched VLLMSteerModel benchmark."""
    from chatspace.generation.vllm_steer_model import (
        VLLMSteerModel, VLLMSteeringConfig, SteeringSpec, LayerSteeringSpec, AddSpec
    )
    
    logger.info(f"Starting PATCHED worker for {args.model_name} (Case: {args.case})")
    
    # Handle max_model_len=0 as None
    max_model_len = args.max_model_len if args.max_model_len and args.max_model_len > 0 else None

    # Patched model always enforces eager execution
    cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    
    # Set diagnostic env vars for special cases
    if args.case == "patched_hooks_disabled":
        os.environ["CHATSPACE_DISABLE_STEERING_HOOKS"] = "1"
    elif args.case == "patched_no_wrapper":
        os.environ["CHATSPACE_DISABLE_MODEL_WRAPPER"] = "1"
    elif args.case == "patched_no_layer_patch":
        os.environ["CHATSPACE_DISABLE_LAYER_PATCH"] = "1"
    elif args.case == "patched_no_runner_patch":
        os.environ["CHATSPACE_DISABLE_RUNNER_PATCH"] = "1"
    elif args.case == "patched_all_disabled":
        os.environ["CHATSPACE_DISABLE_STEERING_HOOKS"] = "1"
        os.environ["CHATSPACE_DISABLE_MODEL_WRAPPER"] = "1"
        os.environ["CHATSPACE_DISABLE_LAYER_PATCH"] = "1"
        os.environ["CHATSPACE_DISABLE_RUNNER_PATCH"] = "1"
    
    # Pass enable_prefix_caching via kwargs
    model = VLLMSteerModel(
        cfg, 
        enforce_eager=True, 
        enable_prefix_caching=not args.disable_prefix_caching,
    )
    
    # Prepare steering spec and capture args based on case
    steering_spec = None
    capture_layers = None
    
    # We need to know dimensions to create vectors. 
    # Model lazy loads, so we force init by accessing properties or running a warmup.
    # However, we can just generate once we have the model instance.
    # Note: VLLMSteerModel initializes engine on first generate() call.
    
    # Create a dummy spec to get dimensions if needed (will happen inside workload after init)
    
    async def workload_runner(prompts, sampling_params):
        # We need to construct the spec *after* model init usually to know hidden size, 
        # but VLLMSteerModel loads config in __init__, so we have self.hidden_size.
        
        nonlocal steering_spec, capture_layers
        
        hidden_size = model.hidden_size
        layer_count = model.layer_count
        
        # Construct spec based on case
        if "steering" in args.case:
            layers_to_steer = []
            if "1layer" in args.case:
                layers_to_steer = [16] if layer_count > 16 else [0]
            elif "all" in args.case:
                layers_to_steer = list(range(layer_count))
            
            spec_dict = {}
            vector = torch.ones(hidden_size, dtype=torch.float32) 
            # Normalize
            vector = vector / vector.norm()
            
            for l in layers_to_steer:
                spec_dict[l] = LayerSteeringSpec(
                    add=AddSpec(vector=vector, scale=1.5)
                )
            steering_spec = SteeringSpec(layers=spec_dict)
            
        if "capture" in args.case:
            if "1layer" in args.case:
                capture_layers = [16] if layer_count > 16 else [0]
            elif "all" in args.case:
                capture_layers = list(range(layer_count))
        
        logger.info(f"Workload config: Steering={steering_spec is not None}, Capture={capture_layers}")
        
        # Run generation
        start_time = time.perf_counter()
        
        # To measure throughput properly with async, we launch all requests
        tasks = []
        for i, prompt in enumerate(prompts):
            # VLLMSteerModel.generate handles batching internally if passed a list,
            # but we want to mimic the engine loop style or just use the batch API.
            # The batch API `model.generate(prompts, ...)` is easiest and representative.
            pass
            
        # For patched model, we use the high-level generate API which handles 
        # steering spec serialization and capture handle management.
        
        # Batch generate
        if capture_layers:
            outputs, handles = await model.generate(
                prompts, 
                sampling_params, 
                steering_spec=steering_spec,
                capture_layers=capture_layers
            )
            
            # Simulate fetching the captures to include that overhead
            fetch_tasks = [h.fetch() for h in handles]
            await asyncio.gather(*fetch_tasks)
            
            # Cleanup handles
            cleanup_tasks = [h.close() for h in handles]
            await asyncio.gather(*cleanup_tasks)
            
            # Count tokens
            total_tokens = sum(len(o.split()) for o in outputs) # Rough approximation or use raw_output
        else:
            outputs = await model.generate(
                prompts,
                sampling_params,
                steering_spec=steering_spec
            )
            total_tokens = sum(len(o.split()) for o in outputs)

        end_time = time.perf_counter()
        return total_tokens, end_time - start_time

    return await _run_custom_workload(model, args, workload_runner)


async def _run_workload(engine, args) -> dict[str, Any]:
    """Common workload runner for vanilla vLLM engine."""
    prompts = get_dummy_prompts(args.num_requests, args.prompt_len)
    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=args.max_tokens,
        ignore_eos=True
    )
    
    # Warmup
    logger.info("Warming up...")
    async for _ in engine.generate(prompts[0], sampling_params, request_id="warmup"):
        pass
        
    logger.info("Starting benchmark...")
    start_time = time.perf_counter()
    
    request_tracker = []
    
    async def process_request(i, prompt):
        req_id = f"req_{i}"
        token_count = 0
        req_start = time.perf_counter()
        async for out in engine.generate(prompt, sampling_params, request_id=req_id):
            if out.finished:
                token_count = len(out.outputs[0].token_ids)
        return token_count, time.perf_counter() - req_start

    tasks = [process_request(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    total_tokens = sum(r[0] for r in results)
    latencies = [r[1] * 1000 for r in results] # ms
    
    return {
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "avg_latency_ms": np.mean(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "throughput_tokens_sec": total_tokens / total_time
    }

def _aggregate_worker_counters(worker_data: list[dict]) -> dict:
    """Sum counters across all tensor-parallel workers."""
    totals = {"counters": {}, "timings": {}}
    for wd in worker_data:
        for k, v in wd.get("counters", {}).items():
            totals["counters"][k] = totals["counters"].get(k, 0) + v
        for k, v in wd.get("timings", {}).items():
            totals["timings"][k] = totals["timings"].get(k, 0.0) + v
    return totals


async def _run_custom_workload(model, args, runner_fn) -> dict[str, Any]:
    """Wrapper for patched model workload."""
    prompts = get_dummy_prompts(args.num_requests, args.prompt_len)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=args.max_tokens,
        ignore_eos=True
    )

    # Warmup
    logger.info("Warming up...")
    # Just run one prompt
    await model.generate([prompts[0]], sampling_params)

    logger.info("Starting benchmark...")
    # The runner_fn handles the batch generation and timing
    total_tokens, total_time = await runner_fn(prompts, sampling_params)

    # Fetch perf counters if enabled
    perf_data = None
    if os.environ.get("CHATSPACE_PERF_COUNTERS") == "1":
        try:
            raw_counters = await model.get_perf_counters()
            perf_data = _aggregate_worker_counters(raw_counters)
            logger.info(f"Perf counters: {perf_data}")
        except Exception as e:
            logger.warning(f"Failed to fetch perf counters: {e}")

    # Estimate per-request latency as total_time (since it's a batch, latency is concurrent)
    # For throughput: total_tokens / total_time
    # For latency: This batch call waits for all.
    # Ideally we'd measure TTFT etc, but coarse overhead is fine.
    # We'll report the batch latency as "avg latency" approx if we treat it as one big batch,
    # but really we want per-request stats.
    # VLLMSteerModel.generate doesn't return per-request timings easily without raw_output=True.

    # Let's simplify: Overhead is best seen in aggregate Throughput.

    result = {
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "avg_latency_ms": (total_time / args.num_requests) * 1000, # Rough avg completion time
        "p95_latency_ms": 0.0, # Not available in batch mode easily
        "throughput_tokens_sec": total_tokens / total_time
    }

    if perf_data:
        result["perf_counters"] = perf_data

    return result


def run_worker_mode(args):
    """Entry point for worker process."""
    try:
        if "baseline" in args.case:
            result = asyncio.run(run_baseline_worker(args))
        else:
            result = asyncio.run(run_patched_worker(args))
        
        # Print JSON result to stdout for parent to capture
        print(json.dumps(result))
    except Exception as e:
        logger.exception("Worker failed")
        sys.exit(1)


def run_orchestrator_mode(args):
    """Entry point for orchestrator."""
    
    cases = [
        "baseline_eager",
        "baseline_with_extension",
        "patched_idle",
        "patched_all_disabled",
    ]
    
    results = []
    
    print(f"{'Case':<25} | {'Throughput (tok/s)':<20} | {'Latency (ms)':<15} | {'Overhead %':<10}")
    print("-" * 80)
    
    baseline_throughput = 0.0
    
    for case in cases:
        logger.info(f"Running case: {case}")
        
        cmd = [
            "uv", "run", "python", __file__,
            "--worker",
            "--case", case,
            "--model", args.model_name,
            "--num-requests", str(args.num_requests),
            "--max-tokens", str(args.max_tokens),
            "--max-model-len", str(args.max_model_len or 0), # 0 or None handling
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        ]
        
        if args.disable_prefix_caching:
            cmd.append("--disable-prefix-caching")
        
        try:
            # Run process
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=os.environ.copy()
            )
            
            # Parse last line as JSON (ignoring logs)
            lines = process.stdout.strip().split('\n')
            json_line = lines[-1]
            data = json.loads(json_line)
            
            tput = data['throughput_tokens_sec']
            lat = data['avg_latency_ms']
            
            if case == "baseline":
                baseline_throughput = tput
                overhead = 0.0
            else:
                overhead = (1.0 - (tput / baseline_throughput)) * 100 if baseline_throughput > 0 else 0.0
            
            print(f"{case:<25} | {tput:<20.2f} | {lat:<15.2f} | {overhead:<10.1f}")

            # Display perf counters if available
            if "perf_counters" in data:
                pc = data["perf_counters"]
                counters = pc.get("counters", {})
                timings = pc.get("timings", {})

                # Key metrics
                fwd = counters.get("forward_calls", 0)
                exit_no_work = counters.get("exit_no_work", 0)
                extraction = counters.get("extraction_triggered", 0)
                steering = counters.get("steering_applied", 0)
                capture = counters.get("capture_processed", 0)
                extraction_time = timings.get("extraction_time", 0.0)

                print(f"  Counters: fwd={fwd}, exit_no_work={exit_no_work}, "
                      f"extraction={extraction}, steering={steering}, capture={capture}")
                if fwd > 0:
                    print(f"  Extraction rate: {extraction/fwd*100:.1f}%, "
                          f"Extraction time: {extraction_time*1000:.2f}ms total")

            results.append({
                "case": case,
                "metrics": data,
                "overhead_percent": overhead
            })
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Case {case} failed:")
            logger.error(e.stderr)
            print(f"{case:<25} | {'FAILED':<20} | {'-':<15} | {'-':<10}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse output for {case}")
            print(f"{case:<25} | {'PARSE ERROR':<20} | {'-':<15} | {'-':<10}")

    # Save full report
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run in worker mode")
    parser.add_argument("--case", type=str, default="baseline", 
                      choices=["baseline", "baseline_eager", "baseline_with_extension", 
                               "patched_idle", "patched_hooks_disabled", "patched_no_wrapper",
                               "patched_no_layer_patch", "patched_no_runner_patch", "patched_all_disabled",
                               "capture_1layer", "steering_1layer", 
                               "capture_all", "steering_all", "capture_steering_all"])
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--disable-prefix-caching", action="store_true", help="Disable vLLM prefix caching")
    
    args = parser.parse_args()
    
    if args.worker:
        run_worker_mode(args)
    else:
        run_orchestrator_mode(args)

