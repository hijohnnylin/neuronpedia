"""End-to-end benchmark for steering performance through full vLLM runtime.

Tests actual throughput through VLLMSteeringModel.generate() with:
1. No steering (baseline)
2. Same steering spec for all requests (should hit fast path)
3. Unique steering specs per request (concurrent single-request calls)

Usage:
    uv run python scripts/benchmark_steering_e2e.py
"""

import asyncio
import os
import sys
import time
import torch
import numpy as np
from dataclasses import dataclass

# Add parent dir to path for steerllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from steerllm import VLLMSteeringModel, SteeringSpec

MODEL_NAME = "Qwen/Qwen3-0.6B"
WARMUP_ITERS = 2
BENCH_ITERS = 5


@dataclass
class BenchResult:
    name: str
    total_time_s: float
    num_requests: int
    tokens_generated: int
    requests_per_sec: float
    tokens_per_sec: float


def print_results(results: list[BenchResult]):
    """Print results as a table."""
    print(f"\n{'='*80}")
    print(f" End-to-End Steering Benchmark Results")
    print(f"{'='*80}")
    print(f"{'Name':<40} | {'Req/s':<10} | {'Tok/s':<10} | {'Total (s)':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<40} | {r.requests_per_sec:<10.1f} | {r.tokens_per_sec:<10.1f} | {r.total_time_s:<10.3f}")
    print()


async def bench_no_steering(
    model: VLLMSteeringModel,
    prompts: list[str],
    max_tokens: int,
    iters: int,
) -> BenchResult:
    """Benchmark generation without steering."""
    # Warmup
    for _ in range(WARMUP_ITERS):
        await model.generate(prompts, max_tokens=max_tokens, temperature=0.0)

    # Timed runs
    times = []
    total_tokens = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        texts, _ = await model.generate(prompts, max_tokens=max_tokens, temperature=0.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += sum(len(t.split()) for t in texts)  # Rough token count

    avg_time = np.mean(times)
    return BenchResult(
        name="No steering (baseline)",
        total_time_s=avg_time,
        num_requests=len(prompts),
        tokens_generated=total_tokens // iters,
        requests_per_sec=len(prompts) / avg_time,
        tokens_per_sec=(total_tokens / iters) / avg_time,
    )


async def bench_same_steering(
    model: VLLMSteeringModel,
    prompts: list[str],
    max_tokens: int,
    steering_spec: SteeringSpec,
    iters: int,
) -> BenchResult:
    """Benchmark generation with same steering spec for all requests."""
    # Warmup
    for _ in range(WARMUP_ITERS):
        await model.generate(
            prompts, max_tokens=max_tokens, temperature=0.0, steering_spec=steering_spec
        )

    # Timed runs
    times = []
    total_tokens = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        texts, _ = await model.generate(
            prompts, max_tokens=max_tokens, temperature=0.0, steering_spec=steering_spec
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += sum(len(t.split()) for t in texts)

    avg_time = np.mean(times)
    return BenchResult(
        name="Same steering (all requests)",
        total_time_s=avg_time,
        num_requests=len(prompts),
        tokens_generated=total_tokens // iters,
        requests_per_sec=len(prompts) / avg_time,
        tokens_per_sec=(total_tokens / iters) / avg_time,
    )


async def bench_unique_steering_concurrent(
    model: VLLMSteeringModel,
    prompts: list[str],
    max_tokens: int,
    steering_specs: list[SteeringSpec],
    iters: int,
) -> BenchResult:
    """Benchmark generation with unique steering per request (concurrent calls)."""
    assert len(prompts) == len(steering_specs)

    async def generate_one(prompt: str, spec: SteeringSpec) -> str:
        texts, _ = await model.generate(
            [prompt], max_tokens=max_tokens, temperature=0.0, steering_spec=spec
        )
        return texts[0]

    # Warmup
    for _ in range(WARMUP_ITERS):
        tasks = [generate_one(p, s) for p, s in zip(prompts, steering_specs)]
        await asyncio.gather(*tasks)

    # Timed runs
    times = []
    total_tokens = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tasks = [generate_one(p, s) for p, s in zip(prompts, steering_specs)]
        texts = await asyncio.gather(*tasks)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += sum(len(t.split()) for t in texts)

    avg_time = np.mean(times)
    return BenchResult(
        name="Unique steering (concurrent)",
        total_time_s=avg_time,
        num_requests=len(prompts),
        tokens_generated=total_tokens // iters,
        requests_per_sec=len(prompts) / avg_time,
        tokens_per_sec=(total_tokens / iters) / avg_time,
    )


async def bench_multi_layer_steering(
    model: VLLMSteeringModel,
    prompts: list[str],
    max_tokens: int,
    vectors: list[torch.Tensor],
    layers: list[int],
    iters: int,
) -> BenchResult:
    """Benchmark generation with steering at multiple layers."""
    # Build multi-layer spec
    from steerllm import LayerSteeringSpec, AddSpec

    layer_specs = {}
    for layer, vec in zip(layers, vectors):
        layer_specs[layer] = LayerSteeringSpec(operations=[
            AddSpec.from_unnormalized(vec, scale=0.1)
        ])

    spec = SteeringSpec(layers=layer_specs)

    # Warmup
    for _ in range(WARMUP_ITERS):
        await model.generate(
            prompts, max_tokens=max_tokens, temperature=0.0, steering_spec=spec
        )

    # Timed runs
    times = []
    total_tokens = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        texts, _ = await model.generate(
            prompts, max_tokens=max_tokens, temperature=0.0, steering_spec=spec
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += sum(len(t.split()) for t in texts)

    avg_time = np.mean(times)
    return BenchResult(
        name=f"Same steering ({len(layers)} layers)",
        total_time_s=avg_time,
        num_requests=len(prompts),
        tokens_generated=total_tokens // iters,
        requests_per_sec=len(prompts) / avg_time,
        tokens_per_sec=(total_tokens / iters) / avg_time,
    )


async def bench_unique_steering_same_vector(
    model: VLLMSteeringModel,
    prompts: list[str],
    max_tokens: int,
    vector: torch.Tensor,
    layer: int,
    iters: int,
) -> BenchResult:
    """Benchmark unique specs that share the SAME vector (should benefit from interning)."""
    # Each request gets its own SteeringSpec object, but they all use the same vector
    # After interning, they should all share the same tensor identity

    async def generate_one(prompt: str) -> str:
        # Create fresh spec each time (simulates RPC where each request deserializes)
        spec = SteeringSpec.simple_add(layer=layer, vector=vector, scale=0.1)
        texts, _ = await model.generate(
            [prompt], max_tokens=max_tokens, temperature=0.0, steering_spec=spec
        )
        return texts[0]

    # Warmup
    for _ in range(WARMUP_ITERS):
        tasks = [generate_one(p) for p in prompts]
        await asyncio.gather(*tasks)

    # Timed runs
    times = []
    total_tokens = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tasks = [generate_one(p) for p in prompts]
        texts = await asyncio.gather(*tasks)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += sum(len(t.split()) for t in texts)

    avg_time = np.mean(times)
    return BenchResult(
        name="Unique specs, same vector (interning)",
        total_time_s=avg_time,
        num_requests=len(prompts),
        tokens_generated=total_tokens // iters,
        requests_per_sec=len(prompts) / avg_time,
        tokens_per_sec=(total_tokens / iters) / avg_time,
    )


async def bench_multi_layer_unique_vectors(
    model: VLLMSteeringModel,
    prompts: list[str],
    max_tokens: int,
    num_layers: int,
    layers: list[int],
    hidden_size: int,
    iters: int,
) -> BenchResult:
    """Benchmark unique vectors per request across multiple layers (worst case)."""
    from steerllm import LayerSteeringSpec, AddSpec

    async def generate_one(prompt: str, seed: int) -> str:
        # Create unique vectors for this request
        torch.manual_seed(seed)
        layer_specs = {}
        for layer in layers:
            vec = torch.randn(hidden_size)
            vec = vec / vec.norm()
            layer_specs[layer] = LayerSteeringSpec(operations=[
                AddSpec.from_unnormalized(vec, scale=0.1)
            ])
        spec = SteeringSpec(layers=layer_specs)
        texts, _ = await model.generate(
            [prompt], max_tokens=max_tokens, temperature=0.0, steering_spec=spec
        )
        return texts[0]

    # Warmup
    for _ in range(WARMUP_ITERS):
        tasks = [generate_one(p, i) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)

    # Timed runs
    times = []
    total_tokens = 0
    for iter_num in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tasks = [generate_one(p, iter_num * 1000 + i) for i, p in enumerate(prompts)]
        texts = await asyncio.gather(*tasks)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += sum(len(t.split()) for t in texts)

    avg_time = np.mean(times)
    return BenchResult(
        name=f"Unique vectors ({len(layers)} layers, concurrent)",
        total_time_s=avg_time,
        num_requests=len(prompts),
        tokens_generated=total_tokens // iters,
        requests_per_sec=len(prompts) / avg_time,
        tokens_per_sec=(total_tokens / iters) / avg_time,
    )


async def bench_multi_layer_shared_vectors(
    model: VLLMSteeringModel,
    prompts: list[str],
    max_tokens: int,
    shared_vectors: list[torch.Tensor],
    layers: list[int],
    iters: int,
) -> BenchResult:
    """Benchmark shared vectors (via interning) across multiple layers."""
    from steerllm import LayerSteeringSpec, AddSpec

    async def generate_one(prompt: str) -> str:
        # All requests use the SAME vectors (will be interned)
        layer_specs = {}
        for layer, vec in zip(layers, shared_vectors):
            layer_specs[layer] = LayerSteeringSpec(operations=[
                AddSpec.from_unnormalized(vec, scale=0.1)
            ])
        spec = SteeringSpec(layers=layer_specs)
        texts, _ = await model.generate(
            [prompt], max_tokens=max_tokens, temperature=0.0, steering_spec=spec
        )
        return texts[0]

    # Warmup
    for _ in range(WARMUP_ITERS):
        tasks = [generate_one(p) for p in prompts]
        await asyncio.gather(*tasks)

    # Timed runs
    times = []
    total_tokens = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tasks = [generate_one(p) for p in prompts]
        texts = await asyncio.gather(*tasks)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += sum(len(t.split()) for t in texts)

    avg_time = np.mean(times)
    return BenchResult(
        name=f"Shared vectors ({len(layers)} layers, interning)",
        total_time_s=avg_time,
        num_requests=len(prompts),
        tokens_generated=total_tokens // iters,
        requests_per_sec=len(prompts) / avg_time,
        tokens_per_sec=(total_tokens / iters) / avg_time,
    )


async def main():
    print("=" * 80)
    print(" End-to-End Steering Performance Benchmark")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model: {MODEL_NAME}")

    # Load model
    print("\nLoading model...")
    model = VLLMSteeringModel(
        MODEL_NAME,
        gpu_memory_utilization=0.3,
        max_model_len=512,
    )

    # Initialize
    await model.generate(["Hello"], max_tokens=1)
    hidden_size = model.hidden_size
    num_layers = model._layer_count
    print(f"Hidden size: {hidden_size}, Layers: {num_layers}")

    # Test parameters
    num_requests = 32
    max_tokens = 32
    steering_layer = num_layers // 2  # Middle layer

    # Create prompts
    prompts = [f"The meaning of life is" for _ in range(num_requests)]

    # Create steering vectors
    torch.manual_seed(42)
    shared_vector = torch.randn(hidden_size)
    shared_vector = shared_vector / shared_vector.norm()

    unique_vectors = [
        torch.randn(hidden_size) for _ in range(num_requests)
    ]
    unique_vectors = [v / v.norm() for v in unique_vectors]

    # Create specs
    shared_spec = SteeringSpec.simple_add(
        layer=steering_layer, vector=shared_vector, scale=0.1
    )
    unique_specs = [
        SteeringSpec.simple_add(layer=steering_layer, vector=v, scale=0.1)
        for v in unique_vectors
    ]

    # Create vectors for multi-layer tests
    multi_layer_vectors = [torch.randn(hidden_size) for _ in range(num_layers)]
    multi_layer_vectors = [v / v.norm() for v in multi_layer_vectors]

    print(f"\nBenchmark config:")
    print(f"  Requests: {num_requests}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Total layers: {num_layers}")
    print(f"  Warmup iters: {WARMUP_ITERS}")
    print(f"  Bench iters: {BENCH_ITERS}")

    results = []

    # Benchmark 1: No steering (baseline)
    print("\n[1/7] Benchmarking: No steering...")
    r = await bench_no_steering(model, prompts, max_tokens, BENCH_ITERS)
    results.append(r)
    baseline_rps = r.requests_per_sec
    print(f"  → {r.requests_per_sec:.1f} req/s")

    # Benchmark 2: Same steering, 1 layer
    print("\n[2/7] Benchmarking: Same steering (1 layer)...")
    r = await bench_same_steering(model, prompts, max_tokens, shared_spec, BENCH_ITERS)
    results.append(r)
    print(f"  → {r.requests_per_sec:.1f} req/s ({(baseline_rps - r.requests_per_sec)/baseline_rps*100:+.1f}%)")

    # Benchmark 3-5: Multi-layer steering (7, 14, 28 layers)
    layer_counts = [7, 14, num_layers]
    for i, n_layers in enumerate(layer_counts):
        print(f"\n[{3+i}/7] Benchmarking: Same steering ({n_layers} layers)...")
        # Select evenly spaced layers
        layers = [int(j * num_layers / n_layers) for j in range(n_layers)]
        vectors = [multi_layer_vectors[l] for l in layers]
        r = await bench_multi_layer_steering(model, prompts, max_tokens, vectors, layers, BENCH_ITERS)
        results.append(r)
        print(f"  → {r.requests_per_sec:.1f} req/s ({(baseline_rps - r.requests_per_sec)/baseline_rps*100:+.1f}%)")

    # Benchmark 6: Unique steering per request (1 layer)
    print("\n[6/7] Benchmarking: Unique steering (1 layer, concurrent)...")
    r = await bench_unique_steering_concurrent(
        model, prompts, max_tokens, unique_specs, BENCH_ITERS
    )
    results.append(r)
    print(f"  → {r.requests_per_sec:.1f} req/s ({(baseline_rps - r.requests_per_sec)/baseline_rps*100:+.1f}%)")

    # Benchmark 7: Unique specs but same vector content (tests interning)
    print("\n[7/11] Benchmarking: Same vector via interning (1 layer, concurrent)...")
    r = await bench_unique_steering_same_vector(
        model, prompts, max_tokens, shared_vector, steering_layer, BENCH_ITERS
    )
    results.append(r)
    print(f"  → {r.requests_per_sec:.1f} req/s ({(baseline_rps - r.requests_per_sec)/baseline_rps*100:+.1f}%)")

    # Benchmark 8-11: Multi-layer unique vs shared (interning benefit at scale)
    print("\n" + "="*60)
    print(" Multi-Layer: Unique vs Shared Vectors (Interning Benefit)")
    print("="*60)

    test_layer_counts = [14, 28]
    for n_layers in test_layer_counts:
        layers = [int(j * num_layers / n_layers) for j in range(n_layers)]
        vectors_for_layers = [multi_layer_vectors[l] for l in layers]

        # Unique vectors (worst case - no interning benefit)
        print(f"\n[{8 + test_layer_counts.index(n_layers)*2}/11] Unique vectors ({n_layers} layers, concurrent)...")
        r = await bench_multi_layer_unique_vectors(
            model, prompts, max_tokens, num_layers, layers, hidden_size, BENCH_ITERS
        )
        results.append(r)
        print(f"  → {r.requests_per_sec:.1f} req/s ({(baseline_rps - r.requests_per_sec)/baseline_rps*100:+.1f}%)")

        # Shared vectors (interning benefit)
        print(f"\n[{9 + test_layer_counts.index(n_layers)*2}/11] Shared vectors ({n_layers} layers, interning)...")
        r = await bench_multi_layer_shared_vectors(
            model, prompts, max_tokens, vectors_for_layers, layers, BENCH_ITERS
        )
        results.append(r)
        print(f"  → {r.requests_per_sec:.1f} req/s ({(baseline_rps - r.requests_per_sec)/baseline_rps*100:+.1f}%)")

    # Print final results
    print_results(results)

    # Calculate overheads
    baseline = results[0].requests_per_sec
    print("Overhead vs baseline:")
    for r in results[1:]:
        overhead = (baseline - r.requests_per_sec) / baseline * 100
        print(f"  {r.name}: {overhead:+.1f}%")

    # Calculate steering overhead breakdown
    print("\nAnalysis:")
    baseline_time = results[0].total_time_s
    steering_time = results[1].total_time_s
    steering_overhead = steering_time - baseline_time
    print(f"  Baseline inference time: {baseline_time*1000:.1f} ms")
    print(f"  With steering: {steering_time*1000:.1f} ms")
    print(f"  Steering overhead: {steering_overhead*1000:.1f} ms ({steering_overhead/baseline_time*100:.1f}% of baseline)")
    print(f"  → Steering is {steering_overhead/steering_time*100:.1f}% of total steered inference time")
    print(f"\n  Note: 6x micro-benchmark speedup → ~2% e2e improvement")
    print(f"        because steering ops are a small fraction of inference.")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())
