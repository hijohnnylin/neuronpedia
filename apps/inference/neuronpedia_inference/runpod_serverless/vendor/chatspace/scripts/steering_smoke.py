"""Quick smoke test to confirm steering vectors affect vLLM outputs.

This script runs a baseline decode, applies a large random steering vector to
demonstrate behavior change, then generates without steering to restore baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import torch

from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec,
)
from vllm import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Question: What is the capital of France? Answer:")
    parser.add_argument("--scale", type=float, default=5000.0)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    vllm_kwargs = {}
    if args.enforce_eager:
        vllm_kwargs["enforce_eager"] = True

    target_layer = args.layer
    model = VLLMSteerModel(cfg, bootstrap_layers=(target_layer,), **vllm_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Baseline (no steering)
    baseline = (await model.generate([args.prompt], params))[0]
    print("=== Baseline ===")
    print(repr(baseline))

    # Build steering spec with large random perturbation
    perturb = torch.randn(model.hidden_size) * args.scale
    perturb_norm = float(perturb.norm().item())
    perturb_unit = perturb / perturb_norm

    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(
            add=AddSpec(vector=perturb_unit, scale=perturb_norm)
        )
    })

    print(f"\nApplying steering vector norm: {perturb_norm:.2f}")

    # Generate with steering
    steered = (await model.generate(
        [args.prompt],
        params,
        steering_spec=steering_spec,
    ))[0]
    print("\n=== Steered ===")
    print(repr(steered))

    # Restored (no steering spec = baseline behavior)
    restored = (await model.generate([args.prompt], params))[0]
    print("\n=== Restored ===")
    print(repr(restored))

    # Verify baseline and restored match
    if baseline == restored:
        print("\n✓ Restored output matches baseline (steering is per-request)")
    else:
        print("\n⚠ Warning: Restored output differs from baseline")


if __name__ == "__main__":
    asyncio.run(main())
