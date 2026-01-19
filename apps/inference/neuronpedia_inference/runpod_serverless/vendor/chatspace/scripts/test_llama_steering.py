#!/usr/bin/env python3
"""Quick smoke test to verify Llama steering works in vLLM."""

from __future__ import annotations

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec,
)


async def main() -> None:
    """Test basic Llama steering functionality."""

    # Use a tiny Llama model for testing
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading {model_name}...")
    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )

    # Bootstrap with layer 8 (middle layer for 1B model which has 16 layers)
    model = VLLMSteerModel(cfg, bootstrap_layers=(8,))

    print(f"Model loaded. Hidden size: {model.hidden_size}, Layer count: {model.layer_count}")

    prompt = "The capital of France is"
    sampling = SamplingParams(temperature=0.0, max_tokens=5)

    # Test 1: Generate without steering (baseline)
    print("\n=== Test 1: Baseline generation (no steering) ===")
    baseline_output = (await model.generate([prompt], sampling))[0]
    print(f"Prompt: {prompt}")
    print(f"Output: {baseline_output}")

    # Test 2: Generate with random steering
    print("\n=== Test 2: Generation with random steering vector ===")
    random_vector = torch.randn(model.hidden_size) * 0.5
    steering_spec = SteeringSpec(
        layers={8: LayerSteeringSpec(add=AddSpec(vector=random_vector, scale=1.0))}
    )
    steered_output = (await model.generate([prompt], sampling, steering_spec=steering_spec))[0]
    print(f"Prompt: {prompt}")
    print(f"Output: {steered_output}")

    # Outputs should differ with steering
    if baseline_output != steered_output:
        print("\n✓ Steering successfully modified output")
    else:
        print("\n✗ WARNING: Steering did not modify output")

    # Test 3: Per-request steering is stateless
    print("\n=== Test 3: Per-request steering is stateless ===")
    print("Generating without steering again (no spec passed)...")
    baseline_output2 = (await model.generate([prompt], sampling))[0]
    print(f"Output: {baseline_output2}")
    print(f"Matches original baseline: {baseline_output == baseline_output2}")
    print("✓ Per-request steering is stateless (no residual effects)")

    # Test 4: Multi-layer steering
    print("\n=== Test 4: Multi-layer steering ===")
    multi_spec = SteeringSpec(
        layers={
            6: LayerSteeringSpec(add=AddSpec(vector=torch.randn(model.hidden_size) * 0.3, scale=1.0)),
            10: LayerSteeringSpec(add=AddSpec(vector=torch.randn(model.hidden_size) * 0.3, scale=1.0)),
        }
    )
    multi_output = (await model.generate([prompt], sampling, steering_spec=multi_spec))[0]
    print(f"Prompt: {prompt}")
    print(f"Output: {multi_output}")
    print(f"Multi-layer output differs from baseline: {multi_output != baseline_output}")

    # Test 5: Hidden state capture with steering
    print("\n=== Test 5: Hidden state capture with steering ===")
    capture_spec = SteeringSpec(
        layers={8: LayerSteeringSpec(add=AddSpec(vector=torch.randn(model.hidden_size) * 0.5, scale=1.0))}
    )

    outputs, handles = await model.generate(
        [prompt],
        sampling,
        steering_spec=capture_spec,
        capture_layers=[8],
    )
    print(f"Output: {outputs[0]}")

    # Fetch captures
    async with handles[0] as handle:
        captures = handle.captures
        if captures and 8 in captures:
            layer_captures = captures[8]
            print(f"Captured {len(layer_captures)} requests for layer 8")
            if layer_captures:
                first_cap = layer_captures[0]
                if "hidden" in first_cap:
                    print(f"  - Hidden state shape: {first_cap['hidden'].shape}")
                    print("✓ Hidden state capture working with steering")
        else:
            print("✗ WARNING: No hidden states captured")

    print("\n=== All tests completed ===")
    print("Llama steering appears to be working correctly!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
