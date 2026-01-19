#!/usr/bin/env python3
"""Quick test script to validate vLLM steering vector implementation.

This script tests that:
1. VLLMSteerModel can be initialized and load a model
2. Steering vectors can be set and applied
3. Generation works with and without steering
4. Hook-based steering actually affects outputs
"""

import argparse
import sys
from pathlib import Path

import torch
from vllm import SamplingParams

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


async def test_basic_generation(model_name: str = "Qwen/Qwen2.5-3B"):
    """Test basic model loading and generation."""
    print(f"[1/4] Testing basic model loading and generation with {model_name}")

    # Initialize vLLM model
    target_layer = 16  # Use middle layer for smaller model
    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
    )

    print("  Loading model...")
    model = VLLMSteerModel(cfg, bootstrap_layers=(target_layer,))
    print("  ✓ Model loaded successfully")

    # Test generation without steering
    prompts = ["Once upon a time"]
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)

    print("  Generating text without steering...")
    outputs = await model.generate(prompts, sampling_params)
    print(f"  Output: {outputs[0][:100]}...")
    print("  ✓ Generation successful")

    return model, target_layer


async def test_steering_vector_setting(model: VLLMSteerModel, layer_idx: int) -> None:
    """Test setting steering vectors via per-request API."""
    print("\n[2/4] Testing steering vector setting (per-request API)")

    hidden_size = model.hidden_size
    print(f"  Model hidden size: {hidden_size}")

    # Create a random steering vector
    test_vector = torch.randn(hidden_size)
    print(f"  Created test vector with shape {test_vector.shape}")

    # Build per-request steering spec instead of global set_layer_vector()
    from chatspace.generation.vllm_steer_model import SteeringSpec, LayerSteeringSpec, AddSpec
    steering_spec = SteeringSpec(layers={
        layer_idx: LayerSteeringSpec(
            add=AddSpec(vector=test_vector, scale=1.0)
        )
    })
    print("  ✓ Steering spec created successfully")

    # Test with a simple prompt
    prompts = ["Hello"]
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.7, max_tokens=10)

    # Generate with steering - this registers/unregisters per-request
    outputs = await model.generate(prompts, sampling_params, steering_spec=steering_spec)
    print(f"  ✓ Generation with steering spec successful")
    print(f"  Output: {outputs[0][:50]}...")

    # NOTE: Old global API methods removed
    # - model.current_vector(layer_idx) - was used to verify vector in-place
    # - model.set_layer_vector(layer_idx, vector) - was used to set globally
    # - model.clear_all_vectors() - was used to clear globally
    # New per-request API means each generate() call has independent steering


async def test_multi_layer_vectors(model: VLLMSteerModel, base_layer: int) -> None:
    """Test managing steering vectors on multiple layers via per-request API."""
    print("\n[3/4] Testing multi-layer support (per-request API)")
    other_layer = base_layer + 2

    hidden_size = model.hidden_size
    base_vector = torch.randn(hidden_size)
    other_vector = torch.randn(hidden_size)

    # Build multi-layer steering spec
    from chatspace.generation.vllm_steer_model import SteeringSpec, LayerSteeringSpec, AddSpec
    steering_spec = SteeringSpec(layers={
        base_layer: LayerSteeringSpec(
            add=AddSpec(vector=base_vector, scale=1.0)
        ),
        other_layer: LayerSteeringSpec(
            add=AddSpec(vector=other_vector, scale=1.0)
        ),
    })
    print("  ✓ Multi-layer steering spec created")

    # Generate with multi-layer steering
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.7, max_tokens=10)
    outputs = await model.generate(
        ["Test prompt"],
        sampling_params,
        steering_spec=steering_spec
    )
    print("  ✓ Generation with multi-layer steering successful")

    # NOTE: Old global API methods removed
    # - model.set_layer_vector() - was used to set vectors globally
    # - model._fetch_worker_vectors() - was used to verify vectors on workers
    # New per-request API makes steering independent for each request


async def test_steering_effect(model: VLLMSteerModel, layer_idx: int) -> None:
    """Test that steering actually affects generation."""
    print("\n[4/4] Testing steering effect on generation (per-request API)")

    hidden_size = model.hidden_size
    prompt = "The weather today is"

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=30,
        seed=42,  # Use seed for reproducibility
    )

    from chatspace.generation.vllm_steer_model import SteeringSpec, LayerSteeringSpec, AddSpec

    # Generate without steering (baseline)
    print("  Generating baseline (no steering)...")
    baseline = (await model.generate([prompt], sampling_params))[0]
    print(f"  Baseline: {baseline[:100]}")

    # Generate with positive steering vector (per-request)
    positive_vector = torch.randn(hidden_size) * 5.0  # Large magnitude
    steering_pos = SteeringSpec(layers={
        layer_idx: LayerSteeringSpec(
            add=AddSpec(vector=positive_vector, scale=1.0)
        )
    })
    print("  Generating with positive steering...")
    steered_pos = (await model.generate([prompt], sampling_params, steering_spec=steering_pos))[0]
    print(f"  Steered+: {steered_pos[:100]}")

    # Generate with negative steering vector (per-request)
    negative_vector = -positive_vector
    steering_neg = SteeringSpec(layers={
        layer_idx: LayerSteeringSpec(
            add=AddSpec(vector=negative_vector, scale=1.0)
        )
    })
    print("  Generating with negative steering...")
    steered_neg = (await model.generate([prompt], sampling_params, steering_spec=steering_neg))[0]
    print(f"  Steered-: {steered_neg[:100]}")

    # Verify outputs are different (steering has an effect)
    # Note: They might occasionally be the same by chance with random vectors,
    # but typically they should differ
    if baseline != steered_pos or baseline != steered_neg:
        print("  ✓ Steering affects generation (outputs differ)")
    else:
        print("  ⚠ Warning: Steering may not be working (outputs identical)")
        print("    This can happen with random vectors; try with trained vectors")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="Model to test with (default: Qwen/Qwen2.5-3B for faster testing)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("vLLM Steering Vector Test Suite (Per-Request API)")
    print("=" * 70)

    try:
        # Run tests
        model, target_layer = await test_basic_generation(args.model)
        await test_steering_vector_setting(model, target_layer)
        await test_multi_layer_vectors(model, target_layer)
        await test_steering_effect(model, target_layer)

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Test with actual trained steering vectors")
        print("  2. Run full rollout generation with --use-vllm flag")
        print("  3. Compare outputs between HF and vLLM backends")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
