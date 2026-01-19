#!/usr/bin/env python3
"""Quick test to verify Gemma decoder layer patching works."""

import os
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec,
    AblationSpec,
)
from chatspace.vllm_steering import runtime as steering_runtime

async def main():
    """Test Gemma patching."""
    model_name = "google/gemma-3-27b-it"  # Gemma3 27B on H200

    print(f"Testing Gemma patching with {model_name}")

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,  # More memory for 27B model on H200
        max_model_len=256,
    )

    target_layer = 23  # Middle layer for 27B model (likely ~46 layers)

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return

    print(f"Model loaded. Hidden size: {model.hidden_size}")

    # Create steering spec with additive vector
    steering_vec = torch.randn(model.hidden_size, dtype=torch.float32) * 10.0
    unit_vector = steering_vec / (steering_vec.norm() + 1e-8)
    vector_norm = steering_vec.norm().item()
    print(f"Set steering vector on layer {target_layer} with norm {vector_norm:.4f}")

    # Create ablation spec with normalized vector
    ablation_vec = torch.randn(model.hidden_size, dtype=torch.float32)
    ablation_unit = ablation_vec / (ablation_vec.norm() + 1e-8)
    print(f"Set ablation on layer {target_layer} with scale 0.5")

    # Build steering spec with both add and ablation
    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(
            add=AddSpec(vector=unit_vector, scale=vector_norm),
            ablation=AblationSpec(vector=ablation_unit, scale=0.5),
        )
    })

    # Run a generation to trigger patching
    prompt = "Hello"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    outputs = await model.generate([prompt], sampling_params=sampling, steering_spec=steering_spec)
    print(f"Generated: {outputs[0]}")

    # Now check inspection
    inspection = await model._engine_client.collective_rpc(
        steering_runtime.STEERING_RPC_METHOD,
        args=steering_runtime.rpc_args("inspect_layer_vector", target_layer),
    )

    if inspection:
        layer_info = inspection[0]
        output_types = layer_info.get("output_types", [])
        print(f"Output types after generation: {output_types}")

        if "tuple" in output_types:
            print("✓ Gemma decoder layer is using tuple output (expected)")
        else:
            print(f"⚠ Unexpected output types: {output_types}")
    else:
        print("⚠ No inspection data returned")

    del model
    torch.cuda.empty_cache()
    print("Test complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
