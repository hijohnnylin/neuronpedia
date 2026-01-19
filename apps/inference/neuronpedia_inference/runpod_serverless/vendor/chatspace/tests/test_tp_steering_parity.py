"""Test that tensor parallelism produces identical steering results to single-GPU.

These tests verify that TP=2 steering produces identical logprobs to single-GPU steering
for all three steering operations (additive, projection capping, ablation). The tests
require at least 2 GPUs to run.

Expected behavior:
- Steering vectors are broadcast to all TP workers via collective_rpc
- Hidden states are full-size on all ranks after RowParallelLinear allreduce
- Each rank applies steering independently with deterministic results
- Logprobs should match exactly (within floating-point tolerance) between TP=1 and TP=2
"""

from __future__ import annotations

import torch
import pytest
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
)


async def _get_final_output(model, prompt, sampling_params, steering_spec=None):
    """Helper to get final output from async generator."""
    import uuid
    # Ensure engine is initialized before accessing model.llm
    await model._ensure_engine_initialized()

    request_id = f"test_{uuid.uuid4().hex}"

    # Register steering spec for this request if provided
    if steering_spec is not None:
        await model._register_steering_spec(request_id, steering_spec)

    try:
        final_output = None
        async for output in model.llm.generate(
            prompt, sampling_params, request_id=request_id
        ):
            final_output = output
        return final_output
    finally:
        # Clean up steering spec registration
        if steering_spec is not None:
            await model._unregister_steering_spec(request_id)



@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for TP=2")
@pytest.mark.asyncio
async def test_tp_additive_steering_matches_single_gpu():
    """Test that TP=2 additive steering produces same logprobs as single GPU."""
    torch.manual_seed(42)
    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 8
    prompt = "The capital of France is"

    # Generate steering vector
    cfg_single = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        dtype="float32",
    )
    model_single = VLLMSteerModel(cfg_single, enforce_eager=True, bootstrap_layers=(target_layer,))
    hidden_size = model_single.hidden_size
    steering_vec = torch.randn(hidden_size, dtype=torch.float32) * 100.0

    # Single GPU baseline and steered
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=10)

    baseline_single = await _get_final_output(model_single, prompt, sampling)
    baseline_logprobs_single = {
        tok: data.logprob
        for tok, data in baseline_single.outputs[0].logprobs[0].items()
    }

    # Build steering spec with normalized vector
    vec_unit = steering_vec / steering_vec.norm()
    vec_scale = steering_vec.norm().item()
    steering_spec_single = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=vec_unit, scale=vec_scale)
        ])
    })
    steered_single = await _get_final_output(
        model_single, prompt, sampling, steering_spec=steering_spec_single
    )
    steered_logprobs_single = {
        tok: data.logprob
        for tok, data in steered_single.outputs[0].logprobs[0].items()
    }

    del model_single
    torch.cuda.empty_cache()

    # TP=2 baseline and steered
    cfg_tp = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        dtype="float32",
    )
    model_tp = VLLMSteerModel(cfg_tp, enforce_eager=True, bootstrap_layers=(target_layer,))

    baseline_tp = await _get_final_output(model_tp, prompt, sampling)
    baseline_logprobs_tp = {
        tok: data.logprob
        for tok, data in baseline_tp.outputs[0].logprobs[0].items()
    }

    # Reuse same steering spec (vectors already normalized and scaled)
    steered_tp = await _get_final_output(
        model_tp, prompt, sampling, steering_spec=steering_spec_single
    )
    steered_logprobs_tp = {
        tok: data.logprob
        for tok, data in steered_tp.outputs[0].logprobs[0].items()
    }

    del model_tp
    torch.cuda.empty_cache()

    # Compare baselines (should match exactly)
    common_baseline_tokens = set(baseline_logprobs_single.keys()) & set(baseline_logprobs_tp.keys())
    assert len(common_baseline_tokens) >= 5, "Need at least 5 common tokens for comparison"

    for tok in common_baseline_tokens:
        assert abs(baseline_logprobs_single[tok] - baseline_logprobs_tp[tok]) < 1e-4, (
            f"Baseline logprobs differ for token {tok}: "
            f"single={baseline_logprobs_single[tok]:.6f}, tp={baseline_logprobs_tp[tok]:.6f}"
        )

    # Compare steered (should match exactly)
    common_steered_tokens = set(steered_logprobs_single.keys()) & set(steered_logprobs_tp.keys())
    assert len(common_steered_tokens) >= 5, "Need at least 5 common tokens for comparison"

    for tok in common_steered_tokens:
        assert abs(steered_logprobs_single[tok] - steered_logprobs_tp[tok]) < 1e-4, (
            f"Steered logprobs differ for token {tok}: "
            f"single={steered_logprobs_single[tok]:.6f}, tp={steered_logprobs_tp[tok]:.6f}"
        )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for TP=2")
@pytest.mark.asyncio
async def test_tp_projection_cap_matches_single_gpu():
    """Test that TP=2 projection capping produces same logprobs as single GPU."""
    torch.manual_seed(43)
    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 8
    prompt = "In a distant galaxy, the empire"

    # Generate direction and caps
    cfg_single = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        dtype="float32",
    )
    model_single = VLLMSteerModel(cfg_single, enforce_eager=True, bootstrap_layers=(target_layer,))
    hidden_size = model_single.hidden_size

    direction = torch.randn(hidden_size, dtype=torch.float32)
    min_val, max_val = -2.0, 3.0

    # Single GPU with projection cap
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=10)

    # Build steering spec with normalized direction
    dir_unit = direction / direction.norm()
    steering_spec_single = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            ProjectionCapSpec(
                vector=dir_unit,
                min=min_val,
                max=max_val,
            )
        ])
    })
    capped_single = await _get_final_output(
        model_single, prompt, sampling, steering_spec=steering_spec_single
    )
    capped_logprobs_single = {
        tok: data.logprob
        for tok, data in capped_single.outputs[0].logprobs[0].items()
    }

    del model_single
    torch.cuda.empty_cache()

    # TP=2 with projection cap
    cfg_tp = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        dtype="float32",
    )
    model_tp = VLLMSteerModel(cfg_tp, enforce_eager=True, bootstrap_layers=(target_layer,))

    # Reuse same steering spec (vector already normalized)
    capped_tp = await _get_final_output(
        model_tp, prompt, sampling, steering_spec=steering_spec_single
    )
    capped_logprobs_tp = {
        tok: data.logprob
        for tok, data in capped_tp.outputs[0].logprobs[0].items()
    }

    del model_tp
    torch.cuda.empty_cache()

    # Compare capped logprobs
    common_tokens = set(capped_logprobs_single.keys()) & set(capped_logprobs_tp.keys())
    assert len(common_tokens) >= 5, "Need at least 5 common tokens for comparison"

    for tok in common_tokens:
        assert abs(capped_logprobs_single[tok] - capped_logprobs_tp[tok]) < 1e-4, (
            f"Capped logprobs differ for token {tok}: "
            f"single={capped_logprobs_single[tok]:.6f}, tp={capped_logprobs_tp[tok]:.6f}"
        )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for TP=2")
@pytest.mark.asyncio
async def test_tp_ablation_matches_single_gpu():
    """Test that TP=2 ablation produces same logprobs as single GPU."""
    torch.manual_seed(44)
    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 8
    prompt = "Once upon a time in a land far away"

    # Generate direction and scale
    cfg_single = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        dtype="float32",
    )
    model_single = VLLMSteerModel(cfg_single, enforce_eager=True, bootstrap_layers=(target_layer,))
    hidden_size = model_single.hidden_size

    direction = torch.randn(hidden_size, dtype=torch.float32)
    scale = 0.3

    # Single GPU with ablation
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=10)

    # Build steering spec with normalized direction
    dir_unit = direction / direction.norm()
    steering_spec_single = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AblationSpec(
                vector=dir_unit,
                scale=scale,
            )
        ])
    })
    ablated_single = await _get_final_output(
        model_single, prompt, sampling, steering_spec=steering_spec_single
    )
    ablated_logprobs_single = {
        tok: data.logprob
        for tok, data in ablated_single.outputs[0].logprobs[0].items()
    }

    del model_single
    torch.cuda.empty_cache()

    # TP=2 with ablation
    cfg_tp = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        dtype="float32",
    )
    model_tp = VLLMSteerModel(cfg_tp, enforce_eager=True, bootstrap_layers=(target_layer,))

    # Reuse same steering spec (vector already normalized)
    ablated_tp = await _get_final_output(
        model_tp, prompt, sampling, steering_spec=steering_spec_single
    )
    ablated_logprobs_tp = {
        tok: data.logprob
        for tok, data in ablated_tp.outputs[0].logprobs[0].items()
    }

    del model_tp
    torch.cuda.empty_cache()

    # Compare ablated logprobs
    common_tokens = set(ablated_logprobs_single.keys()) & set(ablated_logprobs_tp.keys())
    assert len(common_tokens) >= 5, "Need at least 5 common tokens for comparison"

    for tok in common_tokens:
        assert abs(ablated_logprobs_single[tok] - ablated_logprobs_tp[tok]) < 1e-4, (
            f"Ablated logprobs differ for token {tok}: "
            f"single={ablated_logprobs_single[tok]:.6f}, tp={ablated_logprobs_tp[tok]:.6f}"
        )
