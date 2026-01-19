"""CUDA-dependent smoke tests for vLLM steering with Gemma models."""

from __future__ import annotations

import asyncio

import os

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import (
    VLLMSteerModel,
    VLLMSteeringConfig,
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
)
from chatspace.steering.model import TransformerSteerModel, SteeringVectorConfig
from chatspace.vllm_steering import runtime as steering_runtime


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    """Normalize a vector to unit length."""
    norm = torch.norm(vector)
    if float(norm) <= 0:
        raise ValueError("Vector norm must be positive.")
    return vector / norm


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "google/gemma-2b-it",  # Gemma 1 (Gemma2 requires flash attention with softcapping)
])
@pytest.mark.asyncio
async def test_gemma_vllm_steering_vector_round_trip(model_name: str):
    """Test basic steering vector operations with Gemma models."""

    # Use smaller memory for 2B model
    gpu_mem = 0.2

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
    )

    target_layer = 10  # Middle layer for Gemma

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    hidden_size = model.hidden_size

    # Test 1: Additive steering vector
    vector = torch.randn(hidden_size, dtype=torch.float32)
    unit = _normalize(vector)
    scale = float(torch.norm(vector).item())

    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=unit, scale=scale)
        ])
    })

    # Run generation with steering
    prompt = "Test"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    await model.generate([prompt], sampling_params=sampling_params, steering_spec=steering_spec, use_tqdm=False)

    # NOTE: inspect_layer_vector() was removed with global API
    # Per-request steering means we can't inspect worker state - each request is independent
    # Gemma patching is validated through successful generation instead

    # Test 2: Projection cap only
    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    steering_spec_cap = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            ProjectionCapSpec(
                vector=_normalize(cap_vector),
                min=-0.5,
                max=0.75
            )
        ])
    })
    await model.generate([prompt], sampling_params=sampling_params, steering_spec=steering_spec_cap, use_tqdm=False)

    # Test 3: Ablation only
    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)
    steering_spec_ablation = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AblationSpec(
                vector=_normalize(ablation_vector),
                scale=0.4
            )
        ])
    })
    await model.generate([prompt], sampling_params=sampling_params, steering_spec=steering_spec_ablation, use_tqdm=False)

    # Test 4: Multi-layer steering
    layer_a, layer_b = target_layer - 1, target_layer + 1
    vector_a = torch.randn(hidden_size, dtype=torch.float32)
    vector_b = torch.randn(hidden_size, dtype=torch.float32)

    steering_spec_multi = SteeringSpec(layers={
        layer_a: LayerSteeringSpec(operations=[AddSpec(vector=_normalize(vector_a), scale=float(torch.norm(vector_a).item()))]),
        layer_b: LayerSteeringSpec(operations=[AddSpec(vector=_normalize(vector_b), scale=float(torch.norm(vector_b).item()))]),
    })
    await model.generate([prompt], sampling_params=sampling_params, steering_spec=steering_spec_multi, use_tqdm=False)

    del model
    torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skip(reason="vLLM V1 engine doesn't populate logprobs for Gemma models")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "google/gemma-2b-it",
])
@pytest.mark.asyncio
async def test_gemma_vllm_chat_respects_steering(model_name: str):
    """Verify chat generation is perturbed by steering vectors."""
    torch.manual_seed(42)
    gpu_mem = 0.2

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
    )

    target_layer = 10

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    prompt = "The capital of France is"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    # Baseline generation (no steering)
    baseline_spec = SteeringSpec(layers={})
    baseline_result = (await model.generate([prompt], sampling_params, steering_spec=baseline_spec, raw_output=True, use_tqdm=False))[0]
    baseline_logprobs = {
        tok: data.logprob
        for tok, data in baseline_result.outputs[0].logprobs[0].items()
    }

    # Apply strong steering vector
    steering_vec = torch.randn(model.hidden_size, dtype=torch.float32) * 100.0
    unit = _normalize(steering_vec)
    scale = float(torch.norm(steering_vec).item())

    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=unit, scale=scale)
        ])
    })

    # Steered generation
    steered_result = (await model.generate([prompt], sampling_params, steering_spec=steering_spec, raw_output=True, use_tqdm=False))[0]
    steered_logprobs = {
        tok: data.logprob
        for tok, data in steered_result.outputs[0].logprobs[0].items()
    }

    # Verify that logprobs changed significantly for at least some tokens
    common_tokens = set(baseline_logprobs.keys()) & set(steered_logprobs.keys())
    assert len(common_tokens) >= 3, "Need at least 3 common tokens for comparison."

    logprob_diffs = [
        abs(baseline_logprobs[tok] - steered_logprobs[tok])
        for tok in common_tokens
    ]
    max_diff = max(logprob_diffs)
    assert max_diff > 0.5, f"Expected significant logprob change, got max diff {max_diff:.3f}"

    # Verify cleared steering gives baseline results by using empty steering spec
    cleared_spec = SteeringSpec(layers={})
    cleared_result = (await model.generate([prompt], sampling_params, steering_spec=cleared_spec, raw_output=True, use_tqdm=False))[0]
    cleared_logprobs = {
        tok: data.logprob
        for tok, data in cleared_result.outputs[0].logprobs[0].items()
    }

    # After clearing, logprobs should be close to baseline
    common_cleared = set(baseline_logprobs.keys()) & set(cleared_logprobs.keys())
    for tok in common_cleared:
        diff = abs(baseline_logprobs[tok] - cleared_logprobs[tok])
        assert diff < 0.01, f"Expected cleared logprobs to match baseline for token {tok}, got diff {diff:.3f}"

    del model
    torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "google/gemma-2b-it",
])
@pytest.mark.asyncio
async def test_gemma_hidden_state_capture(model_name: str):
    """Test that we can capture hidden states before/after steering."""
    gpu_mem = 0.2

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=128,
    )

    target_layer = 10

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Generate with capture (no steering)
    prompt = "Hello world"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
    baseline_spec = SteeringSpec(layers={})
    texts, handles = await model.generate([prompt], sampling_params=sampling_params, steering_spec=baseline_spec, use_tqdm=False, capture_layers=[target_layer])

    # Fetch captured states
    await asyncio.gather(*[h.fetch() for h in handles])
    assert len(handles) > 0, "Expected at least one request"
    assert target_layer in handles[0].captures, f"Expected captures for layer {target_layer}"

    layer_captures = handles[0].captures[target_layer]
    assert len(layer_captures) > 0, "Expected at least one capture"

    # Check structure (new API has simpler structure)
    first_capture = layer_captures[0]
    assert "hidden" in first_capture, "Expected 'hidden' key in capture"

    # Check shapes
    hidden = first_capture["hidden"]
    assert isinstance(hidden, torch.Tensor)
    assert hidden.size(-1) == model.hidden_size

    # Also test with steering applied
    steering_vec = torch.randn(model.hidden_size, dtype=torch.float32)
    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(
                vector=_normalize(steering_vec),
                scale=float(torch.norm(steering_vec).item())
            )
        ])
    })
    texts_steered, handles_steered = await model.generate(
        [prompt],
        sampling_params=sampling_params,
        steering_spec=steering_spec,
        use_tqdm=False,
        capture_layers=[target_layer]
    )

    # Fetch steered captures
    await asyncio.gather(*[h.fetch() for h in handles_steered])
    assert len(handles_steered) > 0, "Expected at least one steered request"
    assert target_layer in handles_steered[0].captures, f"Expected captures for layer {target_layer}"

    del model
    torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skip(reason="vLLM V1 engine doesn't populate logprobs for Gemma models")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "google/gemma-2b-it",
])
@pytest.mark.asyncio
async def test_gemma_vllm_matches_hf_logprob_shift(model_name: str):
    """Verify that vLLM steering produces similar logprob shifts as HuggingFace baseline."""
    torch.manual_seed(43)
    gpu_mem = 0.2
    target_layer = 10

    # vLLM setup
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=128,
    )

    try:
        vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    hidden_size = vllm_model.hidden_size
    steering_vec = torch.randn(hidden_size, dtype=torch.float32) * 20.0

    prompt = "The capital of Germany is"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    # vLLM baseline (no steering)
    baseline_spec = SteeringSpec(layers={})
    vllm_baseline = (await vllm_model.generate([prompt], sampling_params, steering_spec=baseline_spec, raw_output=True, use_tqdm=False))[0]
    vllm_baseline_logprobs = {
        tok: data.logprob
        for tok, data in vllm_baseline.outputs[0].logprobs[0].items()
    }

    # vLLM steered
    unit = _normalize(steering_vec)
    scale = float(torch.norm(steering_vec).item())

    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=unit, scale=scale)
        ])
    })

    vllm_steered = (await vllm_model.generate([prompt], sampling_params, steering_spec=steering_spec, raw_output=True, use_tqdm=False))[0]
    vllm_steered_logprobs = {
        tok: data.logprob
        for tok, data in vllm_steered.outputs[0].logprobs[0].items()
    }

    del vllm_model
    torch.cuda.empty_cache()

    # HuggingFace setup
    hf_cfg = SteeringVectorConfig(model_name=model_name, target_layer=target_layer, init_scale=0.0)

    try:
        hf_model = TransformerSteerModel(
            hf_cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    except OSError as exc:
        pytest.skip(f"Unable to load HF model ({exc}). Ensure weights are cached.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # HF baseline
    inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.model.device)
    with torch.no_grad():
        hf_baseline_output = hf_model(**inputs)
    hf_baseline_logits = hf_baseline_output.logits[0, -1, :]

    # HF steered - convert normalized vector back to original scale
    scaled_steering_vec = steering_vec.to(
        device=hf_model.steering.vector.device,
        dtype=hf_model.steering.vector.dtype,
    )

    with torch.no_grad():
        hf_model.steering.vector.data = scaled_steering_vec

    with torch.no_grad():
        hf_steered_output = hf_model(**inputs)
    hf_steered_logits = hf_steered_output.logits[0, -1, :]

    # Compare logit shifts
    hf_logit_shift = hf_steered_logits - hf_baseline_logits

    # Get vLLM logprob shifts for the same tokens
    common_tokens = set(vllm_baseline_logprobs.keys()) & set(vllm_steered_logprobs.keys())
    assert len(common_tokens) >= 3, "Need at least 3 common tokens for comparison."

    # Convert logprobs to logit shifts
    for tok_id in list(common_tokens)[:5]:  # Check top 5 tokens
        vllm_shift = vllm_steered_logprobs[tok_id] - vllm_baseline_logprobs[tok_id]
        hf_shift = hf_logit_shift[tok_id].item()

        # Shifts should have same direction (sign)
        assert (vllm_shift * hf_shift) >= 0, (
            f"Token {tok_id} shift direction mismatch: vLLM={vllm_shift:.3f}, HF={hf_shift:.3f}"
        )

        # Magnitudes should be comparable (within 50% relative difference)
        if abs(vllm_shift) > 0.1:  # Only check substantial shifts
            relative_diff = abs(vllm_shift - hf_shift) / abs(vllm_shift)
            assert relative_diff < 0.5, (
                f"Token {tok_id} shift magnitude differs: vLLM={vllm_shift:.3f}, HF={hf_shift:.3f}"
            )

    del hf_model
    torch.cuda.empty_cache()
