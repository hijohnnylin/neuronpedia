"""CUDA-dependent smoke tests for vLLM steering backend."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import (
    AddSpec,
    AblationSpec,
    ChatResponse,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
    VLLMSteerModel,
    VLLMSteeringConfig,
)
from chatspace.steering.model import QwenSteerModel, SteeringVectorConfig
from chatspace.vllm_steering import runtime as steering_runtime


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_steering_vector_round_trip():

    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    hidden_size = model.hidden_size

    # Test 1: Basic additive steering with per-request API
    vector = torch.randn(hidden_size, dtype=torch.float32)
    vector_norm = vector.norm().item()
    unit_vector = vector / (vector.norm() + 1e-8)

    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=unit_vector, scale=vector_norm)
        ])
    })

    # Generate with steering to verify it's applied per-request
    prompt = "Test prompt"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    texts = await model.generate([prompt], sampling_params=sampling, steering_spec=steering_spec)
    assert texts, "Expected text output from steered generation."

    # Test 2: Projection cap and ablation
    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    cap_unit = cap_vector / (cap_vector.norm() + 1e-8)

    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)
    ablation_unit = ablation_vector / (ablation_vector.norm() + 1e-8)

    steering_spec_full = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=unit_vector, scale=vector_norm),
            ProjectionCapSpec(vector=cap_unit, min=-0.5, max=0.75),
            AblationSpec(vector=ablation_unit, scale=0.4),
        ])
    })

    # NOTE: inspect_layer_vector() and fetch_worker_state() were removed with global API
    # Per-request steering means we can't inspect worker state - each request is independent

    # Generate with full steering spec
    texts_full = await model.generate([prompt], sampling_params=sampling, steering_spec=steering_spec_full)
    assert texts_full, "Expected text output from fully steered generation."

    # Test 3: Multi-layer steering
    # Per-request steering means each layer is configured independently
    # Use two different layers for multi-layer test
    other_layer = (target_layer + 1) if target_layer < 23 else (target_layer - 1)

    extra_vector = torch.randn(hidden_size, dtype=torch.float32)
    extra_norm = extra_vector.norm().item()
    extra_unit = extra_vector / (extra_vector.norm() + 1e-8)

    multi_layer_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[AddSpec(vector=unit_vector, scale=vector_norm)]),
        other_layer: LayerSteeringSpec(operations=[AddSpec(vector=extra_unit, scale=extra_norm)]),
    })

    texts_multi = await model.generate([prompt], sampling_params=sampling, steering_spec=multi_layer_spec)
    assert texts_multi, "Expected text output from multi-layer steered generation."

    # Test 4: Baseline generation without steering (per-request steering doesn't need cleanup)
    texts_baseline = await model.generate([prompt], sampling_params=sampling)
    assert texts_baseline, "Expected text output from baseline generation."

    # Clean up to avoid leaving GPU memory allocated between tests.
    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_chat_respects_steering():
    torch.manual_seed(0)

    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    prompt = "State the color of a clear daytime sky."
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    # Generate baseline output without steering
    baseline_req = (await model.generate([prompt], sampling_params=sampling, raw_output=True))[0]
    baseline_out = baseline_req.outputs[0]
    baseline_token = baseline_out.token_ids[0]
    baseline_logprob = baseline_out.logprobs[0][baseline_token].logprob

    # Build steering spec with scaled random vector
    scale = 5_000.0
    random_vector = torch.randn(model.hidden_size, dtype=torch.float32)
    random_norm = random_vector.norm().item()
    random_unit = random_vector / (random_vector.norm() + 1e-8)

    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=random_unit, scale=scale)
        ])
    })

    # Generate with steering
    steered_req = (await model.generate([prompt], sampling_params=sampling, raw_output=True, steering_spec=steering_spec))[0]
    steered_out = steered_req.outputs[0]
    steered_token = steered_out.token_ids[0]
    steered_logprob = steered_out.logprobs[0][steered_token].logprob

    # Generate reset output without steering (per-request steering, no explicit cleanup needed)
    reset_req = (await model.generate([prompt], sampling_params=sampling, raw_output=True))[0]
    reset_out = reset_req.outputs[0]
    reset_token = reset_out.token_ids[0]
    reset_logprob = reset_out.logprobs[0][reset_token].logprob

    # Test basic chat API without steering
    request = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_sampling = SamplingParams(temperature=0.0, max_tokens=4)
    chat_output = (await model.chat(
        request,
        sampling_params=chat_sampling,
    ))[0]

    assert not torch.isclose(
        torch.tensor(baseline_logprob), torch.tensor(steered_logprob)
    ), (
        "Steering did not perturb token logprobs. "
        f"baseline_logprob={baseline_logprob} steered_logprob={steered_logprob}"
    )
    assert torch.isclose(
        torch.tensor(baseline_logprob), torch.tensor(reset_logprob)
    ), "Clearing the steering vector should restore baseline behaviour."
    assert isinstance(chat_output, ChatResponse), "Chat should return ChatResponse."
    assert len(chat_output.full_text()) > 0, "Chat response should have non-empty text."

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_matches_hf_logprob_shift():
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 2
    prompt = "In a quiet village, the baker"

    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hf_cfg = SteeringVectorConfig(model_name=model_name, target_layer=target_layer, init_scale=0.0)
    hf_model = QwenSteerModel(
        hf_cfg,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    with torch.no_grad():
        hf_outputs_base = hf_model(**inputs)
    base_logits = hf_outputs_base.logits[:, -1, :].float()
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    baseline_token = int(torch.argmax(base_logprobs, dim=-1).item())
    hf_baseline_lp = float(base_logprobs[0, baseline_token])

    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 0.01
    steering_norm = steering_vector.norm().item()
    steering_unit = steering_vector / (steering_vector.norm() + 1e-8)

    hf_model.set_vector(steering_vector)
    with torch.no_grad():
        hf_outputs_steered = hf_model(**inputs)
    steered_logits = hf_outputs_steered.logits[:, -1, :].float()
    steered_logprobs = torch.log_softmax(steered_logits, dim=-1)
    hf_steered_lp = float(steered_logprobs[0, baseline_token])
    hf_shift = hf_steered_lp - hf_baseline_lp

    hf_model.set_vector(None)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )

    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))

    def _extract_logprob(output, token_id):
        entry = output.logprobs[0].get(int(token_id))
        if entry is None:
            raise AssertionError(f"Token id {token_id} not present in returned logprobs")
        return float(entry.logprob)

    baseline_out = (await vllm_model.generate([prompt], sampling_params=sampling, raw_output=True))[0].outputs[0]
    vllm_baseline_lp = _extract_logprob(baseline_out, baseline_token)

    # Build steering spec for vLLM
    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=steering_unit, scale=steering_norm)
        ])
    })

    steered_out = (await vllm_model.generate([prompt], sampling_params=sampling, raw_output=True, steering_spec=steering_spec))[0].outputs[0]
    vllm_steered_lp = _extract_logprob(steered_out, baseline_token)
    vllm_shift = vllm_steered_lp - vllm_baseline_lp

    baseline_delta = abs(hf_baseline_lp - vllm_baseline_lp)
    assert baseline_delta <= 2e-2, (
        f"Baseline logprob mismatch {baseline_delta:.4f} exceeds tolerance 0.02"
    )

    shift_delta = abs(hf_shift - vllm_shift)
    shift_tol = max(1e-3, baseline_delta + 5e-4)
    assert shift_delta <= shift_tol, (
        f"Steering logprob shift mismatch {shift_delta:.4f} exceeds tolerance {shift_tol:.4f}"
    )

    # No cleanup needed - per-request steering is automatic
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()
