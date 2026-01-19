"""Parity checks for steering transforms between HuggingFace and vLLM backends."""

from __future__ import annotations

import asyncio

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
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


async def _get_final_output(model, prompt, sampling_params):
    """Helper to get final output from async generator."""
    import uuid
    # Ensure engine is initialized before accessing model.llm
    await model._ensure_engine_initialized()
    final_output = None
    request_id = f"test_{uuid.uuid4().hex}"
    async for output in model.llm.generate(prompt, sampling_params, request_id=request_id):
        final_output = output
    return final_output



@dataclass
class ProjectionCapParams:
    vector: torch.Tensor
    min: float | None
    max: float | None


@dataclass
class AblationParams:
    vector: torch.Tensor
    scale: float


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vector)
    if float(norm) <= 0:
        raise ValueError("Direction vector norm must be positive.")
    return vector / norm


def _apply_projection_cap(hidden: torch.Tensor, unit: torch.Tensor, *, minimum: float | None, maximum: float | None) -> torch.Tensor:
    if minimum is None and maximum is None:
        return hidden
    flat = hidden.reshape(-1, hidden.shape[-1])
    projection = flat @ unit
    clamp_kwargs: dict[str, Any] = {}
    if minimum is not None:
        clamp_kwargs["min"] = projection.new_tensor(float(minimum))
    if maximum is not None:
        clamp_kwargs["max"] = projection.new_tensor(float(maximum))
    clamped = torch.clamp(projection, **clamp_kwargs)  # type: ignore[arg-type]
    if clamped is projection:
        return hidden
    delta = (clamped - projection).unsqueeze(-1) * unit
    return (flat + delta).reshape_as(hidden)


def _apply_ablation(hidden: torch.Tensor, unit: torch.Tensor, *, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return hidden
    flat = hidden.reshape(-1, hidden.shape[-1])
    projection = flat @ unit
    component = projection.unsqueeze(-1) * unit
    return (flat + (scale - 1.0) * component).reshape_as(hidden)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_hf_steering_combinations_match():
    torch.manual_seed(123)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5  # Check that steering propagates to subsequent layers
    prompt = "A curious squirrel contemplates the mysteries of the forest."

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size
    vector = torch.randn(hidden_size, dtype=torch.float32) * 0.05
    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)

    projection_spec = ProjectionCapParams(vector=cap_vector, min=-0.2, max=0.2)
    ablation_spec = AblationParams(vector=ablation_vector, scale=0.6)

    cases: list[dict[str, Any]] = [
        {"name": "vector_only", "vector": vector, "cap": None, "ablation": None},
        {"name": "projection_cap_only", "vector": None, "cap": projection_spec, "ablation": None},
        {"name": "ablation_only", "vector": None, "cap": None, "ablation": ablation_spec},
        {"name": "vector_and_cap", "vector": vector, "cap": projection_spec, "ablation": None},
        {"name": "vector_cap_ablation", "vector": vector, "cap": projection_spec, "ablation": ablation_spec},
    ]

    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        for case in cases:
            # ------------------------------------------------------------------
            # HuggingFace path
            # ------------------------------------------------------------------
            captured_hf_hiddens: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int):
                def hook(module, _args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    orig_dtype = hidden.dtype
                    hidden_fp32 = hidden.to(torch.float32)

                    # Only apply steering at target layer
                    if layer_idx == target_layer:
                        if case["vector"] is not None:
                            hidden_fp32 = hidden_fp32 + case["vector"].to(device=hidden_fp32.device)

                        if case["cap"] is not None:
                            unit = _normalize(case["cap"].vector).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                            hidden_fp32 = _apply_projection_cap(
                                hidden_fp32,
                                unit,
                                minimum=case["cap"].min,
                                maximum=case["cap"].max,
                            )

                        if case["ablation"] is not None:
                            unit = _normalize(case["ablation"].vector).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                            hidden_fp32 = _apply_ablation(hidden_fp32, unit, scale=case["ablation"].scale)

                    captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
                    hidden_out = hidden_fp32.to(dtype=orig_dtype)
                    if isinstance(output, tuple):
                        return (hidden_out,) + output[1:]
                    return hidden_out
                return hook

            # Install hooks on both layers
            target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
            downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

            with torch.no_grad():
                hf_model(**inputs)

            target_handle.remove()
            downstream_handle.remove()

            hf_target_hidden = captured_hf_hiddens[target_layer]
            hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

            # ------------------------------------------------------------------
            # vLLM path
            # ------------------------------------------------------------------
            # Build steering spec for this case (only at target layer)
            operations = []

            if case["vector"] is not None:
                # Normalize vector: unit = vec / vec.norm(), scale = vec.norm()
                norm = torch.norm(case["vector"]).item()
                unit_vec = case["vector"] / torch.norm(case["vector"])
                operations.append(AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm))

            if case["cap"] is not None:
                # Normalize cap vector
                cap_unit = _normalize(case["cap"].vector).to(dtype=torch.float32)
                operations.append(ProjectionCapSpec(
                    vector=cap_unit,
                    min=case["cap"].min,
                    max=case["cap"].max,
                ))

            if case["ablation"] is not None:
                # Normalize ablation vector
                ablation_unit = _normalize(case["ablation"].vector).to(dtype=torch.float32)
                operations.append(AblationSpec(
                    vector=ablation_unit,
                    scale=case["ablation"].scale,
                ))

            layer_spec = LayerSteeringSpec(operations=operations)
            steering_spec = SteeringSpec(layers={target_layer: layer_spec})

            texts, handles = await vllm_model.generate(
                [prompt],
                sampling_params=sampling,
                steering_spec=steering_spec,
                capture_layers=[target_layer, downstream_layer]
            )
            await asyncio.gather(*[h.fetch() for h in handles])
            vllm_target_hidden = handles[0].captures[target_layer][0]["hidden"].to(dtype=torch.float32)
            vllm_downstream_hidden = handles[0].captures[downstream_layer][0]["hidden"].to(dtype=torch.float32)

            # ------------------------------------------------------------------
            # Comparison
            # ------------------------------------------------------------------
            # Check target layer (where steering is applied)
            hf_target_flat = hf_target_hidden.reshape(-1)
            vllm_target_flat = vllm_target_hidden.reshape(-1)
            target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
            target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

            assert target_mae < 2e-3, f"{case['name']} (target layer): mean abs diff too large ({target_mae:.6f})"
            assert target_cos > 0.9995, f"{case['name']} (target layer): cosine similarity degraded ({target_cos:.6f})"

            # Check downstream layer (verify steering propagates)
            hf_downstream_flat = hf_downstream_hidden.reshape(-1)
            vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
            downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
            downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

            assert downstream_mae < 2e-3, f"{case['name']} (downstream layer): mean abs diff too large ({downstream_mae:.6f})"
            assert downstream_cos > 0.9995, f"{case['name']} (downstream layer): cosine similarity degraded ({downstream_cos:.6f})"
    finally:
        del vllm_model
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_decode_steering_matches_hf_prefill():
    """Test that vLLM decode-time steering matches HF prefill-time steering.

    This validates that steering applied during vLLM's autoregressive decode phase
    produces the same hidden states as HF's prefill phase with the same tokens.

    Strategy:
    1. HF: Forward pass with [prompt + generated_tokens] in prefill mode (with steering)
    2. vLLM: Generate tokens autoregressively starting from prompt (with steering)
    3. Compare hidden states at each token position

    If vLLM decode-time steering works correctly, the hidden states should match.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5  # Check that steering propagates to subsequent layers
    prompt = "The capital of France is"

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Create steering vector
    hidden_size = hf_model.config.hidden_size
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 0.1

    # -------------------------------------------------------------------------
    # Step 1: vLLM generate with steering (decode mode)
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=64,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Build steering spec for target layer
    norm = torch.norm(steering_vector).item()
    unit_vec = steering_vector / torch.norm(steering_vector)
    steering_spec = SteeringSpec(
        layers={
            target_layer: LayerSteeringSpec(operations=[
                AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
            ])
        }
    )

    # Generate tokens with steering
    sampling = SamplingParams(temperature=0.0, max_tokens=5, logprobs=0, ignore_eos=True)
    texts, handles = await vllm_model.generate(
        [prompt],
        sampling_params=sampling,
        steering_spec=steering_spec,
        capture_layers=[target_layer, downstream_layer]
    )

    # Get the generated text (outputs is list[str] of completion only)
    generated_text = texts[0]
    full_text = prompt + generated_text

    # Fetch vLLM hidden states (decode mode)
    await asyncio.gather(*[h.fetch() for h in handles])

    # New API: captures are concatenated tensors [seq_len, hidden_size]
    vllm_target_hidden_full = handles[0].captures[target_layer][0]["hidden"]  # [seq_len, hidden_size]
    vllm_downstream_hidden_full = handles[0].captures[downstream_layer][0]["hidden"]  # [seq_len, hidden_size]

    print(f"\nGenerated text: {repr(generated_text)}")
    print(f"Full text: {repr(full_text)}")
    print(f"vLLM target capture shape: {vllm_target_hidden_full.shape}")
    print(f"vLLM downstream capture shape: {vllm_downstream_hidden_full.shape}")

    # -------------------------------------------------------------------------
    # Step 2: HF forward pass with full text (prefill mode with steering)
    # -------------------------------------------------------------------------
    # Tokenize the full text (what vLLM generated)
    full_inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    prompt_len = prompt_inputs.input_ids.shape[1]
    full_len = full_inputs.input_ids.shape[1]
    generated_len = full_len - prompt_len

    # Account for autoregressive generation: vLLM captures prompt + (generated - 1) tokens
    # The final sampled token never flows through the model
    expected_capture_len = prompt_len + (generated_len - 1)

    print(f"Prompt tokens: {prompt_len}, Full tokens: {full_len}, Generated: {generated_len}")
    print(f"Expected vLLM capture length: {expected_capture_len} (prompt + generated-1)")

    assert vllm_target_hidden_full.shape[0] == expected_capture_len, (
        f"vLLM capture length {vllm_target_hidden_full.shape[0]} != expected {expected_capture_len}"
    )

    # We can compare the first (generated - 1) decode tokens
    num_decode_steps = generated_len - 1
    assert num_decode_steps >= 3, f"Need at least 3 decode steps to compare, got {num_decode_steps}"

    # Capture hidden states from HF with steering
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def capture_hook(module, args, output):
            """Hook to capture and steer hidden states from HF model."""
            hidden = output[0] if isinstance(output, tuple) else output
            orig_dtype = hidden.dtype
            hidden_fp32 = hidden.to(torch.float32)

            # Apply steering only at target layer (same vector vLLM used)
            if layer_idx == target_layer:
                hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

            # Capture the steered hidden state
            captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()

            # Return steered hidden state
            hidden_out = hidden_fp32.to(dtype=orig_dtype)
            if isinstance(output, tuple):
                return (hidden_out,) + output[1:]
            return hidden_out
        return capture_hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**full_inputs)

    target_handle.remove()
    downstream_handle.remove()

    # HF captured the full sequence: [prompt_tokens + generated_tokens]
    assert len(captured_hf_hiddens) == 2, "Should have captured both layers"
    hf_target_hidden_full = captured_hf_hiddens[target_layer].to(dtype=torch.float32).squeeze(0)  # [seq_len, hidden_size]
    hf_downstream_hidden_full = captured_hf_hiddens[downstream_layer].to(dtype=torch.float32).squeeze(0)  # [seq_len, hidden_size]

    print(f"HF target hidden shape: {hf_target_hidden_full.shape}")
    print(f"HF downstream hidden shape: {hf_downstream_hidden_full.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Compare vLLM decode-time vs HF prefill-time hidden states
    # -------------------------------------------------------------------------
    print(f"\nComparing vLLM decode (steered) vs HF prefill (steered):")

    # Convert to float32 for comparison
    vllm_target_hidden_full = vllm_target_hidden_full.to(dtype=torch.float32)
    vllm_downstream_hidden_full = vllm_downstream_hidden_full.to(dtype=torch.float32)

    for decode_idx in range(num_decode_steps):
        # Token position in the full sequence
        # vLLM: token is at position (prompt_len + decode_idx) in the concatenated captures
        # HF: same position in HF's captures
        vllm_token_idx = prompt_len + decode_idx
        hf_token_idx = prompt_len + decode_idx

        if hf_token_idx >= hf_target_hidden_full.shape[0]:
            print(f"  Decode step {decode_idx}: Skipping (out of bounds)")
            continue

        # -------------------------------------------------------------------------
        # Target layer comparison
        # -------------------------------------------------------------------------
        # vLLM: hidden state at this token position (slice the concatenated tensor)
        vllm_target_hidden = vllm_target_hidden_full[vllm_token_idx]  # [hidden_size]

        # HF: hidden state from prefill at the corresponding token position
        hf_target_hidden = hf_target_hidden_full[hf_token_idx]  # [hidden_size]

        # Compare
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)

        target_cos_sim = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()
        target_mae = torch.mean(torch.abs(hf_target_flat - vllm_target_flat)).item()

        # -------------------------------------------------------------------------
        # Downstream layer comparison (verify steering propagates)
        # -------------------------------------------------------------------------
        vllm_downstream_hidden = vllm_downstream_hidden_full[vllm_token_idx]  # [hidden_size]
        hf_downstream_hidden = hf_downstream_hidden_full[hf_token_idx]  # [hidden_size]

        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)

        downstream_cos_sim = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()
        downstream_mae = torch.mean(torch.abs(hf_downstream_flat - vllm_downstream_flat)).item()

        print(f"  Decode step {decode_idx} (token position {hf_token_idx}):")
        print(f"    Target layer:     cos={target_cos_sim:.6f}, MAE={target_mae:.6f}")
        print(f"    Downstream layer: cos={downstream_cos_sim:.6f}, MAE={downstream_mae:.6f}")

        # Assert high similarity for target layer (steering is working during decode)
        assert target_cos_sim > 0.999, (
            f"Decode step {decode_idx} (target layer): Cosine similarity {target_cos_sim:.6f} should be >0.999. "
            f"This indicates decode-time steering may not match prefill-time steering."
        )
        assert target_mae < 0.01, (
            f"Decode step {decode_idx} (target layer): MAE {target_mae:.6f} should be <0.01"
        )

        # Assert high similarity for downstream layer (steering propagates)
        assert downstream_cos_sim > 0.999, (
            f"Decode step {decode_idx} (downstream layer): Cosine similarity {downstream_cos_sim:.6f} should be >0.999. "
            f"This indicates steering may not propagate correctly through layers."
        )
        assert downstream_mae < 0.01, (
            f"Decode step {decode_idx} (downstream layer): MAE {downstream_mae:.6f} should be <0.01"
        )

    print(f"\n✓ vLLM decode-time steering matches HF prefill-time steering (target + downstream layers)")

    # Cleanup
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_hf_high_magnitude_steering():
    """Test that steering works correctly at high magnitudes (10x).

    This test validates that the fix works under aggressive steering conditions.
    High-magnitude steering can expose bugs in residual stream handling that
    might not be visible with subtle perturbations.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "The quick brown fox jumps over the lazy dog."

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # HIGH MAGNITUDE: 10.0 instead of 0.05-0.1
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 10.0

    print(f"\nSteering vector magnitude: {torch.norm(steering_vector).item():.2f}")

    # -------------------------------------------------------------------------
    # HuggingFace path
    # -------------------------------------------------------------------------
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            orig_dtype = hidden.dtype
            hidden_fp32 = hidden.to(torch.float32)

            # Apply steering only at target layer
            if layer_idx == target_layer:
                hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

            captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
            hidden_out = hidden_fp32.to(dtype=orig_dtype)
            if isinstance(output, tuple):
                return (hidden_out,) + output[1:]
            return hidden_out
        return hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**inputs)

    target_handle.remove()
    downstream_handle.remove()

    hf_target_hidden = captured_hf_hiddens[target_layer]
    hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

    # -------------------------------------------------------------------------
    # vLLM path
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Build steering spec with high-magnitude vector
    norm = torch.norm(steering_vector).item()
    unit_vec = steering_vector / torch.norm(steering_vector)
    steering_spec = SteeringSpec(
        layers={
            target_layer: LayerSteeringSpec(operations=[
                AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
            ])
        }
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        texts, handles = await vllm_model.generate(
            [prompt],
            sampling_params=sampling,
            steering_spec=steering_spec,
            capture_layers=[target_layer, downstream_layer]
        )

        await asyncio.gather(*[h.fetch() for h in handles])
        vllm_target_hidden = handles[0].captures[target_layer][0]["hidden"].to(dtype=torch.float32)
        vllm_downstream_hidden = handles[0].captures[downstream_layer][0]["hidden"].to(dtype=torch.float32)

        # -------------------------------------------------------------------------
        # Comparison
        # -------------------------------------------------------------------------
        # Target layer
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)
        target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
        target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

        # Downstream layer
        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
        downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
        downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

        # Print statistics
        hf_target_norm = torch.norm(hf_target_flat).item()
        vllm_target_norm = torch.norm(vllm_target_flat).item()
        hf_downstream_norm = torch.norm(hf_downstream_flat).item()
        vllm_downstream_norm = torch.norm(vllm_downstream_flat).item()

        print(f"\nTarget layer (where steering is applied):")
        print(f"  HF norm: {hf_target_norm:.2f}, vLLM norm: {vllm_target_norm:.2f}")
        print(f"  Cosine similarity: {target_cos:.6f}")
        print(f"  MAE: {target_mae:.6f}")

        print(f"\nDownstream layer (steering propagated):")
        print(f"  HF norm: {hf_downstream_norm:.2f}, vLLM norm: {vllm_downstream_norm:.2f}")
        print(f"  Cosine similarity: {downstream_cos:.6f}")
        print(f"  MAE: {downstream_mae:.6f}")

        # Assert high similarity even at high magnitude
        assert target_cos > 0.999, (
            f"Target layer: Cosine similarity {target_cos:.6f} should be >0.999 even at high magnitude"
        )
        assert target_mae < 0.01, (
            f"Target layer: MAE {target_mae:.6f} should be <0.01"
        )

        assert downstream_cos > 0.999, (
            f"Downstream layer: Cosine similarity {downstream_cos:.6f} should be >0.999, "
            f"indicating steering propagates correctly even at high magnitude"
        )
        assert downstream_mae < 0.01, (
            f"Downstream layer: MAE {downstream_mae:.6f} should be <0.01"
        )

        print(f"\n✓ High-magnitude steering (10x) works correctly at target and downstream layers")

    finally:
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_hf_high_magnitude_ablation_and_capping():
    """Test that ablation and projection capping work correctly at high magnitudes.

    This test validates that the dual-mode transform system (mode="hidden" for
    ablations and caps) works correctly under aggressive steering conditions.
    """
    torch.manual_seed(123)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "The capital of France is Paris, which is known for"

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # HIGH MAGNITUDE: 10.0 for vector, wide range for projection cap
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 10.0
    cap_direction = torch.randn(hidden_size, dtype=torch.float32)
    ablation_direction = torch.randn(hidden_size, dtype=torch.float32)

    # Extreme projection cap range and aggressive ablation
    projection_spec = ProjectionCapParams(vector=cap_direction, min=-50.0, max=50.0)  # Wide range
    ablation_spec = AblationParams(vector=ablation_direction, scale=0.1)  # Aggressive ablation (90% removal)

    print(f"\nSteering vector norm: {torch.norm(steering_vector).item():.2f}")
    print(f"Projection cap range: [{projection_spec.min}, {projection_spec.max}]")
    print(f"Ablation scale: {ablation_spec.scale}")

    # -------------------------------------------------------------------------
    # HuggingFace path
    # -------------------------------------------------------------------------
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            orig_dtype = hidden.dtype
            hidden_fp32 = hidden.to(torch.float32)

            # Apply all steering operations only at target layer
            if layer_idx == target_layer:
                # Vector addition
                hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

                # Projection cap
                unit_cap = _normalize(cap_direction).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                hidden_fp32 = _apply_projection_cap(
                    hidden_fp32,
                    unit_cap,
                    minimum=projection_spec.min,
                    maximum=projection_spec.max,
                )

                # Ablation
                unit_ablation = _normalize(ablation_direction).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                hidden_fp32 = _apply_ablation(hidden_fp32, unit_ablation, scale=ablation_spec.scale)

            captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
            hidden_out = hidden_fp32.to(dtype=orig_dtype)
            if isinstance(output, tuple):
                return (hidden_out,) + output[1:]
            return hidden_out
        return hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**inputs)

    target_handle.remove()
    downstream_handle.remove()

    hf_target_hidden = captured_hf_hiddens[target_layer]
    hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

    # -------------------------------------------------------------------------
    # vLLM path
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Build steering spec with high-magnitude operations
    norm = torch.norm(steering_vector).item()
    unit_vec = steering_vector / torch.norm(steering_vector)
    cap_unit = _normalize(cap_direction).to(dtype=torch.float32)
    ablation_unit = _normalize(ablation_direction).to(dtype=torch.float32)

    steering_spec = SteeringSpec(
        layers={
            target_layer: LayerSteeringSpec(operations=[
                AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm),
                ProjectionCapSpec(
                    vector=cap_unit,
                    min=projection_spec.min,
                    max=projection_spec.max,
                ),
                AblationSpec(
                    vector=ablation_unit,
                    scale=ablation_spec.scale,
                ),
            ])
        }
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        texts, handles = await vllm_model.generate(
            [prompt],
            sampling_params=sampling,
            steering_spec=steering_spec,
            capture_layers=[target_layer, downstream_layer]
        )

        await asyncio.gather(*[h.fetch() for h in handles])
        vllm_target_hidden = handles[0].captures[target_layer][0]["hidden"].to(dtype=torch.float32)
        vllm_downstream_hidden = handles[0].captures[downstream_layer][0]["hidden"].to(dtype=torch.float32)

        # -------------------------------------------------------------------------
        # Comparison
        # -------------------------------------------------------------------------
        # Target layer
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)
        target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
        target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

        # Downstream layer
        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
        downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
        downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

        # Print statistics
        hf_target_norm = torch.norm(hf_target_flat).item()
        vllm_target_norm = torch.norm(vllm_target_flat).item()
        hf_downstream_norm = torch.norm(hf_downstream_flat).item()
        vllm_downstream_norm = torch.norm(vllm_downstream_flat).item()

        print(f"\nTarget layer (vector + cap + ablation):")
        print(f"  HF norm: {hf_target_norm:.2f}, vLLM norm: {vllm_target_norm:.2f}")
        print(f"  Cosine similarity: {target_cos:.6f}")
        print(f"  MAE: {target_mae:.6f}")

        print(f"\nDownstream layer (all operations propagated):")
        print(f"  HF norm: {hf_downstream_norm:.2f}, vLLM norm: {vllm_downstream_norm:.2f}")
        print(f"  Cosine similarity: {downstream_cos:.6f}")
        print(f"  MAE: {downstream_mae:.6f}")

        # Assert high similarity even with aggressive ablation and capping
        assert target_cos > 0.999, (
            f"Target layer: Cosine similarity {target_cos:.6f} should be >0.999 "
            f"even with high-magnitude vector + cap + ablation"
        )
        assert target_mae < 0.01, (
            f"Target layer: MAE {target_mae:.6f} should be <0.01"
        )

        assert downstream_cos > 0.999, (
            f"Downstream layer: Cosine similarity {downstream_cos:.6f} should be >0.999, "
            f"indicating all operations propagate correctly"
        )
        assert downstream_mae < 0.01, (
            f"Downstream layer: MAE {downstream_mae:.6f} should be <0.01"
        )

        print(f"\n✓ High-magnitude ablation and projection capping work correctly")

    finally:
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_hf_multi_magnitude_steering():
    """Test that steering works correctly across a wide spectrum of magnitudes.

    This test validates the fix works from very subtle (0.001) to very aggressive (10.0)
    steering, ensuring no magnitude-dependent bugs in the residual stream handling.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "In the beginning, there was"

    # Load HF model once
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # Test a spectrum of magnitudes
    magnitudes = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]

    # Load vLLM model once
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    print(f"\nTesting {len(magnitudes)} magnitudes:")

    try:
        for magnitude in magnitudes:
            # Create steering vector at this magnitude
            torch.manual_seed(42)  # Same direction for all magnitudes
            steering_vector = torch.randn(hidden_size, dtype=torch.float32) * magnitude

            # -------------------------------------------------------------------------
            # HuggingFace path
            # -------------------------------------------------------------------------
            captured_hf_hiddens: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int):
                def hook(module, _args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    orig_dtype = hidden.dtype
                    hidden_fp32 = hidden.to(torch.float32)

                    # Apply steering only at target layer
                    if layer_idx == target_layer:
                        hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

                    captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
                    hidden_out = hidden_fp32.to(dtype=orig_dtype)
                    if isinstance(output, tuple):
                        return (hidden_out,) + output[1:]
                    return hidden_out
                return hook

            # Install hooks on both layers
            target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
            downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

            with torch.no_grad():
                hf_model(**inputs)

            target_handle.remove()
            downstream_handle.remove()

            hf_target_hidden = captured_hf_hiddens[target_layer]
            hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

            # -------------------------------------------------------------------------
            # vLLM path
            # -------------------------------------------------------------------------
            # Build steering spec for this magnitude
            norm = torch.norm(steering_vector).item()
            unit_vec = steering_vector / torch.norm(steering_vector)
            steering_spec = SteeringSpec(
                layers={
                    target_layer: LayerSteeringSpec(operations=[
                        AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
                    ])
                }
            )

            texts, handles = await vllm_model.generate(
                [prompt],
                sampling_params=sampling,
                steering_spec=steering_spec,
                capture_layers=[target_layer, downstream_layer]
            )

            await asyncio.gather(*[h.fetch() for h in handles])
            vllm_target_hidden = handles[0].captures[target_layer][0]["hidden"].to(dtype=torch.float32)
            vllm_downstream_hidden = handles[0].captures[downstream_layer][0]["hidden"].to(dtype=torch.float32)

            # -------------------------------------------------------------------------
            # Comparison
            # -------------------------------------------------------------------------
            # Target layer
            hf_target_flat = hf_target_hidden.reshape(-1)
            vllm_target_flat = vllm_target_hidden.reshape(-1)
            target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
            target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

            # Downstream layer
            hf_downstream_flat = hf_downstream_hidden.reshape(-1)
            vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
            downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
            downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

            vector_norm = torch.norm(steering_vector).item()

            print(f"\n  Magnitude {magnitude:5.3f} (vector norm: {vector_norm:6.2f}):")
            print(f"    Target:     cos={target_cos:.6f}, MAE={target_mae:.6f}")
            print(f"    Downstream: cos={downstream_cos:.6f}, MAE={downstream_mae:.6f}")

            # Assert high similarity across all magnitudes
            assert target_cos > 0.999, (
                f"Magnitude {magnitude}: Target layer cosine similarity {target_cos:.6f} should be >0.999"
            )
            assert target_mae < 0.01, (
                f"Magnitude {magnitude}: Target layer MAE {target_mae:.6f} should be <0.01"
            )

            assert downstream_cos > 0.999, (
                f"Magnitude {magnitude}: Downstream layer cosine similarity {downstream_cos:.6f} should be >0.999"
            )
            assert downstream_mae < 0.01, (
                f"Magnitude {magnitude}: Downstream layer MAE {downstream_mae:.6f} should be <0.01"
            )

        print(f"\n✓ All {len(magnitudes)} magnitudes (0.001 - 10.0) work correctly")

    finally:
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_hf_high_precision_steering():
    """Test steering with float32 precision to check for subtle numerical issues.

    If there's a bug in the residual stream handling, running both models in
    higher precision should still show errors. If errors decrease significantly,
    it suggests the implementation is correct but precision-limited.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "The meaning of life is"

    # Load HF model in FLOAT32
    print("\nLoading models in float32 precision...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # FLOAT32 instead of float16
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # Normal magnitude steering
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 1.0

    print(f"Steering vector norm: {torch.norm(steering_vector).item():.2f}")

    # -------------------------------------------------------------------------
    # HuggingFace path (float32)
    # -------------------------------------------------------------------------
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Already float32, no conversion needed

            # Apply steering only at target layer
            if layer_idx == target_layer:
                hidden = hidden + steering_vector.to(device=hidden.device)

            captured_hf_hiddens[layer_idx] = hidden.detach().cpu().clone()

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**inputs)

    target_handle.remove()
    downstream_handle.remove()

    hf_target_hidden = captured_hf_hiddens[target_layer]
    hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

    # -------------------------------------------------------------------------
    # vLLM path (float32)
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,  # Slightly more memory for float32
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float32",  # FLOAT32 instead of float16
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Build steering spec
    norm = torch.norm(steering_vector).item()
    unit_vec = steering_vector / torch.norm(steering_vector)
    steering_spec = SteeringSpec(
        layers={
            target_layer: LayerSteeringSpec(operations=[
                AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
            ])
        }
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        texts, handles = await vllm_model.generate(
            [prompt],
            sampling_params=sampling,
            steering_spec=steering_spec,
            capture_layers=[target_layer, downstream_layer]
        )

        await asyncio.gather(*[h.fetch() for h in handles])
        vllm_target_hidden = handles[0].captures[target_layer][0]["hidden"]
        vllm_downstream_hidden = handles[0].captures[downstream_layer][0]["hidden"]

        # -------------------------------------------------------------------------
        # Comparison (everything in float32)
        # -------------------------------------------------------------------------
        # Target layer
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)
        target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
        target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()
        target_max_diff = torch.max(torch.abs(vllm_target_flat - hf_target_flat)).item()

        # Downstream layer
        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
        downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
        downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()
        downstream_max_diff = torch.max(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()

        print(f"\nTarget layer (float32 precision):")
        print(f"  Cosine similarity: {target_cos:.12f}")
        print(f"  MAE: {target_mae:.12f}")
        print(f"  Max diff: {target_max_diff:.12f}")

        print(f"\nDownstream layer (float32 precision):")
        print(f"  Cosine similarity: {downstream_cos:.12f}")
        print(f"  MAE: {downstream_mae:.12f}")
        print(f"  Max diff: {downstream_max_diff:.12f}")

        # With float32, errors should be MUCH smaller if implementation is correct
        # If MAE is still ~0.001-0.003, something might be wrong
        print(f"\n{'='*60}")
        if target_mae < 1e-5 and downstream_mae < 1e-5:
            print("✓ Excellent: MAE < 1e-5 in float32")
            print("  This suggests the implementation is correct and float16")
            print("  precision was the limiting factor.")
        elif target_mae < 1e-4:
            print("⚠ Good but not perfect: MAE ~1e-4 to 1e-5")
            print("  Implementation likely correct, but check for subtle issues.")
        else:
            print(f"⚠ WARNING: MAE still high ({target_mae:.6e}) in float32!")
            print("  This suggests there may be a bug in the implementation.")
        print(f"{'='*60}")

        # More lenient assertion for float32 - should be near machine precision
        assert target_cos > 0.99999, (
            f"Target layer: Cosine similarity {target_cos:.12f} should be >0.99999 in float32"
        )
        assert downstream_cos > 0.99999, (
            f"Downstream layer: Cosine similarity {downstream_cos:.12f} should be >0.99999 in float32"
        )

    finally:
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_delta_vs_residual_numerics():
    """Compare numerical precision of adding steering to delta vs residual.

    Tests both float16 and bfloat16, comparing delta vs residual approaches.
    Uses float32 HuggingFace as ground truth for all comparisons.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "The quick brown fox"

    # -------------------------------------------------------------------------
    # Ground truth: Float32 HuggingFace
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Getting float32 HuggingFace ground truth...")
    print("="*70)
    
    hf_model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model_fp32.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model_fp32.config.hidden_size
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 1.0

    captured_ground_truth: dict[int, torch.Tensor] = {}

    def make_hook_fp32(layer_idx: int):
        def hook(module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer_idx == target_layer:
                hidden = hidden + steering_vector.to(device=hidden.device)
            captured_ground_truth[layer_idx] = hidden.detach().cpu().clone()
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    target_handle = hf_model_fp32.model.layers[target_layer].register_forward_hook(make_hook_fp32(target_layer))
    downstream_handle = hf_model_fp32.model.layers[downstream_layer].register_forward_hook(make_hook_fp32(downstream_layer))

    with torch.no_grad():
        hf_model_fp32(**inputs)

    target_handle.remove()
    downstream_handle.remove()

    ground_truth_target = captured_ground_truth[target_layer]
    ground_truth_downstream = captured_ground_truth[downstream_layer]

    del hf_model_fp32
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Test both dtypes and both approaches
    # -------------------------------------------------------------------------
    from chatspace.vllm_steering import runtime as steering_runtime
    original_apply_vector = steering_runtime._apply_vector_to_output

    def _apply_vector_to_residual(output, vector):
        """Alternative: add to RESIDUAL instead of delta."""
        if isinstance(output, tuple) and len(output) >= 2:
            delta, residual = output[0], output[1]
            if isinstance(delta, torch.Tensor) and isinstance(residual, torch.Tensor):
                new_residual = residual + vector
                return (delta, new_residual) + output[2:]
        return original_apply_vector(output, vector)

    results = {}
    dtypes = ["float16", "bfloat16"]
    approaches = [("delta", original_apply_vector), ("residual", _apply_vector_to_residual)]

    for dtype_name in dtypes:
        for approach_name, apply_fn in approaches:
            test_name = f"{dtype_name}_{approach_name}"
            print(f"\nTesting {dtype_name} with add-to-{approach_name}...")

            # Monkey-patch the apply function
            steering_runtime._apply_vector_to_output = apply_fn

            try:
                vllm_cfg = VLLMSteeringConfig(
                    model_name=model_name,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.05,
                    max_model_len=inputs.input_ids.shape[1] + 16,
                    dtype=dtype_name,
                )
                vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

                # Build steering spec
                norm = torch.norm(steering_vector).item()
                unit_vec = steering_vector / torch.norm(steering_vector)
                steering_spec = SteeringSpec(
                    layers={
                        target_layer: LayerSteeringSpec(operations=[
                            AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
                        ])
                    }
                )

                sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
                texts, handles = await vllm_model.generate(
                    [prompt],
                    sampling_params=sampling,
                    steering_spec=steering_spec,
                    capture_layers=[target_layer, downstream_layer]
                )

                await asyncio.gather(*[h.fetch() for h in handles])
                target_hidden = handles[0].captures[target_layer][0]["hidden"].to(dtype=torch.float32)
                downstream_hidden = handles[0].captures[downstream_layer][0]["hidden"].to(dtype=torch.float32)

                # Compute errors vs ground truth
                target_error = torch.mean(torch.abs(target_hidden.reshape(-1) - ground_truth_target.reshape(-1))).item()
                downstream_error = torch.mean(torch.abs(downstream_hidden.reshape(-1) - ground_truth_downstream.reshape(-1))).item()
                avg_error = (target_error + downstream_error) / 2

                results[test_name] = {
                    "target_error": target_error,
                    "downstream_error": downstream_error,
                    "avg_error": avg_error,
                }

                del vllm_model
                torch.cuda.empty_cache()

            finally:
                # Restore original
                steering_runtime._apply_vector_to_output = original_apply_vector

    # -------------------------------------------------------------------------
    # Print comparison results
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("RESULTS: Error vs float32 HuggingFace ground truth")
    print("="*70)

    for dtype_name in dtypes:
        print(f"\n{dtype_name.upper()}:")
        delta_result = results[f"{dtype_name}_delta"]
        residual_result = results[f"{dtype_name}_residual"]

        print(f"  Add to DELTA (current):")
        print(f"    Target:     {delta_result['target_error']:.6e}")
        print(f"    Downstream: {delta_result['downstream_error']:.6e}")
        print(f"    Average:    {delta_result['avg_error']:.6e}")

        print(f"  Add to RESIDUAL (alternative):")
        print(f"    Target:     {residual_result['target_error']:.6e}")
        print(f"    Downstream: {residual_result['downstream_error']:.6e}")
        print(f"    Average:    {residual_result['avg_error']:.6e}")

        if residual_result['avg_error'] < delta_result['avg_error'] * 0.95:
            improvement = (delta_result['avg_error'] - residual_result['avg_error']) / delta_result['avg_error'] * 100
            print(f"  → RESIDUAL is better by {improvement:.1f}%")
        elif delta_result['avg_error'] < residual_result['avg_error'] * 0.95:
            improvement = (residual_result['avg_error'] - delta_result['avg_error']) / residual_result['avg_error'] * 100
            print(f"  → DELTA is better by {improvement:.1f}%")
        else:
            print(f"  → Similar precision")

    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)

    # Find best overall
    best_name = min(results.keys(), key=lambda k: results[k]['avg_error'])
    best_error = results[best_name]['avg_error']

    print(f"Best configuration: {best_name}")
    print(f"  Average error: {best_error:.6e}")

    # Compare dtypes
    fp16_best = min(results[f"float16_{a}"]["avg_error"] for a, _ in approaches)
    bf16_best = min(results[f"bfloat16_{a}"]["avg_error"] for a, _ in approaches)

    print(f"\nBest by dtype:")
    print(f"  float16:  {fp16_best:.6e}")
    print(f"  bfloat16: {bf16_best:.6e}")

    if bf16_best < fp16_best * 0.95:
        improvement = (fp16_best - bf16_best) / fp16_best * 100
        print(f"  → bfloat16 is better by {improvement:.1f}%")
    elif fp16_best < bf16_best * 0.95:
        improvement = (bf16_best - fp16_best) / bf16_best * 100
        print(f"  → float16 is better by {improvement:.1f}%")
    else:
        print(f"  → Similar precision")

    print("="*70)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_delta_vs_residual_instrumented():
    """Instrument code paths to verify delta vs residual approaches are actually different.
    
    The previous test showed identical results, but we need to verify we're actually
    hitting different code paths and that the steering is being applied correctly.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    prompt = "The quick brown"

    # Get HF ground truth
    hf_model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cuda", attn_implementation="eager"
    )
    hf_model_fp32.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model_fp32.config.hidden_size
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 1.0

    captured_hf = None
    def hf_hook(module, args, output):
        nonlocal captured_hf
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + steering_vector.to(device=hidden.device)
        captured_hf = hidden.detach().cpu().clone()
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = hf_model_fp32.model.layers[target_layer].register_forward_hook(hf_hook)
    with torch.no_grad():
        hf_model_fp32(**inputs)
    handle.remove()

    print(f"\nHF ground truth: shape={captured_hf.shape}, mean={captured_hf.mean().item():.6f}, norm={torch.norm(captured_hf).item():.2f}")

    del hf_model_fp32
    torch.cuda.empty_cache()

    # Test vLLM with instrumentation
    from chatspace.vllm_steering import runtime as steering_runtime

    original_apply = steering_runtime._apply_vector_to_output
    call_logs = []

    def instrumented_apply(approach_name):
        def wrapped(output, vector):
            if isinstance(output, tuple) and len(output) >= 2:
                delta, residual = output[0], output[1]
                log_entry = {
                    "approach": approach_name,
                    "delta_mean": delta.mean().item(),
                    "residual_mean": residual.mean().item(),
                    "vector_norm": torch.norm(vector).item(),
                }
                
                if approach_name == "delta":
                    result = original_apply(output, vector)
                    if isinstance(result, tuple) and len(result) >= 2:
                        log_entry["modified_delta_mean"] = result[0].mean().item()
                        log_entry["modified_residual_mean"] = result[1].mean().item()
                        log_entry["delta_changed"] = abs(result[0].mean().item() - delta.mean().item()) > 1e-6
                        log_entry["residual_changed"] = abs(result[1].mean().item() - residual.mean().item()) > 1e-6
                else:  # residual
                    new_residual = residual + vector
                    result = (delta, new_residual) + output[2:]
                    log_entry["modified_delta_mean"] = delta.mean().item()
                    log_entry["modified_residual_mean"] = new_residual.mean().item()
                    log_entry["delta_changed"] = False
                    log_entry["residual_changed"] = True
                
                call_logs.append(log_entry)
                return result
            return original_apply(output, vector)
        return wrapped

    results = {}
    for approach_name in ["delta", "residual"]:
        call_logs.clear()
        steering_runtime._apply_vector_to_output = instrumented_apply(approach_name)

        try:
            vllm_cfg = VLLMSteeringConfig(
                model_name=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.2,
                max_model_len=inputs.input_ids.shape[1] + 16,
                dtype="float32",
            )
            vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))

            # Build steering spec
            norm = torch.norm(steering_vector).item()
            unit_vec = steering_vector / torch.norm(steering_vector)
            steering_spec = SteeringSpec(
                layers={
                    target_layer: LayerSteeringSpec(operations=[
                        AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
                    ])
                }
            )

            sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
            texts, handles = await vllm_model.generate(
                [prompt],
                sampling_params=sampling,
                steering_spec=steering_spec,
                capture_layers=[target_layer]
            )

            await asyncio.gather(*[h.fetch() for h in handles])
            hidden = handles[0].captures[target_layer][0]["hidden"]

            results[approach_name] = {
                "hidden": hidden,
                "call_count": len(call_logs),
                "call_logs": call_logs.copy(),
            }

            print(f"\n{approach_name.upper()} approach:")
            print(f"  Steering calls: {len(call_logs)}")
            print(f"  Hidden: shape={hidden.shape}, mean={hidden.mean().item():.6f}, norm={torch.norm(hidden).item():.2f}")
            
            for i, log in enumerate(call_logs):
                print(f"  Call {i+1}: delta_changed={log['delta_changed']}, residual_changed={log['residual_changed']}")

            error = torch.mean(torch.abs(hidden.cpu() - captured_hf)).item()
            results[approach_name]["error_vs_hf"] = error
            print(f"  Error vs HF: {error:.6e}")

            del vllm_model
            torch.cuda.empty_cache()

        finally:
            steering_runtime._apply_vector_to_output = original_apply

    # Compare approaches
    delta_hidden = results["delta"]["hidden"]
    residual_hidden = results["residual"]["hidden"]
    diff = torch.mean(torch.abs(delta_hidden.cpu() - residual_hidden.cpu())).item()

    print(f"\n{'='*70}")
    print(f"Difference between delta vs residual approaches: {diff:.6e}")
    
    # Check if they're actually hitting different code paths
    delta_modified_delta = any(log["delta_changed"] for log in results["delta"]["call_logs"])
    delta_modified_residual = any(log["residual_changed"] for log in results["delta"]["call_logs"])
    residual_modified_delta = any(log["delta_changed"] for log in results["residual"]["call_logs"])
    residual_modified_residual = any(log["residual_changed"] for log in results["residual"]["call_logs"])

    print(f"\nCode path verification:")
    print(f"  Delta approach:    modified delta={delta_modified_delta}, modified residual={delta_modified_residual}")
    print(f"  Residual approach: modified delta={residual_modified_delta}, modified residual={residual_modified_residual}")

    if delta_modified_delta and not delta_modified_residual and not residual_modified_delta and residual_modified_residual:
        print(f"  ✓ Code paths are DIFFERENT (as intended)")
    else:
        print(f"  ⚠ WARNING: Code paths may not be what we expect!")

    print(f"{'='*70}")

    # They should be mathematically equivalent
    assert diff < 1e-5, f"Delta and residual should give same results, got diff={diff:.6e}"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_vllm_hf_multi_layer_steering_float32():
    """Test multi-layer steering in float32 to verify logic correctness.

    This test addresses real-world observations of performance degradation that gets worse
    with more layers (and collapses with every-layer steering). We use float32 to eliminate
    numerical precision as a confounding factor and focus on logic correctness.

    We validate:
    1. Multiple consecutive steered layers (3, 4, 5, 6)
    2. Downstream propagation (check layers 7, 8 which are 2+ layers beyond last steered layer)
    3. Both multi-layer and every-layer steering scenarios
    """
    torch.manual_seed(789)

    model_name = "Qwen/Qwen3-0.6B"
    prompt = "The nature of consciousness and the boundaries between"

    # Load HF model in float32
    print("\n" + "="*70)
    print("Loading HuggingFace model in float32...")
    print("="*70)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size
    num_layers = hf_model.config.num_hidden_layers

    # Test 1: Steer on layers 3, 4, 5, 6 (4 consecutive mid-layers)
    print("\n" + "="*70)
    print("Test 1: Multi-layer steering (layers 3, 4, 5, 6)")
    print("="*70)

    steered_layers = [3, 4, 5, 6]
    downstream_layers = [7, 8]  # Check 2 layers beyond last steered layer
    all_check_layers = steered_layers + downstream_layers

    # Create independent steering vectors for each layer
    steering_vectors = {}
    for layer_idx in steered_layers:
        steering_vectors[layer_idx] = torch.randn(hidden_size, dtype=torch.float32) * 0.1

    # HF: Install hooks on all layers we want to check
    hf_captured = {layer: None for layer in all_check_layers}

    def make_hf_hook(layer_idx: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Apply steering if this is a steered layer
            if layer_idx in steering_vectors:
                hidden = hidden + steering_vectors[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
            # Capture for comparison
            hf_captured[layer_idx] = hidden.detach().cpu().clone()
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    handles = []
    for layer_idx in all_check_layers:
        handle = hf_model.model.layers[layer_idx].register_forward_hook(make_hf_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        hf_model(**inputs)

    for handle in handles:
        handle.remove()

    print("\nHuggingFace results:")
    for layer_idx in all_check_layers:
        captured = hf_captured[layer_idx]
        print(f"  Layer {layer_idx}: shape={captured.shape}, mean={captured.mean().item():.6f}, "
              f"std={captured.std().item():.6f}, norm={torch.norm(captured).item():.2f}")

    # Clean up HF model
    del hf_model
    torch.cuda.empty_cache()

    # vLLM: Set up model with steering
    print("\n" + "="*70)
    print("Loading vLLM model in float32...")
    print("="*70)

    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float32",
    )

    vllm_model = VLLMSteerModel(
        vllm_cfg,
        enforce_eager=True,
        bootstrap_layers=tuple(all_check_layers),
    )

    # Build steering spec for all steered layers
    layers_dict = {}
    for layer_idx, vector in steering_vectors.items():
        norm = torch.norm(vector).item()
        unit_vec = vector / torch.norm(vector)
        layers_dict[layer_idx] = LayerSteeringSpec(operations=[
            AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
        ])
    steering_spec = SteeringSpec(layers=layers_dict)

    # Generate with capture on all layers we want to check
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
    texts, handles = await vllm_model.generate(
        [prompt],
        sampling_params=sampling,
        steering_spec=steering_spec,
        capture_layers=list(all_check_layers)
    )

    await asyncio.gather(*[h.fetch() for h in handles])

    print("\nvLLM results:")
    for layer_idx in all_check_layers:
        vllm_hidden = handles[0].captures[layer_idx][0]["hidden"]
        print(f"  Layer {layer_idx}: shape={vllm_hidden.shape}, mean={vllm_hidden.mean().item():.6f}, "
              f"std={vllm_hidden.std().item():.6f}, norm={torch.norm(vllm_hidden).item():.2f}")

    # Compare HF vs vLLM for all layers
    print("\n" + "="*70)
    print("Multi-layer steering comparison (HF vs vLLM):")
    print("="*70)

    all_pass = True
    for layer_idx in all_check_layers:
        hf_hidden = hf_captured[layer_idx]
        vllm_hidden = handles[0].captures[layer_idx][0]["hidden"]

        cos_sim = F.cosine_similarity(
            hf_hidden.flatten().unsqueeze(0),
            vllm_hidden.cpu().flatten().unsqueeze(0),
        ).item()

        mae = torch.mean(torch.abs(vllm_hidden.cpu() - hf_hidden)).item()
        max_diff = torch.max(torch.abs(vllm_hidden.cpu() - hf_hidden)).item()

        is_steered = layer_idx in steered_layers
        is_downstream = layer_idx in downstream_layers

        layer_type = "STEERED" if is_steered else "DOWNSTREAM"
        print(f"\nLayer {layer_idx} ({layer_type}):")
        print(f"  Cosine similarity: {cos_sim:.9f}")
        print(f"  MAE: {mae:.6e}")
        print(f"  Max diff: {max_diff:.6e}")

        # Stricter threshold for float32
        threshold = 0.99999
        if cos_sim < threshold:
            print(f"  ❌ FAIL: cosine similarity {cos_sim:.9f} < {threshold}")
            all_pass = False
        else:
            print(f"  ✓ PASS: cosine similarity {cos_sim:.9f} >= {threshold}")

    del vllm_model
    torch.cuda.empty_cache()

    print("\n" + "="*70)
    if all_pass:
        print("✓ Multi-layer steering: ALL LAYERS PASS")
    else:
        print("❌ Multi-layer steering: SOME LAYERS FAILED")
    print("="*70)

    # Test 2: Steer on EVERY layer (stress test to reproduce collapse)
    print("\n" + "="*70)
    print("Test 2: Every-layer steering (stress test)")
    print("="*70)

    # Only test on a subset of layers due to memory constraints
    every_layer_test_layers = list(range(0, min(10, num_layers)))  # First 10 layers
    downstream_for_every = [10, 11]  # Check downstream propagation
    all_every_check = every_layer_test_layers + downstream_for_every

    print(f"Steering layers: {every_layer_test_layers}")
    print(f"Checking downstream: {downstream_for_every}")

    # Create steering vectors for every layer
    every_layer_vectors = {}
    for layer_idx in every_layer_test_layers:
        every_layer_vectors[layer_idx] = torch.randn(hidden_size, dtype=torch.float32) * 0.1

    # Load HF model again
    print("\nLoading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    # HF: Install hooks
    hf_every_captured = {layer: None for layer in all_every_check}

    def make_every_hf_hook(layer_idx: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer_idx in every_layer_vectors:
                hidden = hidden + every_layer_vectors[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
            hf_every_captured[layer_idx] = hidden.detach().cpu().clone()
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    handles = []
    for layer_idx in all_every_check:
        handle = hf_model.model.layers[layer_idx].register_forward_hook(make_every_hf_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        hf_model(**inputs)

    for handle in handles:
        handle.remove()

    print("\nHuggingFace results (sample layers):")
    for layer_idx in [0, 5, 9, 10, 11]:
        if layer_idx in hf_every_captured and hf_every_captured[layer_idx] is not None:
            captured = hf_every_captured[layer_idx]
            print(f"  Layer {layer_idx}: mean={captured.mean().item():.6f}, "
                  f"norm={torch.norm(captured).item():.2f}")

    del hf_model
    torch.cuda.empty_cache()

    # vLLM: Every-layer steering
    print("\nLoading vLLM model...")
    vllm_cfg_every = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float32",
    )

    vllm_every_model = VLLMSteerModel(
        vllm_cfg_every,
        enforce_eager=True,
        bootstrap_layers=tuple(all_every_check),
    )

    # Build steering spec for every layer
    every_layers_dict = {}
    for layer_idx, vector in every_layer_vectors.items():
        norm = torch.norm(vector).item()
        unit_vec = vector / torch.norm(vector)
        every_layers_dict[layer_idx] = LayerSteeringSpec(operations=[
            AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
        ])
    every_steering_spec = SteeringSpec(layers=every_layers_dict)

    # Generate with capture on all layers
    texts, handles = await vllm_every_model.generate(
        [prompt],
        sampling_params=sampling,
        steering_spec=every_steering_spec,
        capture_layers=list(all_every_check)
    )
    await asyncio.gather(*[h.fetch() for h in handles])

    print("\nvLLM results (sample layers):")
    for layer_idx in [0, 5, 9, 10, 11]:
        if layer_idx in all_every_check:
            vllm_hidden = handles[0].captures[layer_idx][0]["hidden"]
            print(f"  Layer {layer_idx}: mean={vllm_hidden.mean().item():.6f}, "
                  f"norm={torch.norm(vllm_hidden).item():.2f}")

    # Compare key layers
    print("\n" + "="*70)
    print("Every-layer steering comparison (HF vs vLLM):")
    print("="*70)

    every_pass = True
    for layer_idx in all_every_check:
        hf_hidden = hf_every_captured[layer_idx]
        vllm_hidden = handles[0].captures[layer_idx][0]["hidden"]

        cos_sim = F.cosine_similarity(
            hf_hidden.flatten().unsqueeze(0),
            vllm_hidden.cpu().flatten().unsqueeze(0),
        ).item()

        mae = torch.mean(torch.abs(vllm_hidden.cpu() - hf_hidden)).item()

        is_steered = layer_idx in every_layer_vectors
        is_downstream = layer_idx in downstream_for_every

        layer_type = "STEERED" if is_steered else "DOWNSTREAM"
        print(f"\nLayer {layer_idx} ({layer_type}):")
        print(f"  Cosine similarity: {cos_sim:.9f}")
        print(f"  MAE: {mae:.6e}")

        if cos_sim < 0.99999:
            print(f"  ❌ FAIL")
            every_pass = False
        else:
            print(f"  ✓ PASS")

    del vllm_every_model
    torch.cuda.empty_cache()

    print("\n" + "="*70)
    if every_pass:
        print("✓ Every-layer steering: ALL LAYERS PASS")
    else:
        print("❌ Every-layer steering: SOME LAYERS FAILED")
    print("="*70)

    # Overall assertion
    assert all_pass, "Multi-layer steering test failed - some layers showed divergence"
    assert every_pass, "Every-layer steering test failed - some layers showed divergence"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_bf16_degradation_hf_vs_vllm():
    """Compare bfloat16 degradation patterns between HuggingFace and vLLM.

    This test measures how numerical errors accumulate differently in HF vs vLLM
    as we increase the number of steered layers. We use float32 HF as ground truth
    and measure MAE for both bf16 implementations across different layer counts.

    Expected: If vLLM degrades worse, this reveals amplification in our steering path.
    """
    torch.manual_seed(999)

    model_name = "Qwen/Qwen3-0.6B"
    prompt = "In the depths of computational theory and practice"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Get model config
    print("\n" + "="*70)
    print("Loading config to determine architecture...")
    print("="*70)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    print(f"Model: {num_layers} layers, hidden_size={hidden_size}")

    # Test configurations: increasing layer counts
    layer_configs = [
        {"name": "1_layer", "layers": [10]},
        {"name": "4_layers", "layers": [8, 9, 10, 11]},
        {"name": "10_layers", "layers": list(range(5, 15))},
        {"name": "all_layers", "layers": list(range(num_layers))},
    ]

    results = {
        "config": [],
        "num_steered": [],
        "check_layer": [],
        "hf_bf16_mae": [],
        "vllm_bf16_mae": [],
        "hf_bf16_cos": [],
        "vllm_bf16_cos": [],
    }

    for config_idx, layer_config in enumerate(layer_configs):
        config_name = layer_config["name"]
        steered_layers = layer_config["layers"]
        # Check the last steered layer and 2 layers downstream
        check_layer = max(steered_layers) + 2
        if check_layer >= num_layers:
            check_layer = num_layers - 1

        print(f"\n{'='*70}")
        print(f"Config {config_idx+1}/{len(layer_configs)}: {config_name}")
        print(f"  Steering {len(steered_layers)} layers: {steered_layers[:3]}...{steered_layers[-3:]}" if len(steered_layers) > 6 else f"  Steering layers: {steered_layers}")
        print(f"  Checking layer {check_layer} (downstream)")
        print(f"{'='*70}")

        # Create steering vectors for this config
        steering_vectors = {}
        for layer_idx in steered_layers:
            steering_vectors[layer_idx] = torch.randn(hidden_size, dtype=torch.float32) * 0.1

        # GROUND TRUTH: HuggingFace float32
        print("\n[1/3] Loading HuggingFace float32 (ground truth)...")
        hf_fp32 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cuda",
            attn_implementation="eager",
        )
        hf_fp32.eval()

        hf_fp32_hidden = None
        def hf_fp32_hook(module, args, output):
            nonlocal hf_fp32_hidden
            hidden = output[0] if isinstance(output, tuple) else output
            # Apply steering if this layer is in our config
            for layer_idx in steered_layers:
                if module == hf_fp32.model.layers[layer_idx]:
                    hidden = hidden + steering_vectors[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
                    break
            # Capture at check layer
            if module == hf_fp32.model.layers[check_layer]:
                hf_fp32_hidden = hidden.detach().cpu().clone()
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handles = []
        for layer_idx in steered_layers + [check_layer]:
            handle = hf_fp32.model.layers[layer_idx].register_forward_hook(hf_fp32_hook)
            handles.append(handle)

        with torch.no_grad():
            hf_fp32(**inputs)

        for handle in handles:
            handle.remove()

        print(f"  Ground truth: norm={torch.norm(hf_fp32_hidden).item():.2f}")

        del hf_fp32
        torch.cuda.empty_cache()

        # TEST 1: HuggingFace bfloat16
        print("\n[2/3] Loading HuggingFace bfloat16...")
        hf_bf16 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="eager",
        )
        hf_bf16.eval()

        hf_bf16_hidden = None
        def hf_bf16_hook(module, args, output):
            nonlocal hf_bf16_hidden
            hidden = output[0] if isinstance(output, tuple) else output
            for layer_idx in steered_layers:
                if module == hf_bf16.model.layers[layer_idx]:
                    hidden = hidden + steering_vectors[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
                    break
            if module == hf_bf16.model.layers[check_layer]:
                hf_bf16_hidden = hidden.detach().cpu().clone()
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handles = []
        for layer_idx in steered_layers + [check_layer]:
            handle = hf_bf16.model.layers[layer_idx].register_forward_hook(hf_bf16_hook)
            handles.append(handle)

        with torch.no_grad():
            hf_bf16(**inputs)

        for handle in handles:
            handle.remove()

        hf_bf16_mae = torch.mean(torch.abs(hf_bf16_hidden.float() - hf_fp32_hidden)).item()
        hf_bf16_cos = F.cosine_similarity(
            hf_fp32_hidden.flatten().unsqueeze(0),
            hf_bf16_hidden.float().flatten().unsqueeze(0),
        ).item()

        print(f"  HF bf16 vs fp32: MAE={hf_bf16_mae:.6e}, cos={hf_bf16_cos:.9f}")

        del hf_bf16
        torch.cuda.empty_cache()

        # TEST 2: vLLM bfloat16
        print("\n[3/3] Loading vLLM bfloat16...")
        vllm_cfg = VLLMSteeringConfig(
            model_name=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_model_len=inputs.input_ids.shape[1] + 16,
            dtype="bfloat16",
        )

        all_layers_to_bootstrap = list(set(steered_layers + [check_layer]))
        vllm_model = VLLMSteerModel(
            vllm_cfg,
            enforce_eager=True,
            bootstrap_layers=tuple(all_layers_to_bootstrap),
        )

        # Build steering spec for all steered layers
        vllm_layers_dict = {}
        for layer_idx, vector in steering_vectors.items():
            norm = torch.norm(vector).item()
            unit_vec = vector / torch.norm(vector)
            vllm_layers_dict[layer_idx] = LayerSteeringSpec(operations=[
                AddSpec(vector=unit_vec.to(dtype=torch.float32), scale=norm)
            ])
        vllm_steering_spec = SteeringSpec(layers=vllm_layers_dict)

        sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
        texts, handles = await vllm_model.generate(
            [prompt],
            sampling_params=sampling,
            steering_spec=vllm_steering_spec,
            capture_layers=[check_layer]
        )

        await asyncio.gather(*[h.fetch() for h in handles])
        vllm_bf16_hidden = handles[0].captures[check_layer][0]["hidden"]

        vllm_bf16_mae = torch.mean(torch.abs(vllm_bf16_hidden.cpu().float() - hf_fp32_hidden)).item()
        vllm_bf16_cos = F.cosine_similarity(
            hf_fp32_hidden.flatten().unsqueeze(0),
            vllm_bf16_hidden.cpu().float().flatten().unsqueeze(0),
        ).item()

        print(f"  vLLM bf16 vs fp32: MAE={vllm_bf16_mae:.6e}, cos={vllm_bf16_cos:.9f}")

        del vllm_model
        torch.cuda.empty_cache()

        # Record results
        results["config"].append(config_name)
        results["num_steered"].append(len(steered_layers))
        results["check_layer"].append(check_layer)
        results["hf_bf16_mae"].append(hf_bf16_mae)
        results["vllm_bf16_mae"].append(vllm_bf16_mae)
        results["hf_bf16_cos"].append(hf_bf16_cos)
        results["vllm_bf16_cos"].append(vllm_bf16_cos)

        # Compare degradation
        ratio = vllm_bf16_mae / hf_bf16_mae if hf_bf16_mae > 0 else float('inf')
        print(f"\n  Degradation ratio (vLLM/HF): {ratio:.2f}x")
        if ratio > 1.5:
            print(f"  ⚠️  vLLM degrades {ratio:.1f}x WORSE than HF")
        elif ratio < 0.67:
            print(f"  ✓ vLLM degrades {1/ratio:.1f}x BETTER than HF")
        else:
            print(f"  ≈ Similar degradation")

    # Summary report
    print("\n" + "="*70)
    print("DEGRADATION SUMMARY")
    print("="*70)
    print(f"{'Config':<15} {'Layers':<8} {'HF bf16 MAE':<15} {'vLLM bf16 MAE':<15} {'Ratio':<10}")
    print("-"*70)

    for i in range(len(results["config"])):
        config_name = results["config"][i]
        num_steered = results["num_steered"][i]
        hf_mae = results["hf_bf16_mae"][i]
        vllm_mae = results["vllm_bf16_mae"][i]
        ratio = vllm_mae / hf_mae if hf_mae > 0 else float('inf')

        print(f"{config_name:<15} {num_steered:<8} {hf_mae:<15.6e} {vllm_mae:<15.6e} {ratio:<10.2f}x")

    print("="*70)

    # Analyze trend
    import numpy as np
    ratios = np.array([results["vllm_bf16_mae"][i] / results["hf_bf16_mae"][i]
                       for i in range(len(results["config"]))
                       if results["hf_bf16_mae"][i] > 0])

    print(f"\nDegradation ratio statistics:")
    print(f"  Mean: {ratios.mean():.2f}x")
    print(f"  Std:  {ratios.std():.2f}x")
    print(f"  Min:  {ratios.min():.2f}x")
    print(f"  Max:  {ratios.max():.2f}x")

    if ratios.mean() > 2.0:
        print(f"\n❌ vLLM degrades significantly worse than HF (avg {ratios.mean():.1f}x)")
        print("   → This suggests our steering path amplifies bf16 precision errors")
    elif ratios.mean() < 0.5:
        print(f"\n✓ vLLM degrades less than HF (avg {1/ratios.mean():.1f}x better)")
    else:
        print(f"\n≈ Similar degradation patterns between HF and vLLM")

    # Check if degradation grows with layer count
    num_layers_list = np.array(results["num_steered"])
    if len(num_layers_list) >= 3:
        # Simple correlation check
        from scipy.stats import pearsonr
        hf_corr, _ = pearsonr(num_layers_list, results["hf_bf16_mae"])
        vllm_corr, _ = pearsonr(num_layers_list, results["vllm_bf16_mae"])

        print(f"\nError growth with layer count:")
        print(f"  HF bf16:   correlation = {hf_corr:.3f}")
        print(f"  vLLM bf16: correlation = {vllm_corr:.3f}")

        if vllm_corr > hf_corr + 0.1:
            print(f"  → vLLM error grows faster with more layers")
        elif hf_corr > vllm_corr + 0.1:
            print(f"  → HF error grows faster with more layers")
        else:
            print(f"  → Similar growth patterns")

    print("="*70)
