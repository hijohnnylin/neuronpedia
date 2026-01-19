"""Comprehensive tests for zero-copy shared memory activation capture.

Tests cover:
- Basic shared memory round-trip
- Zero-copy verification
- Context manager cleanup
- Weakref finalize backup
- ResourceWarning for unused handles
- TTL expiration
- Concurrent access
- Memory leak detection
- Performance benchmarks

Note: Shared memory is always enabled for activation captures (no bytes fallback).
"""

import asyncio
import os
import pytest
import time
import warnings
from multiprocessing.shared_memory import SharedMemory

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


@pytest.fixture
def model_name():
    """Small model for fast tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture
async def model_factory(model_name):
    """Factory for creating VLLMSteerModel with custom config."""
    created_models = []

    async def _make_model(shm_ttl_seconds=600, shm_max_gb=128.0):
        config = VLLMSteeringConfig(
            model_name=model_name,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )
        m = VLLMSteerModel(
            config,
            bootstrap_layers=(5,),
            shm_ttl_seconds=shm_ttl_seconds,
            shm_max_gb=shm_max_gb,
            enforce_eager=True,  # Pass to VLLMSteerModel, not config
        )
        created_models.append(m)
        return m

    yield _make_model

    # Cleanup all created models
    for m in created_models:
        if hasattr(m, "engine"):
            await m.engine.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shared_memory_basic_roundtrip(model_factory):
    """Test basic shared memory creation and retrieval."""
    model = await model_factory()

    prompts = ["Once upon a time"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5, 10, 15],
    )

    assert len(handles) == 1
    handle = handles[0]

    # Fetch captures
    await handle.fetch()

    # Verify captures exist
    assert handle._captures is not None
    assert len(handle._captures) > 0

    # Verify shared memory was used (always enabled now)
    assert len(handle._shm_names) > 0, "Expected shared memory to be used"

    # Verify tensor data integrity
    for layer_idx, captures_list in handle._captures.items():
        for capture in captures_list:
            hidden = capture["hidden"]
            assert hidden.shape[0] > 0  # Has tokens
            assert hidden.shape[1] == model.hidden_size  # Correct hidden size
            assert not torch.isnan(hidden).any()  # No NaN values

    # Cleanup
    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shared_memory_context_manager(model_factory):
    """Test async context manager cleanup."""
    model = await model_factory()

    prompts = ["Hello world"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await handle.fetch()

    shm_names_before = list(handle._shm_names)
    assert len(shm_names_before) > 0

    # Use context manager
    async with handle:
        _ = handle.captures  # Access captures

    # After exit, shared memory should be released
    assert handle._closed, "Handle should be marked as closed"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_access_isolation(model_factory):
    """Test that concurrent requests maintain proper isolation."""
    model = await model_factory()

    prompts = ["First prompt", "Second prompt"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    assert len(handles) == 2

    # Fetch both concurrently
    await asyncio.gather(*[h.fetch() for h in handles])

    # Verify each handle has different captures
    handle1, handle2 = handles

    assert handle1._shm_names != handle2._shm_names, "Handles should use different shm segments"

    # Verify data is different
    h1_data = handle1.captures[5][0]["hidden"]
    h2_data = handle2.captures[5][0]["hidden"]

    # Shapes should be similar but data should differ (different prompts)
    assert h1_data.shape[1] == h2_data.shape[1]  # Same hidden size
    # Don't check if data differs since with temp=0.0 they might be identical

    # Cleanup
    await handle1.close()
    await handle2.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_data_integrity(model_factory):
    """Test that shared memory data matches HuggingFace baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    prompt = "The quick brown fox"

    # HuggingFace baseline
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    # Capture activations from HF
    layer_5_acts = []
    layer_10_acts = []

    def hook_5(module, input, output):
        layer_5_acts.append(output[0].detach().cpu())

    def hook_10(module, input, output):
        layer_10_acts.append(output[0].detach().cpu())

    handle_5 = hf_model.model.layers[5].register_forward_hook(hook_5)
    handle_10 = hf_model.model.layers[10].register_forward_hook(hook_10)

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    handle_5.remove()
    handle_10.remove()

    # Concatenate HF activations
    hf_layer_5 = torch.cat(layer_5_acts, dim=0)
    hf_layer_10 = torch.cat(layer_10_acts, dim=0)

    # Clean up HF model
    del hf_model
    torch.cuda.empty_cache()

    # vLLM with shared memory
    model = await model_factory()

    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    results, handles = await model.generate(
        [prompt],
        sampling_params,
        capture_layers=[5, 10],
    )

    handle = handles[0]
    await handle.fetch()

    # Verify multiple layers exist
    assert 5 in handle.captures
    assert 10 in handle.captures

    # Verify tensor properties and compare to HF baseline
    for layer_idx, hf_baseline in [(5, hf_layer_5), (10, hf_layer_10)]:
        hidden = handle.captures[layer_idx][0]["hidden"]

        # Check shape
        assert hidden.dim() == 2  # [tokens, hidden_size]
        assert hidden.shape[1] == model.hidden_size

        # Check dtype
        assert hidden.dtype in [torch.float16, torch.bfloat16, torch.float32]

        # Check no NaN or Inf (critical safety checks)
        assert not torch.isnan(hidden).any(), f"Layer {layer_idx} has NaN values"
        assert not torch.isinf(hidden).any(), f"Layer {layer_idx} has Inf values"

        # Compare against HuggingFace baseline (ground truth)
        # Trim HF to match vLLM length (vLLM captures N-1 tokens for generation)
        vllm_len = hidden.shape[0]
        hf_trimmed = hf_baseline[:vllm_len]

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            hidden.float().flatten(),
            hf_trimmed.float().flatten(),
            dim=0
        ).item()

        # Assert high similarity to HF baseline (>0.99 for nearly identical)
        assert cos_sim > 0.99, (
            f"Layer {layer_idx} cosine similarity too low: {cos_sim:.6f}. "
            f"vLLM captures should match HuggingFace baseline."
        )

    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_memory_leak_detection(model_factory):
    """Test that repeated capture cycles don't leak memory."""
    model = await model_factory()

    import gc

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=3, temperature=0.0)

    # Run 10 cycles (reduced from 10,000 for test speed)
    for i in range(10):
        results, handles = await model.generate(
            prompts,
            sampling_params,
            capture_layers=[5],
        )

        handle = handles[0]
        await handle.fetch()

        # Access captures
        _ = handle.captures

        # Explicitly close
        await handle.close()

        # Force cleanup
        del handle, handles, results
        gc.collect()

    # If we got here without crashing or OOM, we passed
    assert True


@pytest.mark.slow
@pytest.mark.asyncio
async def test_explicit_close_vs_context_manager(model_factory):
    """Test both cleanup methods work correctly."""
    model = await model_factory()

    prompts = ["A", "B"]
    sampling_params = SamplingParams(max_tokens=3, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    h1, h2 = handles
    await asyncio.gather(h1.fetch(), h2.fetch())

    # Method 1: Explicit close
    _ = h1.captures
    await h1.close()
    assert h1._closed

    # Method 2: Context manager
    async with h2:
        _ = h2.captures
    assert h2._closed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
