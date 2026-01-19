"""Comprehensive integration tests for VLLMSteeringModel.

These tests verify the full functionality of the vLLM backend including:
- Model loading and initialization
- Basic text generation
- Steering (additive, projection cap, ablation)
- Activation capture
- Chat generation
- Concurrent generation
- Per-request steering isolation

Note: All tests are combined into a single test function to avoid the complexity
of module-scoped async fixtures which can cause deadlocks with pytest-asyncio.
"""

from __future__ import annotations

import asyncio

import pytest
import torch


# Skip all tests if vLLM is not available
try:
    from vllm import SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


# Test model - small for fast testing
MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.asyncio
async def test_vllm_integration_comprehensive():
    """Comprehensive integration test covering all vLLM steering functionality.

    This combines all tests into one function to share the model instance
    and avoid async fixture issues with pytest-asyncio.
    """
    from steerllm import (
        VLLMSteeringModel,
        SteeringSpec,
        LayerSteeringSpec,
        AddSpec,
        ProjectionCapSpec,
        AblationSpec,
    )

    # Load model once for all tests
    model = VLLMSteeringModel(
        MODEL_NAME,
        gpu_memory_utilization=0.3,
        max_model_len=512,
    )

    try:
        # Initialize
        await model.generate(["Hello"], max_tokens=1)

        # === Test Basic Generation ===

        # Single prompt generation
        texts, handles = await model.generate(
            ["The capital of France is"],
            max_tokens=20,
            temperature=0.0,
        )
        assert len(texts) == 1
        assert len(texts[0]) > 0
        assert handles is None

        # Batch generation
        prompts = ["The sky is", "Water is", "The sun is", "Trees are"]
        texts, handles = await model.generate(prompts, max_tokens=20, temperature=0.0)
        assert len(texts) == 4
        for text in texts:
            assert len(text) > 0

        # === Test Steering ===

        # Additive steering
        prompt = ["The meaning of life is"]
        texts_base, _ = await model.generate(prompt, max_tokens=30, temperature=0.0)

        direction = torch.randn(model.hidden_size)
        steering = SteeringSpec.simple_add(layer=5, vector=direction, scale=5.0)
        texts_steered, _ = await model.generate(
            prompt, max_tokens=30, temperature=0.0, steering_spec=steering
        )
        assert texts_steered[0] != texts_base[0]  # Steering should change output

        # Multi-layer steering
        v1 = torch.randn(model.hidden_size)
        v2 = torch.randn(model.hidden_size)
        v1 = v1 / v1.norm()
        v2 = v2 / v2.norm()

        steering = SteeringSpec(layers={
            3: LayerSteeringSpec(operations=[AddSpec(vector=v1, scale=2.0)]),
            7: LayerSteeringSpec(operations=[AddSpec(vector=v2, scale=2.0)]),
        })
        texts, _ = await model.generate(
            ["Once upon a time"], max_tokens=30, temperature=0.0, steering_spec=steering
        )
        assert len(texts[0]) > 0

        # Projection cap steering
        direction = torch.randn(model.hidden_size)
        steering = SteeringSpec.simple_cap(layer=5, vector=direction, min=-0.5, max=0.5)
        texts, _ = await model.generate(
            ["The quick brown fox"], max_tokens=30, temperature=0.0, steering_spec=steering
        )
        assert len(texts[0]) > 0

        # Ablation steering
        direction = torch.randn(model.hidden_size)
        steering = SteeringSpec.simple_ablation(layer=5, vector=direction, scale=0.0)
        texts, _ = await model.generate(
            ["The answer is"], max_tokens=30, temperature=0.0, steering_spec=steering
        )
        assert len(texts[0]) > 0

        # Multi-operation steering (add + cap + ablation in one layer)
        v1 = torch.randn(model.hidden_size)
        v2 = torch.randn(model.hidden_size)
        v3 = torch.randn(model.hidden_size)
        v1 = v1 / v1.norm()
        v2 = v2 / v2.norm()
        v3 = v3 / v3.norm()

        steering = SteeringSpec(layers={
            5: LayerSteeringSpec(operations=[
                AddSpec(vector=v1, scale=1.0),
                ProjectionCapSpec(vector=v2, min=-1.0, max=1.0),
                AblationSpec(vector=v3, scale=0.5),
            ])
        })
        texts, _ = await model.generate(
            ["In a galaxy far away"], max_tokens=30, temperature=0.0, steering_spec=steering
        )
        assert len(texts[0]) > 0

        # === Test Activation Capture ===

        # Basic capture
        capture_layers = [5, 10]
        texts, handles = await model.generate(
            ["Hello world"], max_tokens=10, temperature=0.0, capture_layers=capture_layers
        )
        assert handles is not None
        assert len(handles) == 1

        async with handles[0] as handle:
            await handle.fetch()  # Must fetch before accessing captures
            assert handle.captures is not None
            assert 5 in handle.captures
            assert 10 in handle.captures
            for layer_idx in capture_layers:
                layer_captures = handle.captures[layer_idx]
                assert len(layer_captures) > 0
                hidden = layer_captures[0]["hidden"]
                assert hidden.ndim == 2
                assert hidden.shape[1] == model.hidden_size

        # Capture with steering
        direction = torch.randn(model.hidden_size)
        steering = SteeringSpec.simple_add(layer=5, vector=direction, scale=1.0)
        texts, handles = await model.generate(
            ["Testing capture"], max_tokens=10, temperature=0.0,
            steering_spec=steering, capture_layers=[5, 10]
        )
        assert handles is not None
        async with handles[0] as handle:
            await handle.fetch()
            assert 5 in handle.captures
            assert 10 in handle.captures

        # Batch capture
        prompts = ["First prompt", "Second prompt", "Third prompt"]
        texts, handles = await model.generate(
            prompts, max_tokens=10, temperature=0.0, capture_layers=[5]
        )
        assert len(handles) == 3
        for handle in handles:
            async with handle:
                await handle.fetch()
                assert 5 in handle.captures
                assert len(handle.captures[5]) > 0

        # === Test Chat Generation ===

        messages = [{"role": "user", "content": "What is 2+2?"}]
        responses, handles = await model.chat(messages, max_tokens=30, temperature=0.0)
        assert len(responses) == 1
        assert len(responses[0].generated) > 0

        # Chat with steering
        direction = torch.randn(model.hidden_size)
        steering = SteeringSpec.simple_add(layer=5, vector=direction, scale=1.0)
        responses, handles = await model.chat(
            [{"role": "user", "content": "Tell me a story"}],
            max_tokens=30, temperature=0.0, steering_spec=steering
        )
        assert len(responses[0].generated) > 0

        # Chat with capture
        responses, handles = await model.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=10, temperature=0.0, capture_layers=[5]
        )
        assert handles is not None
        async with handles[0] as handle:
            await handle.fetch()
            assert 5 in handle.captures
            assert handle.message_boundaries is not None

        # === Test Concurrent Generation ===

        async def gen_task(prompt: str, task_id: int):
            texts, _ = await model.generate([prompt], max_tokens=20, temperature=0.0)
            return task_id, texts[0]

        prompts = ["The sun is", "Water is", "Fire is", "Earth is"]
        tasks = [gen_task(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        for task_id, text in results:
            assert len(text) > 0

        # Concurrent with different steering configs
        async def gen_steered(prompt: str, steering, task_id: int):
            texts, _ = await model.generate(
                [prompt], max_tokens=20, temperature=0.0, steering_spec=steering
            )
            return task_id, texts[0]

        v1 = torch.randn(model.hidden_size)
        v2 = torch.randn(model.hidden_size)
        steering1 = SteeringSpec.simple_add(layer=5, vector=v1, scale=2.0)
        steering2 = SteeringSpec.simple_ablation(layer=7, vector=v2, scale=0.0)

        tasks = [
            gen_steered("The answer is", None, 0),
            gen_steered("The answer is", steering1, 1),
            gen_steered("The answer is", steering2, 2),
            gen_steered("The answer is", steering1, 3),
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 4
        for task_id, text in results:
            assert len(text) > 0

        # === Test Model Properties ===

        assert model.hidden_size > 0
        assert isinstance(model.hidden_size, int)
        assert model.layer_count > 0
        assert isinstance(model.layer_count, int)
        assert model.model_name == MODEL_NAME
        tokenizer = model.tokenizer
        assert tokenizer is not None
        tokens = tokenizer.encode("Hello world")
        assert len(tokens) > 0

        # === Test Input Validation ===

        # Invalid layer index should raise
        direction = torch.randn(model.hidden_size)
        steering = SteeringSpec.simple_add(layer=999, vector=direction, scale=1.0)
        with pytest.raises(ValueError, match="out of range"):
            await model.generate(["Test"], max_tokens=10, steering_spec=steering)

    finally:
        # Cleanup
        del model
        torch.cuda.empty_cache()
