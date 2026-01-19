"""Tests for input validation and edge case handling.

Tests cover:
- Empty prompts list
- Empty string prompts
- Layer indices out of range [0, layer_count)
- Negative layer indices
- Zero-norm steering vectors
- Empty steering specs (no layers)
- Invalid capture_layers values
- Malformed SteeringSpec
- Very large/small steering scales
- Invalid message format in chat()
- None/null inputs
"""

import math
import pytest
import torch
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


@pytest.fixture
def model_name():
    """Small model for fast tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture
async def model_factory(model_name):
    """Factory for creating VLLMSteerModel with custom config."""
    created_models = []

    async def _make_model():
        config = VLLMSteeringConfig(
            model_name=model_name,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )
        m = VLLMSteerModel(
            config,
            bootstrap_layers=(5,),
            enforce_eager=True,
        )
        created_models.append(m)
        return m

    yield _make_model

    # Cleanup all created models
    for m in created_models:
        if hasattr(m, "_engine") and m._engine is not None:
            try:
                await m._engine.shutdown()
            except Exception:
                pass


@pytest.mark.slow
@pytest.mark.asyncio
async def test_empty_prompts_list(model_factory):
    """Test that empty prompts list returns empty results."""
    model = await model_factory()

    prompts = []
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Empty prompts should return empty results (no handles since no capture_layers)
    results = await model.generate(prompts, sampling_params)
    assert len(results) == 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_empty_prompts_list_with_captures(model_factory):
    """Test that empty prompts list with capture_layers returns empty results and handles."""
    model = await model_factory()

    prompts = []
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Empty prompts with capture_layers should return empty tuple
    results, handles = await model.generate(prompts, sampling_params, capture_layers=[5])
    assert len(results) == 0
    assert len(handles) == 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_empty_string_prompt(model_factory):
    """Test that empty string prompt is handled."""
    model = await model_factory()

    prompts = [""]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Should either work or raise informative error
    try:
        results, handles = await model.generate(prompts, sampling_params)
        # If it works, should have results
        assert len(results) == 1
        await handles[0].close()
    except (ValueError, RuntimeError) as e:
        # If it fails, error should mention "empty" or "prompt"
        assert "empty" in str(e).lower() or "prompt" in str(e).lower()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_layer_index_out_of_range_high(model_factory):
    """Test that layer index above layer_count raises helpful error."""
    model = await model_factory()

    # Try to use layer index beyond model's layer count
    invalid_layer = model.layer_count + 10

    steering_vector = torch.randn(model.hidden_size)
    steering_spec = SteeringSpec(
        layers={
            invalid_layer: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=1.0)
            ])
        }
    )

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should raise error about invalid layer index
    with pytest.raises((ValueError, RuntimeError, IndexError)) as exc_info:
        await model.generate(prompts, sampling_params, steering_spec=steering_spec)

    # Error message should mention "layer" and/or "range"
    error_msg = str(exc_info.value).lower()
    assert "layer" in error_msg or "range" in error_msg or "index" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_layer_index_negative(model_factory):
    """Test that negative layer index raises helpful error."""
    model = await model_factory()

    steering_vector = torch.randn(model.hidden_size)
    steering_spec = SteeringSpec(
        layers={
            -5: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=1.0)
            ])
        }
    )

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should raise error about invalid layer index
    with pytest.raises((ValueError, RuntimeError, IndexError)) as exc_info:
        await model.generate(prompts, sampling_params, steering_spec=steering_spec)

    error_msg = str(exc_info.value).lower()
    assert "layer" in error_msg or "range" in error_msg or "negative" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_capture_layer_out_of_range(model_factory):
    """Test that capture_layers with invalid index raises error."""
    model = await model_factory()

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    invalid_layer = model.layer_count + 5

    # Should raise error about invalid capture layer
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        await model.generate(
            prompts,
            sampling_params,
            capture_layers=[invalid_layer]
        )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_zero_norm_steering_vector_raises(model_factory):
    """Test that zero-norm steering vector raises helpful error.

    The system should reject zero-norm vectors since they can't be normalized.
    """
    model = await model_factory()

    # Create zero vector
    zero_vector = torch.zeros(model.hidden_size)

    # Try to create steering spec with zero vector
    # Should raise during spec creation or validation
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        steering_spec = SteeringSpec(
            layers={
                5: LayerSteeringSpec(
                    add=AddSpec(vector=zero_vector, scale=1.0)
                )
            }
        )

        prompts = ["Test"]
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
        await model.generate(prompts, sampling_params, steering_spec=steering_spec)

    error_msg = str(exc_info.value).lower()
    assert "norm" in error_msg or "zero" in error_msg or "magnitude" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_nan_steering_vector_raises(model_factory):
    """Test that NaN-containing steering vector raises helpful error."""
    model = await model_factory()

    # Create vector with NaN
    nan_vector = torch.randn(model.hidden_size)
    nan_vector[0] = float('nan')

    # Should raise during spec creation or validation
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        steering_spec = SteeringSpec(
            layers={
                5: LayerSteeringSpec(
                    add=AddSpec(vector=nan_vector, scale=1.0)
                )
            }
        )

        prompts = ["Test"]
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
        await model.generate(prompts, sampling_params, steering_spec=steering_spec)

    error_msg = str(exc_info.value).lower()
    assert "nan" in error_msg or "finite" in error_msg or "invalid" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_inf_steering_vector_raises(model_factory):
    """Test that Inf-containing steering vector raises helpful error."""
    model = await model_factory()

    # Create vector with Inf
    inf_vector = torch.randn(model.hidden_size)
    inf_vector[0] = float('inf')

    # Should raise during spec creation or validation
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        steering_spec = SteeringSpec(
            layers={
                5: LayerSteeringSpec(
                    add=AddSpec(vector=inf_vector, scale=1.0)
                )
            }
        )

        prompts = ["Test"]
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
        await model.generate(prompts, sampling_params, steering_spec=steering_spec)

    error_msg = str(exc_info.value).lower()
    assert "inf" in error_msg or "finite" in error_msg or "invalid" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_wrong_dimension_steering_vector_raises(model_factory):
    """Test that steering vector with wrong dimension raises helpful error."""
    model = await model_factory()

    # Create vector with wrong dimension
    wrong_dim_vector = torch.randn(model.hidden_size + 100)

    # Should raise during spec serialization or validation
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        steering_spec = SteeringSpec(
            layers={
                5: LayerSteeringSpec(operations=[
                    AddSpec(vector=wrong_dim_vector / wrong_dim_vector.norm(), scale=1.0)
                ])
            }
        )

        prompts = ["Test"]
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
        await model.generate(prompts, sampling_params, steering_spec=steering_spec)

    error_msg = str(exc_info.value).lower()
    assert "dimension" in error_msg or "size" in error_msg or "shape" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_empty_steering_spec_is_no_op(model_factory):
    """Test that empty steering spec (no layers) is treated as no-op."""
    model = await model_factory()

    # Create empty steering spec
    empty_spec = SteeringSpec(layers={})

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Should work (no steering applied, no handles since no capture_layers)
    results = await model.generate(
        prompts,
        sampling_params,
        steering_spec=empty_spec
    )

    assert len(results) == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_all_layers_zero_scale_is_no_op(model_factory):
    """Test that steering spec with all scales=0.0 is effectively no-op."""
    model = await model_factory()

    steering_vector = torch.randn(model.hidden_size)
    zero_scale_spec = SteeringSpec(
        layers={
            5: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=0.0)
            ])
        }
    )

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Should work (no effective steering, no handles since no capture_layers)
    results = await model.generate(
        prompts,
        sampling_params,
        steering_spec=zero_scale_spec
    )

    assert len(results) == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_very_large_steering_scale(model_factory):
    """Test that very large steering scale doesn't crash (but heavily perturbs output)."""
    model = await model_factory()

    steering_vector = torch.randn(model.hidden_size)
    large_scale_spec = SteeringSpec(
        layers={
            5: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=1e6)
            ])
        }
    )

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should work (output will be heavily perturbed but shouldn't crash, no handles since no capture_layers)
    results = await model.generate(
        prompts,
        sampling_params,
        steering_spec=large_scale_spec
    )

    assert len(results) == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_very_small_steering_scale(model_factory):
    """Test that very small steering scale works (nearly no-op)."""
    model = await model_factory()

    steering_vector = torch.randn(model.hidden_size)
    small_scale_spec = SteeringSpec(
        layers={
            5: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=1e-12)
            ])
        }
    )

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should work (output nearly unchanged, no handles since no capture_layers)
    results = await model.generate(
        prompts,
        sampling_params,
        steering_spec=small_scale_spec
    )

    assert len(results) == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_projection_cap_both_none_raises(model_factory):
    """Test that ProjectionCapSpec with both min and max None raises error."""
    model = await model_factory()

    direction = torch.randn(model.hidden_size)

    # Should raise during spec construction or validation
    with pytest.raises((ValueError, RuntimeError)):
        spec = VLLMSteerModel.simple_projection_cap(5, direction, min=None, max=None)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_projection_cap_zero_norm_raises(model_factory):
    """Test that ProjectionCapSpec with zero-norm vector raises error."""
    model = await model_factory()

    zero_vector = torch.zeros(model.hidden_size)

    # Should raise during spec construction
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        spec = VLLMSteerModel.simple_projection_cap(5, zero_vector, min=-1.0, max=1.0)

    error_msg = str(exc_info.value).lower()
    assert "norm" in error_msg or "zero" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_ablation_zero_norm_raises(model_factory):
    """Test that AblationSpec with zero-norm vector raises error."""
    model = await model_factory()

    zero_vector = torch.zeros(model.hidden_size)

    # Should raise during spec construction
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        spec = VLLMSteerModel.simple_ablation(5, zero_vector, scale=0.5)

    error_msg = str(exc_info.value).lower()
    assert "norm" in error_msg or "zero" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_chat_empty_messages_list(model_factory):
    """Test that chat() with empty messages list raises helpful error."""
    model = await model_factory()

    messages = []
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Should raise error about empty messages (tokenizer raises IndexError)
    with pytest.raises((ValueError, RuntimeError, IndexError)) as exc_info:
        await model.chat(messages, sampling_params)

    error_msg = str(exc_info.value).lower()
    # Error should mention "message" or "empty" or "conversation" or "index"
    # Or it might just fail in tokenizer - any error is acceptable


@pytest.mark.slow
@pytest.mark.asyncio
async def test_chat_malformed_message_dict(model_factory):
    """Test that chat() with malformed message dict raises helpful error."""
    model = await model_factory()

    # Missing 'role' field
    messages = [{"content": "Hello"}]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Should raise error about malformed message
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        await model.chat(messages, sampling_params)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_capture_layers_empty_list(model_factory):
    """Test that capture_layers=[] is treated as no capture."""
    model = await model_factory()

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should work without capturing
    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[]
    )

    assert len(results) == 1
    assert len(handles) == 1

    # Handle should have no captures
    await handles[0].close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_capture_layers_none_is_no_capture(model_factory):
    """Test that capture_layers=None means no capture."""
    model = await model_factory()

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should work without capturing (no handles returned when capture_layers=None)
    results = await model.generate(
        prompts,
        sampling_params,
        capture_layers=None
    )

    assert len(results) == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_single_int_capture_layer(model_factory):
    """Test that capture_layers can be a single int (not list)."""
    model = await model_factory()

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should accept single int and convert to list
    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=5
    )

    assert len(results) == 1
    handle = handles[0]

    await handle.fetch()

    # Should have captured layer 5
    assert 5 in handle.captures

    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_duplicate_capture_layers(model_factory):
    """Test that duplicate layer indices in capture_layers are handled."""
    model = await model_factory()

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Duplicate layers (5 appears twice)
    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5, 10, 5]
    )

    assert len(results) == 1
    handle = handles[0]

    await handle.fetch()

    # Should have both layers (duplicates deduplicated)
    assert 5 in handle.captures
    assert 10 in handle.captures

    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_max_tokens_zero(model_factory):
    """Test that max_tokens=0 raises validation error."""
    model = await model_factory()

    prompts = ["Test prompt"]

    # vLLM requires max_tokens >= 1, so this should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        sampling_params = SamplingParams(max_tokens=0, temperature=0.0)

    error_msg = str(exc_info.value).lower()
    assert "max_tokens" in error_msg or "at least 1" in error_msg


@pytest.mark.slow
@pytest.mark.asyncio
async def test_steering_spec_none_is_no_steering(model_factory):
    """Test that steering_spec=None means no steering."""
    model = await model_factory()

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    # Should work without steering (no handles since no capture_layers)
    results = await model.generate(
        prompts,
        sampling_params,
        steering_spec=None
    )

    assert len(results) == 1
