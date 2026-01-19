"""Tests for vLLM chat() API activation capture with message boundaries."""

from __future__ import annotations

import asyncio

import pytest
import torch
from vllm import SamplingParams

from chatspace.generation import ChatResponse, VLLMSteerModel, VLLMSteeringConfig


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_basic_activation_capture():
    """Test basic chat() with activation capture."""
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

    # Single-turn conversation
    messages = [{"role": "user", "content": "What is 2+2?"}]
    sampling = SamplingParams(temperature=0.0, max_tokens=5)

    # Generate with capture
    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    # Verify response and handle
    assert len(responses) == 1, "Expected one response"
    assert isinstance(responses[0], ChatResponse), "Response should be a ChatResponse"
    assert len(responses[0].full_text()) > 0, "Response should have text"
    assert len(handles) == 1, "Expected one capture handle"

    handle = handles[0]
    assert handle.request_id, "Expected non-empty request ID"
    assert target_layer in handle.layer_indices, f"Expected layer {target_layer} in handle"

    # Fetch captures
    await handle.fetch()
    captures_dict = handle.captures

    assert target_layer in captures_dict, f"Layer {target_layer} not in captured states"
    layer_captures = captures_dict[target_layer]
    assert len(layer_captures) > 0, "Expected at least one capture"

    hidden_state = layer_captures[0]["hidden"]
    assert hidden_state.shape[-1] == model.hidden_size, "Hidden dimension should match model"
    assert hidden_state.dtype in (torch.float16, torch.bfloat16, torch.float32), "Expected float dtype"

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_message_boundaries_single_turn():
    """Test that message boundaries are correctly computed for a single-turn conversation."""
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

    messages = [{"role": "user", "content": "Hello"}]
    sampling = SamplingParams(temperature=0.0, max_tokens=3)

    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    await asyncio.gather(*[h.fetch() for h in handles])
    handle = handles[0]

    # Verify message boundaries exist
    assert handle.message_boundaries is not None, "Message boundaries should be available"
    assert len(handle.message_boundaries) == 1, "Expected one message boundary"

    boundary = handle.message_boundaries[0]
    assert boundary.role == "user", f"Expected role 'user', got '{boundary.role}'"
    assert boundary.content == "Hello", f"Expected content 'Hello', got '{boundary.content}'"
    assert boundary.start_token == 0, "First message should start at token 0"
    assert boundary.end_token > boundary.start_token, "Message should have positive length"
    assert boundary.num_tokens == boundary.end_token - boundary.start_token

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_message_boundaries_multi_turn():
    """Test message boundaries for multi-turn conversations."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=256,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is its population?"},
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=10)

    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    await asyncio.gather(*[h.fetch() for h in handles])
    handle = handles[0]

    # Verify all messages have boundaries
    assert handle.message_boundaries is not None
    assert len(handle.message_boundaries) == 4, "Expected four message boundaries"

    # Check that boundaries are sequential and non-overlapping
    prev_end = 0
    for i, boundary in enumerate(handle.message_boundaries):
        assert boundary.start_token == prev_end, (
            f"Message {i} should start where previous ended. "
            f"Expected {prev_end}, got {boundary.start_token}"
        )
        assert boundary.end_token > boundary.start_token, f"Message {i} should have positive length"
        prev_end = boundary.end_token

    # Verify roles match
    expected_roles = ["system", "user", "assistant", "user"]
    actual_roles = [b.role for b in handle.message_boundaries]
    assert actual_roles == expected_roles, f"Roles mismatch: expected {expected_roles}, got {actual_roles}"

    # Verify content matches
    for i, (boundary, msg) in enumerate(zip(handle.message_boundaries, messages)):
        assert boundary.content == msg["content"], f"Message {i} content mismatch"

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_get_message_activations():
    """Test extracting activations for individual messages."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=256,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+3?"},
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=5)

    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    await asyncio.gather(*[h.fetch() for h in handles])
    handle = handles[0]

    # Extract activations for each message
    for msg_idx in range(len(handle.message_boundaries)):
        msg_acts = handle.get_message_activations(
            message_idx=msg_idx,
            layer_idx=target_layer,
        )

        # Verify shape
        assert msg_acts.ndim == 2, f"Message {msg_idx} activations should be 2D"
        assert msg_acts.shape[1] == model.hidden_size, "Hidden size should match model"

        # Verify length matches boundary
        boundary = handle.message_boundaries[msg_idx]
        expected_len = boundary.num_tokens
        assert msg_acts.shape[0] == expected_len, (
            f"Message {msg_idx} activation length mismatch. "
            f"Expected {expected_len}, got {msg_acts.shape[0]}"
        )

    # Verify that concatenating all message activations gives the full capture
    full_hidden = handle.captures[target_layer][0]["hidden"]
    all_message_acts = torch.cat([
        handle.get_message_activations(i, target_layer)
        for i in range(len(handle.message_boundaries))
    ], dim=0)

    # Should match the prefill portion (excluding generated tokens)
    expected_prompt_len = handle.message_boundaries[-1].end_token
    prompt_hidden = full_hidden[:expected_prompt_len]

    assert torch.allclose(all_message_acts, prompt_hidden, atol=1e-6), (
        "Concatenated message activations should match the prompt portion of full capture"
    )

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_get_message_activations_with_generated():
    """Test get_message_activations with include_generated=True."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=256,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    messages = [
        {"role": "user", "content": "Say hello"},
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=5)

    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    await asyncio.gather(*[h.fetch() for h in handles])
    handle = handles[0]

    # Get just the message activations (no generated tokens)
    msg_acts_only = handle.get_message_activations(
        message_idx=0,
        layer_idx=target_layer,
        include_generated=False,
    )

    # Get message + generated activations
    msg_acts_with_gen = handle.get_message_activations(
        message_idx=0,
        layer_idx=target_layer,
        include_generated=True,
    )

    # With generated should be longer
    assert msg_acts_with_gen.shape[0] > msg_acts_only.shape[0], (
        "Activations with generated tokens should be longer"
    )

    # The prompt portion should match
    prompt_len = msg_acts_only.shape[0]
    assert torch.allclose(
        msg_acts_with_gen[:prompt_len],
        msg_acts_only,
        atol=1e-6
    ), "Prompt portion should match"

    # Full capture should match msg_acts_with_gen for a single-turn conversation
    full_hidden = handle.captures[target_layer][0]["hidden"]
    assert torch.allclose(msg_acts_with_gen, full_hidden, atol=1e-6), (
        "For single-turn, message + generated should equal full capture"
    )

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_batch_with_captures():
    """Test batch chat with activation capture."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=256,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Multiple conversations
    conversations = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Explain Python."}],
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ],
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=5)

    responses, handles = await model.chat(
        conversations,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    # Verify batch sizes match
    assert len(responses) == len(conversations), "Response count mismatch"
    assert len(handles) == len(conversations), "Handle count mismatch"

    # Fetch all captures at once
    await asyncio.gather(*[h.fetch() for h in handles])

    # Verify each handle has the correct number of message boundaries
    for i, (handle, conv) in enumerate(zip(handles, conversations)):
        assert handle.message_boundaries is not None, f"Handle {i} should have message boundaries"
        assert len(handle.message_boundaries) == len(conv), (
            f"Handle {i} message boundary count mismatch. "
            f"Expected {len(conv)}, got {len(handle.message_boundaries)}"
        )

        # Verify each message boundary matches the conversation
        for j, (boundary, msg) in enumerate(zip(handle.message_boundaries, conv)):
            assert boundary.role == msg["role"], f"Handle {i}, message {j} role mismatch"
            assert boundary.content == msg["content"], f"Handle {i}, message {j} content mismatch"

        # Verify captures exist
        assert target_layer in handle.captures, f"Handle {i} missing layer {target_layer}"
        hidden = handle.captures[target_layer][0]["hidden"]
        assert hidden.shape[-1] == model.hidden_size

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_multiple_layers_capture():
    """Test chat with multiple layer capture."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    layers = [1, 2, 3]

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=layers)
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    messages = [{"role": "user", "content": "Hello"}]
    sampling = SamplingParams(temperature=0.0, max_tokens=3)

    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=layers,
    )

    await asyncio.gather(*[h.fetch() for h in handles])
    handle = handles[0]

    # Verify all layers captured
    for layer_idx in layers:
        assert layer_idx in handle.captures, f"Layer {layer_idx} should have captures"

        # Verify get_message_activations works for all layers
        msg_acts = handle.get_message_activations(
            message_idx=0,
            layer_idx=layer_idx,
        )
        assert msg_acts.shape[1] == model.hidden_size
        assert msg_acts.shape[0] == handle.message_boundaries[0].num_tokens

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_without_capture_has_no_boundaries():
    """Test that chat without capture_layers doesn't return handles."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True)
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    messages = [{"role": "user", "content": "Hello"}]
    sampling = SamplingParams(temperature=0.0, max_tokens=3)

    # Chat without capture_layers
    result = await model.chat(
        messages,
        sampling_params=sampling,
    )

    # Should return just responses, not a tuple
    assert isinstance(result, list), "Without capture_layers, should return list of responses"
    assert len(result) == 1
    assert isinstance(result[0], ChatResponse), "Response should be ChatResponse"
    assert len(result[0].full_text()) > 0, "Response should have text"

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_get_message_activations_errors():
    """Test error handling for get_message_activations."""
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

    messages = [{"role": "user", "content": "Hello"}]
    sampling = SamplingParams(temperature=0.0, max_tokens=3)

    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    handle = handles[0]

    # Should error if captures not fetched
    with pytest.raises(RuntimeError, match="Captures not fetched yet"):
        handle.get_message_activations(0, target_layer)

    # Fetch captures
    await handle.fetch()

    # Should error on invalid message index
    with pytest.raises(ValueError, match="message_idx.*out of range"):
        handle.get_message_activations(999, target_layer)

    # Should error on invalid layer index
    with pytest.raises(ValueError, match="layer_idx.*not in captured layers"):
        handle.get_message_activations(0, 999)

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_compare_user_messages():
    """Test comparing activations between different user messages in a conversation."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=256,
    )

    target_layer = 4

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    messages = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What is 10*10?"},
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=5)

    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
    )

    await asyncio.gather(*[h.fetch() for h in handles])
    handle = handles[0]

    # Find user message indices
    user_msg_indices = [
        i for i, b in enumerate(handle.message_boundaries)
        if b.role == "user"
    ]
    assert len(user_msg_indices) == 2, "Expected two user messages"

    # Extract activations for each user message
    acts_1 = handle.get_message_activations(user_msg_indices[0], target_layer)
    acts_2 = handle.get_message_activations(user_msg_indices[1], target_layer)

    # Compute mean representations
    mean_1 = acts_1.mean(dim=0)
    mean_2 = acts_2.mean(dim=0)

    # They should be different (different questions)
    cos_sim = torch.nn.functional.cosine_similarity(
        mean_1.unsqueeze(0),
        mean_2.unsqueeze(0)
    ).item()

    # They should be somewhat similar (both math questions) but not identical
    assert 0.5 < cos_sim < 0.99, (
        f"Expected moderate similarity between related questions, got {cos_sim:.4f}"
    )

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_raw_output_with_captures():
    """Test chat with raw_output=True and activation capture."""
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

    messages = [{"role": "user", "content": "Hello"}]
    sampling = SamplingParams(temperature=0.0, max_tokens=3, logprobs=1)

    # Get raw output with captures
    outputs, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
        raw_output=True,
    )

    # Verify we get RequestOutput objects
    assert len(outputs) == 1
    # Check it has RequestOutput-like attributes
    assert hasattr(outputs[0], "outputs"), "Should have RequestOutput structure"
    assert hasattr(outputs[0], "prompt"), "Should have prompt field"

    # Verify captures still work
    await asyncio.gather(*[h.fetch() for h in handles])
    assert target_layer in handles[0].captures

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_assistant_prefill():
    """Test assistant response prefilling using continue_final_message."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True)
    except OSError as exc: 
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Test JSON format prefilling
    prefill_text = '{"answer": "'
    messages = [
        {"role": "user", "content": "What is 2+2? Answer in JSON format."},
        {"role": "assistant", "content": prefill_text},  # Partial assistant response
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=10)

    # Use chat_options to enable prefill mode
    responses = await model.chat(
        messages,
        sampling_params=sampling,
        chat_options={
            "add_generation_prompt": False,  # Required when continuing
            "continue_final_message": True,   # Enable prefill mode
        },
    )

    # Verify response structure
    assert len(responses) == 1, "Expected one response"
    response = responses[0]
    assert isinstance(response, ChatResponse), "Response should be a ChatResponse"

    # Verify prefill and generated separation
    assert response.has_prefill, "Response should have prefill"
    assert response.prefill == prefill_text, f"Prefill mismatch. Expected: {prefill_text!r}, got: {response.prefill!r}"
    assert len(response.generated) > 0, "Response should have generated text"

    # Verify full_text() combines both
    full_response = response.full_text()
    assert full_response.startswith('{"answer":'), (
        f"Full response should look like JSON. Got: {full_response}"
    )
    assert '"' in response.generated, "Generated text should contain at least a closing quote"

    # Test with reasoning block prefill
    reasoning_prefill = "<think>\n"
    messages_with_reasoning = [
        {"role": "user", "content": "Solve this step by step: What is 5*6?"},
        {"role": "assistant", "content": reasoning_prefill},
    ]

    responses_reasoning = await model.chat(
        messages_with_reasoning,
        sampling_params=sampling,
        chat_options={
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
    )

    # Verify reasoning response with prefill
    assert len(responses_reasoning) == 1
    reasoning_response = responses_reasoning[0]
    assert isinstance(reasoning_response, ChatResponse), "Should be ChatResponse"
    assert reasoning_response.has_prefill, "Should have reasoning prefill"
    assert reasoning_response.prefill == reasoning_prefill, "Prefill should match"
    assert len(reasoning_response.generated) > 0, "Should have generated reasoning content"

    # Verify full_text() starts with think tag
    full_reasoning = reasoning_response.full_text()
    assert full_reasoning.startswith("<think>"), "Should start with think tag"

    del model


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_chat_assistant_prefill_with_capture():
    """Test that message boundaries work correctly with assistant prefill."""
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

    prefill_text = "Sure, "
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": prefill_text},
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=5)

    # Generate with capture and prefill
    responses, handles = await model.chat(
        messages,
        sampling_params=sampling,
        capture_layers=target_layer,
        chat_options={
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
    )

    # Verify response with prefill
    assert len(responses) == 1
    response = responses[0]
    assert isinstance(response, ChatResponse), "Should be ChatResponse"
    assert response.has_prefill, "Should have prefill"
    assert response.prefill == prefill_text, "Prefill should match"
    assert len(response.generated) > 0, "Should have generated text"

    # Verify message boundaries
    await asyncio.gather(*[h.fetch() for h in handles])
    handle = handles[0]

    assert handle.message_boundaries is not None, "Should have message boundaries"
    assert len(handle.message_boundaries) == 2, "Expected two message boundaries"

    # Verify both messages are tracked
    assert handle.message_boundaries[0].role == "user"
    assert handle.message_boundaries[0].content == "Hello"
    assert handle.message_boundaries[1].role == "assistant"
    assert handle.message_boundaries[1].content == prefill_text

    # Verify activations can be extracted for both messages
    user_acts = handle.get_message_activations(0, target_layer)
    assistant_acts = handle.get_message_activations(1, target_layer)

    assert user_acts.shape[1] == model.hidden_size
    assert assistant_acts.shape[1] == model.hidden_size

    # Assistant prefill should have some tokens
    assert assistant_acts.shape[0] > 0, "Assistant prefill should have tokens"

    del model
