import pytest
import torch
from fastapi.testclient import TestClient
from neuronpedia_inference.shared import Model
from neuronpedia_inference_client.models.np_steer_chat_message import NPSteerChatMessage
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_vector import NPSteerVector
from neuronpedia_inference_client.models.steer_completion_chat_post200_response import (
    SteerCompletionChatPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_chat_post_request import (
    SteerCompletionChatPostRequest,
)
from transformer_lens import HookedTransformer

from tests.conftest import (
    FREQ_PENALTY,
    MODEL_ID,
    N_COMPLETION_TOKENS,
    SAE_SELECTED_SOURCES,
    SEED,
    STEER_FEATURE_INDEX,
    STEER_SPECIAL_TOKENS,
    STRENGTH,
    STRENGTH_MULTIPLIER,
    TEMPERATURE,
    TEST_PROMPT,
    X_SECRET_KEY,
)
from tests.utils.assertions import assert_deterministic_output_match

ENDPOINT = "/v1/steer/completion-chat"

TEST_STEER_FEATURE = NPSteerFeature(
    model=MODEL_ID,
    source=SAE_SELECTED_SOURCES[0],
    index=STEER_FEATURE_INDEX,
    strength=STRENGTH,
)

TEST_STEER_VECTOR = NPSteerVector(
    steering_vector=[1000.0] * 768,
    strength=STRENGTH,
    hook="blocks.7.hook_resid_post",
)


def _make_additive_feature_request() -> SteerCompletionChatPostRequest:
    return SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=TEST_PROMPT, role="user")],
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[TEST_STEER_FEATURE],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )


def _patch_generate_stream_cache(
    monkeypatch,
    *,
    use_past_kv_cache: bool,
) -> None:
    model = Model.get_instance()
    assert isinstance(model, HookedTransformer)
    _ensure_generate_stream_compat(model)
    original_generate_stream = getattr(model, "generate_stream")

    def wrapped_generate_stream(*args, **kwargs):
        kwargs["use_past_kv_cache"] = use_past_kv_cache
        kwargs["do_sample"] = False
        return original_generate_stream(*args, **kwargs)

    monkeypatch.setattr(model, "generate_stream", wrapped_generate_stream, raising=False)


def _has_native_generate_stream(model: HookedTransformer) -> bool:
    return callable(getattr(type(model), "generate_stream", None))


def _ensure_generate_stream_compat(model: HookedTransformer) -> None:
    if hasattr(model, "generate_stream"):
        return

    def generate_stream_compat(*args, **kwargs):
        input_tokens = kwargs.pop("input")
        max_new_tokens = kwargs.pop("max_new_tokens")
        stop_at_eos = kwargs.pop("stop_at_eos", True)
        do_sample = kwargs.pop("do_sample", True)
        temperature = kwargs.pop("temperature", 1.0)
        freq_penalty = kwargs.pop("freq_penalty", 0.0)
        use_past_kv_cache = kwargs.pop("use_past_kv_cache", True)
        return_logits = kwargs.pop("return_logits", False)
        kwargs.pop("max_tokens_per_yield", None)

        generated = model.generate(
            input=input_tokens,
            max_new_tokens=max_new_tokens,
            stop_at_eos=stop_at_eos,
            do_sample=do_sample,
            temperature=temperature,
            freq_penalty=freq_penalty,
            use_past_kv_cache=use_past_kv_cache,
            return_type="tokens",
            verbose=False,
        )
        logits = None
        if return_logits:
            with torch.no_grad():
                logits = model(generated[:, -1:].clone())
        yield generated, logits

    setattr(model, "generate_stream", generate_stream_compat)


def _run_chat_request(
    client: TestClient,
    request: SteerCompletionChatPostRequest,
) -> dict[NPSteerType, str]:
    model = Model.get_instance()
    assert isinstance(model, HookedTransformer)
    _ensure_generate_stream_compat(model)
    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionChatPost200Response(**data)
    return {output.type: output.raw for output in response_model.outputs}


def test_completion_chat_steered_with_features_additive(client: TestClient):
    """
    Test steering using features with additive method for chat completion.
    """
    outputs_by_type = _run_chat_request(client, _make_additive_feature_request())

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should contain some completion text
    assert len(outputs_by_type[NPSteerType.STEERED]) > 0
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > 0

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n\n<|im_start|>user\n"
    expected_default_output = "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n<|"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_chat_feature_additive_cache_parity(client: TestClient, monkeypatch):
    """
    Cache should not change deterministic chat steering outputs.
    """
    model = Model.get_instance()
    assert isinstance(model, HookedTransformer)
    if not _has_native_generate_stream(model):
        pytest.skip("cache parity test requires a native generate_stream implementation")

    with monkeypatch.context() as cache_on_patch:
        _patch_generate_stream_cache(cache_on_patch, use_past_kv_cache=True)
        cached_outputs = _run_chat_request(client, _make_additive_feature_request())

    with monkeypatch.context() as cache_off_patch:
        _patch_generate_stream_cache(cache_off_patch, use_past_kv_cache=False)
        uncached_outputs = _run_chat_request(client, _make_additive_feature_request())

    assert_deterministic_output_match(
        cached_outputs[NPSteerType.DEFAULT],
        uncached_outputs[NPSteerType.DEFAULT],
        left_label="cached default output",
        right_label="uncached default output",
    )
    assert_deterministic_output_match(
        cached_outputs[NPSteerType.STEERED],
        uncached_outputs[NPSteerType.STEERED],
        left_label="cached steered output",
        right_label="uncached steered output",
    )


def test_completion_chat_steered_with_vectors_additive(client: TestClient):
    """
    Test steering using vectors with additive method for chat completion.
    """
    request = SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=TEST_PROMPT, role="user")],
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        vectors=[TEST_STEER_VECTOR],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionChatPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.raw for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should contain some completion text
    assert len(outputs_by_type[NPSteerType.STEERED]) > 0
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > 0

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = (
        "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n!!!!!!!!!!"
    )
    expected_default_output = "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n<|"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_chat_steered_with_features_orthogonal(client: TestClient):
    """
    Test steering using features with orthogonal decomposition method for chat completion.
    """
    request = SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=TEST_PROMPT, role="user")],
        model=MODEL_ID,
        steer_method=NPSteerMethod.ORTHOGONAL_DECOMP,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[TEST_STEER_FEATURE],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionChatPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.raw for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should contain some completion text
    assert len(outputs_by_type[NPSteerType.STEERED]) > 0
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > 0

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n (?, Asahi, Asahi, Asahi,"
    expected_default_output = "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n<|"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_chat_token_limit_exceeded(client: TestClient):
    """
    Test handling of a chat prompt that exceeds the token limit.
    """
    long_content = "This is a test message. " * 1000
    request = SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=long_content, role="user")],
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        features=[TEST_STEER_FEATURE],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Text too long" in data["error"]
    assert "tokens, max is" in data["error"]


def test_completion_chat_invalid_request_no_features_or_vectors(client: TestClient):
    """
    Test error handling when neither features nor vectors are provided.
    """
    request = SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=TEST_PROMPT, role="user")],
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400
    data = response.json()
    assert "exactly one of features or vectors must be provided" in data["error"]


def test_completion_chat_invalid_request_both_features_and_vectors(client: TestClient):
    """
    Test error handling when both features and vectors are provided.
    """
    request = SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=TEST_PROMPT, role="user")],
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        features=[TEST_STEER_FEATURE],
        vectors=[TEST_STEER_VECTOR],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400
    data = response.json()
    assert "exactly one of features or vectors must be provided" in data["error"]
