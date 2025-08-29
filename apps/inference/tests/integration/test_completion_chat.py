from fastapi.testclient import TestClient
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


def test_completion_chat_steered_with_features_additive(client: TestClient):
    """
    Test steering using features with additive method for chat completion.
    """
    request = SteerCompletionChatPostRequest(
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
    print("this is outputs_by_type[NPSteerType.STEERED]", outputs_by_type[NPSteerType.STEERED])

    expected_steered_output = "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n\n<|im_start|>user\n"
    expected_default_output = "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n<|"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


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
