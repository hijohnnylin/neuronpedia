import pytest
from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_topk_by_token_post200_response import (
    ActivationTopkByTokenPost200Response,
)
from neuronpedia_inference_client.models.activation_topk_by_token_post_request import (
    ActivationTopkByTokenPostRequest,
)

from tests.conftest import (
    ABS_TOLERANCE,
    BOS_TOKEN_STR,
    INVALID_SAE_SOURCE,
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    TEST_PROMPT,
    X_SECRET_KEY,
)

# Test specific constants
TOP_K = 3
ENDPOINT = "/v1/activation/topk-by-token"


def test_activation_topk_by_token_basic(client: TestClient):
    """
    Test basic functionality of the /activation/topk-by-token endpoint with a simple request.
    """
    # Test setup
    request = ActivationTopkByTokenPostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        top_k=TOP_K,
        ignore_bos=True,
    )

    # Expected response data
    expected_tokens = ["Hello", ",", " world", "!"]
    expected_features_by_token = {
        "Hello": [
            (16653, 46.481327056884766),
            (13715, 43.21952438354492),
            (20750, 22.139848709106445),
        ],
        ",": [
            (2494, 33.579681396484375),
            (1457, 19.781414031982422),
            (15348, 14.109557151794434),
        ],
        " world": [
            (22763, 26.626646041870117),
            (17362, 22.529861450195312),
            (2494, 15.487942695617676),
        ],
        "!": [
            (13413, 25.002201080322266),
            (21841, 17.194128036499023),
            (2494, 15.784616470336914),
        ],
    }

    # Make request
    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    # Basic response validation
    assert response.status_code == 200
    data = response.json()
    response_model = ActivationTopkByTokenPost200Response(**data)

    # Validate tokens match expected
    assert (
        response_model.tokens == expected_tokens
    ), "Tokens don't match expected values"

    # Validate we have results for each token
    assert len(response_model.results) == len(
        expected_tokens
    ), "Number of results doesn't match number of tokens"

    # Validate each token's top features
    for _, (token, result) in enumerate(zip(expected_tokens, response_model.results)):
        expected_features = expected_features_by_token[token]

        # Check we have the right number of features
        assert (
            len(result.top_features) == TOP_K
        ), f"Token '{token}' doesn't have {TOP_K} features"

        # Check each feature matches expected values
        for j, (expected_idx, expected_val) in enumerate(expected_features):
            actual_feature = result.top_features[j]
            assert (
                actual_feature.feature_index == expected_idx
            ), f"Token '{token}' feature {j}: expected index {expected_idx}, got {actual_feature.feature_index}"
            assert (
                pytest.approx(actual_feature.activation_value, abs=ABS_TOLERANCE)
                == expected_val
            ), f"Token '{token}' feature {j}: expected value {expected_val}, got {actual_feature.activation_value}"


def test_activation_topk_by_token_with_bos(client: TestClient):
    """
    Test the endpoint with ignore_bos=False to verify BOS token handling.
    """
    # Test setup
    request = ActivationTopkByTokenPostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        top_k=TOP_K,
        ignore_bos=False,
    )

    # Expected response data (including BOS token)
    expected_tokens = [BOS_TOKEN_STR, "Hello", ",", " world", "!"]
    expected_features_by_token = {
        BOS_TOKEN_STR: [
            (9663, 941.2152099609375),
            (14519, 631.7471923828125),
            (8598, 608.943603515625),
        ],
        "Hello": [
            (16653, 46.481327056884766),
            (13715, 43.21952438354492),
            (20750, 22.139848709106445),
        ],
        ",": [
            (2494, 33.579681396484375),
            (1457, 19.781414031982422),
            (15348, 14.109557151794434),
        ],
        " world": [
            (22763, 26.626646041870117),
            (17362, 22.529861450195312),
            (2494, 15.487942695617676),
        ],
        "!": [
            (13413, 25.002201080322266),
            (21841, 17.194128036499023),
            (2494, 15.784616470336914),
        ],
    }

    # Make request
    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    # Basic response validation
    assert response.status_code == 200
    data = response.json()
    response_model = ActivationTopkByTokenPost200Response(**data)

    # Verify BOS token is included at the start
    assert response_model.tokens[0] == BOS_TOKEN_STR, "BOS token not found at start"

    # Validate tokens match expected (including BOS)
    assert (
        response_model.tokens == expected_tokens
    ), "Tokens don't match expected values"

    # Validate we have results for each token
    assert len(response_model.results) == len(
        expected_tokens
    ), "Number of results doesn't match number of tokens"

    # Validate each token's top features
    for _, (token, result) in enumerate(zip(expected_tokens, response_model.results)):
        expected_features = expected_features_by_token[token]

        # Check we have the right number of features
        assert (
            len(result.top_features) == TOP_K
        ), f"Token '{token}' doesn't have {TOP_K} features"

        # Check each feature matches expected values
        for j, (expected_idx, expected_val) in enumerate(expected_features):
            actual_feature = result.top_features[j]
            assert (
                actual_feature.feature_index == expected_idx
            ), f"Token '{token}' feature {j}: expected index {expected_idx}, got {actual_feature.feature_index}"
            assert (
                pytest.approx(actual_feature.activation_value, abs=ABS_TOLERANCE)
                == expected_val
            ), f"Token '{token}' feature {j}: expected value {expected_val}, got {actual_feature.activation_value}"


def test_activation_topk_by_token_invalid_source(client: TestClient):
    """
    Test error handling for an invalid source.
    """
    request = ActivationTopkByTokenPostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source=INVALID_SAE_SOURCE,
        top_k=TOP_K,
        ignore_bos=True,
    )

    with pytest.raises(AssertionError) as excinfo:
        client.post(
            ENDPOINT,
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

    assert f"Found 0 entries when searching for {MODEL_ID}/{INVALID_SAE_SOURCE}" in str(
        excinfo.value
    )


def test_activation_topk_by_token_long_prompt(client: TestClient):
    """
    Test handling of a prompt that exceeds the token limit.
    """
    # Create a very long prompt that will exceed the token limit
    long_prompt = "This is a test prompt. " * 1000  # Repeat the prompt 1000 times

    request = ActivationTopkByTokenPostRequest(
        prompt=long_prompt,
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        top_k=TOP_K,
        ignore_bos=True,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 400
    data = response.json()

    expected_error_message = "Text too long: 6002 tokens, max is 500"
    assert data["error"] == expected_error_message
