import math

from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_single_post200_response import (
    ActivationSinglePost200Response,
)
from neuronpedia_inference_client.models.activation_single_post_request import (
    ActivationSinglePostRequest,
)

from tests.conftest import (
    BOS_TOKEN_STR,
    DIM_MODEL,
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    TEST_PROMPT,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/activation/single"


def test_activation_single_with_source_and_index(client: TestClient):
    """
    Test the /activation/single endpoint with source and index parameters.

    This test verifies:
    - API returns 200 and valid response structure
    - Activation values for a specific SAE feature are returned
    - Values are finite and non-negative
    - max_value and max_value_index are consistent with values
    - Tokenization is correct
    """
    feature_index = "0"  # Request activations for feature 0
    request = ActivationSinglePostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        index=feature_index,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200

    # Validate response structure with Pydantic model
    data = response.json()
    response_model = ActivationSinglePost200Response(**data)

    # Check tokenization (this endpoint doesn't prepend BOS)
    expected_tokens = ["Hello", ",", " world", "!"]
    assert (
        response_model.tokens == expected_tokens
    ), f"Tokenization mismatch: expected {expected_tokens}, got {response_model.tokens}"

    activation = response_model.activation

    # Values should match token count
    assert len(activation.values) == len(
        expected_tokens
    ), f"Expected {len(expected_tokens)} values, got {len(activation.values)}"

    # All values should be finite and non-negative
    for i, val in enumerate(activation.values):
        assert math.isfinite(val), f"Token {i}: non-finite value {val}"
        assert val >= 0, f"Token {i}: negative activation {val}"

    # max_value should equal the maximum of values
    computed_max = max(activation.values)
    assert (
        abs(activation.max_value - computed_max) < 1e-5
    ), f"max_value {activation.max_value} != max(values) {computed_max}"

    # max_value_index should point to the max value
    assert (
        activation.values[activation.max_value_index] == computed_max
    ), f"max_value_index {activation.max_value_index} doesn't point to max"


def test_activation_single_with_vector_and_hook(client: TestClient):
    """
    Test the /activation/single endpoint with custom vector and hook parameters.

    This test verifies:
    - API accepts custom steering vectors and hook points
    - Returns valid activation structure
    - Dot product activations are computed correctly (positive for aligned vectors)
    """
    # Create a test vector matching the residual stream dimension
    test_vector = [0.1] * DIM_MODEL
    test_hook = "blocks.0.hook_resid_post"

    request = ActivationSinglePostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        vector=test_vector,
        hook=test_hook,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200

    # Validate response structure
    data = response.json()
    response_model = ActivationSinglePost200Response(**data)

    # For custom vector+hook, BOS is not prepended
    expected_tokens = ["Hello", ",", " world", "!"]
    assert (
        response_model.tokens == expected_tokens
    ), f"Tokenization mismatch: expected {expected_tokens}, got {response_model.tokens}"

    activation = response_model.activation

    # Values should match token count
    assert len(activation.values) == len(
        expected_tokens
    ), f"Expected {len(expected_tokens)} values, got {len(activation.values)}"

    # All values should be finite
    for i, val in enumerate(activation.values):
        assert math.isfinite(val), f"Token {i}: non-finite value {val}"

    # With a uniform positive vector, we expect positive dot products with most residuals
    # (This is a weak sanity check that the computation is happening)
    assert any(
        v > 0 for v in activation.values
    ), "Expected at least one positive activation for uniform positive vector"

    # max_value should equal the maximum of values
    computed_max = max(activation.values)
    assert (
        abs(activation.max_value - computed_max) < 1e-5
    ), f"max_value {activation.max_value} != max(values) {computed_max}"

    # max_value_index should point to the max value
    assert (
        activation.values[activation.max_value_index] == computed_max
    ), f"max_value_index {activation.max_value_index} doesn't point to max"
