import math

from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_all_post200_response import (
    ActivationAllPost200Response,
)
from neuronpedia_inference_client.models.activation_all_post_request import (
    ActivationAllPostRequest,
)

from tests.conftest import (
    BOS_TOKEN_STR,
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    SAE_SOURCE_SET,
    TEST_PROMPT,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/activation/all"


def test_activation_all(client: TestClient):
    """
    Test the /activation/all endpoint returns valid SAE feature activations.

    This test verifies:
    - API returns 200 and valid response structure
    - Correct number of activations returned
    - Each activation has valid structure and sensible values
    - Tokenization matches expected behavior
    - Results are sorted by max activation value (descending)
    """
    num_results = 5
    request = ActivationAllPostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source_set=SAE_SOURCE_SET,
        selected_sources=SAE_SELECTED_SOURCES,
        sort_by_token_indexes=[],
        num_results=num_results,
        ignore_bos=True,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200

    # Validate response structure with Pydantic model
    data = response.json()
    response_model = ActivationAllPost200Response(**data)

    # Verify we got the requested number of activations
    assert len(response_model.activations) == num_results

    # Check tokenization is correct
    expected_tokens = [BOS_TOKEN_STR, "Hello", ",", " world", "!"]
    assert (
        response_model.tokens == expected_tokens
    ), f"Tokenization mismatch: expected {expected_tokens}, got {response_model.tokens}"

    # Verify each activation has valid structure and values
    prev_max_value = float("inf")
    for i, activation in enumerate(response_model.activations):
        # Source should match the requested SAE
        expected_source = SAE_SELECTED_SOURCES[0]
        assert (
            activation.source == expected_source
        ), f"Activation {i}: expected source '{expected_source}', got '{activation.source}'"

        # Feature index should be a valid non-negative integer
        assert (
            isinstance(activation.index, int) and activation.index >= 0
        ), f"Activation {i}: invalid feature index {activation.index}"

        # Values should match token count and contain no NaN/inf
        assert (
            len(activation.values) == len(expected_tokens)
        ), f"Activation {i}: expected {len(expected_tokens)} values, got {len(activation.values)}"
        for j, val in enumerate(activation.values):
            assert math.isfinite(
                val
            ), f"Activation {i}, token {j}: non-finite value {val}"
            assert val >= 0, f"Activation {i}, token {j}: negative activation {val}"

        # max_value should equal the maximum of values
        computed_max = max(activation.values)
        assert (
            abs(activation.max_value - computed_max) < 1e-5
        ), f"Activation {i}: max_value {activation.max_value} != max(values) {computed_max}"

        # max_value_index should point to the max value
        assert (
            activation.values[activation.max_value_index] == computed_max
        ), f"Activation {i}: max_value_index {activation.max_value_index} doesn't point to max"

        # Results should be sorted by max_value descending
        assert activation.max_value <= prev_max_value, (
            f"Activation {i}: results not sorted descending by max_value "
            f"({activation.max_value} > {prev_max_value})"
        )
        prev_max_value = activation.max_value

    # Top activation should have a reasonably high value (sanity check that SAE is working)
    top_activation = response_model.activations[0]
    assert (
        top_activation.max_value > 1.0
    ), f"Top activation value {top_activation.max_value} is suspiciously low"
