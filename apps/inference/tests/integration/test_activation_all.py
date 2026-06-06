from fastapi.testclient import TestClient
import pytest
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
    Test basic functionality of the /activation/all endpoint with a simple request.
    """
    request = ActivationAllPostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source_set=SAE_SOURCE_SET,
        selected_sources=SAE_SELECTED_SOURCES,
        sort_by_token_indexes=[],
        num_results=5,
        ignore_bos=True,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200

    # Validate the structure with Pydantic model
    # This will check all required fields are present with correct types
    data = response.json()
    response_model = ActivationAllPost200Response(**data)

    # Verify we have the expected number of activations
    assert len(response_model.activations) == 5

    activation_rows = [list(activation.values) for activation in response_model.activations]
    assert all(
        activation.source in SAE_SELECTED_SOURCES for activation in response_model.activations
    )
    assert all(len(row) == len(response_model.tokens) for row in activation_rows)
    assert all(any(abs(value) > 0 for value in row) for row in activation_rows)

    max_values = []
    for activation, row in zip(response_model.activations, activation_rows, strict=True):
        row_max = max(row)
        row_max_index = row.index(row_max)
        assert pytest.approx(activation.max_value, abs=1e-5) == row_max
        assert activation.max_value_index == row_max_index
        max_values.append(float(activation.max_value))

    assert max_values == sorted(max_values, reverse=True)

    # Check expected tokens sequence
    assert response_model.tokens[-4:] == ["Hello", ",", " world", "!"]
    if response_model.tokens[0] == BOS_TOKEN_STR:
        assert len(response_model.tokens) == 5
