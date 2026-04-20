from fastapi.testclient import TestClient
import pytest
from neuronpedia_inference_client.models.activation_single_post200_response import (
    ActivationSinglePost200Response,
)
from neuronpedia_inference_client.models.activation_single_post_request import (
    ActivationSinglePostRequest,
)

from tests.conftest import (
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    TEST_PROMPT,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/activation/single"


def test_activation_single_with_source_and_index(client: TestClient):
    """
    Test the /activation/single endpoint with source and index parameters.
    """
    request = ActivationSinglePostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        index="0",
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200

    # Validate the structure with Pydantic model
    data = response.json()
    response_model = ActivationSinglePost200Response(**data)

    values = list(response_model.activation.values)
    assert len(values) == len(response_model.tokens)
    assert any(abs(value) > 0 for value in values)
    row_max = max(values)
    row_max_index = values.index(row_max)
    assert pytest.approx(response_model.activation.max_value, abs=1e-5) == row_max
    assert response_model.activation.max_value_index == row_max_index

    # Check tokens
    assert response_model.tokens[-4:] == ["Hello", ",", " world", "!"]


def test_activation_single_with_vector_and_hook(client: TestClient):
    """
    Test the /activation/single endpoint with vector and hook parameters.
    """
    # Create a test vector matching the residual stream dimension (768)
    test_vector = [0.1] * 768
    test_hook = "blocks.0.hook_resid_post"  # _pre works here too

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

    # Validate the structure with Pydantic model
    data = response.json()
    response_model = ActivationSinglePost200Response(**data)

    values = list(response_model.activation.values)
    assert len(values) == len(response_model.tokens)
    assert any(abs(value) > 0 for value in values)
    row_max = max(values)
    row_max_index = values.index(row_max)
    assert pytest.approx(response_model.activation.max_value, abs=1e-5) == row_max
    assert response_model.activation.max_value_index == row_max_index

    # Check token values
    expected_tokens = ["Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens
