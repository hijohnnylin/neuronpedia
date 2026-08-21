import pytest
from fastapi.testclient import TestClient

from neuronpedia_inference.schemas import (
    ActivationSingleRequest,
    ActivationSingleResponse,
)
from tests.conftest import (
    ABS_TOLERANCE,
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
    request = ActivationSingleRequest(
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
    response_model = ActivationSingleResponse(**data)

    # Check activation values. The engine strips the BOS position from the returned
    # activations/tokens (matching the vector/hook path below), so this is the 4-token
    # ["Hello", ",", " world", "!"] sequence.
    expected_activations = [0.04774539917707443, 0.0, 0.0, 0.0]
    expected_max_value = 0.04774539917707443
    expected_max_value_index = 0
    assert pytest.approx(response_model.activation.values, abs=ABS_TOLERANCE) == expected_activations
    assert pytest.approx(response_model.activation.max_value, abs=ABS_TOLERANCE) == expected_max_value
    assert response_model.activation.max_value_index == expected_max_value_index

    # Check tokens
    expected_tokens = ["Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens


def test_activation_single_with_vector_and_hook(client: TestClient):
    """
    Test the /activation/single endpoint with vector and hook parameters.
    """
    # Create a test vector matching the residual stream dimension (768)
    test_vector = [0.1] * 768
    test_hook = "blocks.0.hook_resid_post"  # _pre works here too

    request = ActivationSingleRequest(
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
    response_model = ActivationSingleResponse(**data)

    # Check activation values
    expected_activations = [5.4140625, 3.23828125, 1.9462890625, 1.671875]
    expected_max_value = 5.4140625
    expected_max_value_index = 0
    assert pytest.approx(response_model.activation.values, abs=ABS_TOLERANCE) == expected_activations
    assert pytest.approx(response_model.activation.max_value, abs=ABS_TOLERANCE) == expected_max_value
    assert response_model.activation.max_value_index == expected_max_value_index

    # Check token values
    expected_tokens = ["Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens
