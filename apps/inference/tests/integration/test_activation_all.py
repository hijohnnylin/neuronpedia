import pytest
from fastapi.testclient import TestClient

from neuronpedia_inference.schemas import (
    ActivationAllRequest,
    ActivationAllResponse,
)
from tests.conftest import (
    ABS_TOLERANCE,
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
    request = ActivationAllRequest(
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
    response_model = ActivationAllResponse(**data)

    # Expected data based on the provided response
    expected_activations_data = [
        {
            "source": "7-res-jb",
            "index": 16653,
            "values": [0.0, 46.47268295288086, 11.284173965454102, 0.0, 0.0],
            "max_value": 46.47268295288086,
            "max_value_index": 1,
        },
        {
            "source": "7-res-jb",
            "index": 13715,
            "values": [0.0, 43.2230224609375, 5.321485996246338, 0.0, 0.0],
            "max_value": 43.2230224609375,
            "max_value_index": 1,
        },
        {
            "source": "7-res-jb",
            "index": 2494,
            "values": [
                0.0,
                3.085273504257202,
                33.54616165161133,
                15.50393295288086,
                15.789239883422852,
            ],
            "max_value": 33.54616165161133,
            "max_value_index": 2,
        },
        {
            "source": "7-res-jb",
            "index": 22763,
            "values": [0.0, 0.0, 0.0, 26.60720443725586, 0.0],
            "max_value": 26.60720443725586,
            "max_value_index": 3,
        },
        {
            "source": "7-res-jb",
            "index": 13413,
            "values": [0.0, 0.0, 0.0, 0.0, 24.993972778320312],
            "max_value": 24.993972778320312,
            "max_value_index": 4,
        },
    ]

    # Verify we have the expected number of activations
    assert len(response_model.activations) == len(expected_activations_data)

    # Check each activation against expected data
    for i, (actual, expected) in enumerate(zip(response_model.activations, expected_activations_data)):
        assert actual.source == expected["source"], f"Activation {i}: source mismatch"
        assert actual.index == expected["index"], f"Activation {i}: index mismatch"
        assert pytest.approx(actual.values, abs=ABS_TOLERANCE) == expected["values"], f"Activation {i}: values mismatch"
        assert pytest.approx(actual.max_value, abs=ABS_TOLERANCE) == expected["max_value"], (
            f"Activation {i}: max_value mismatch"
        )
        assert actual.max_value_index == expected["max_value_index"], f"Activation {i}: max_value_index mismatch"

    # Check expected tokens sequence
    expected_tokens = [BOS_TOKEN_STR, "Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens
