import pytest
from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_all_post200_response import (
    ActivationAllPost200Response,
)
from neuronpedia_inference_client.models.activation_all_post_request import (
    ActivationAllPostRequest,
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

    # Expected data based on the provided response
    expected_activations_data = [
        {
            "source": "7-res-jb",
            "index": 16653,
            "values": [0.0, 46.481327056884766, 11.279630661010742, 0.0, 0.0],
            "max_value": 46.481327056884766,
            "max_value_index": 1,
        },
        {
            "source": "7-res-jb",
            "index": 11553,
            "values": [
                0.0,
                0.0,
                3.798774480819702,
                6.36670446395874,
                8.832769393920898,
            ],
            "max_value": 8.832769393920898,
            "max_value_index": 4,
        },
        {
            "source": "7-res-jb",
            "index": 9810,
            "values": [
                0.0,
                8.095728874206543,
                3.749096632003784,
                4.03702449798584,
                6.3894195556640625,
            ],
            "max_value": 8.095728874206543,
            "max_value_index": 1,
        },
        {
            "source": "7-res-jb",
            "index": 14806,
            "values": [
                0.0,
                0.7275917530059814,
                6.788952827453613,
                5.938947677612305,
                0.0,
            ],
            "max_value": 6.788952827453613,
            "max_value_index": 2,
        },
        {
            "source": "7-res-jb",
            "index": 16488,
            "values": [
                0.0,
                3.8083033561706543,
                2.710123062133789,
                6.348649501800537,
                2.1380198001861572,
            ],
            "max_value": 6.348649501800537,
            "max_value_index": 3,
        },
    ]

    # Verify we have the expected number of activations
    assert len(response_model.activations) == len(expected_activations_data)

    # Check each activation against expected data
    for i, (actual, expected) in enumerate(
        zip(response_model.activations, expected_activations_data)
    ):
        assert actual.source == expected["source"], f"Activation {i}: source mismatch"
        assert actual.index == expected["index"], f"Activation {i}: index mismatch"
        assert (
            pytest.approx(actual.values, abs=ABS_TOLERANCE) == expected["values"]
        ), f"Activation {i}: values mismatch"
        assert (
            pytest.approx(actual.max_value, abs=ABS_TOLERANCE) == expected["max_value"]
        ), f"Activation {i}: max_value mismatch"
        assert (
            actual.max_value_index == expected["max_value_index"]
        ), f"Activation {i}: max_value_index mismatch"

    # Check expected tokens sequence
    expected_tokens = [BOS_TOKEN_STR, "Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens
