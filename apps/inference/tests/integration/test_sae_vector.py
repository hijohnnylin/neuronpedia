import json
import os

import pytest
from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.util_sae_vector_post200_response import (
    UtilSaeVectorPost200Response,
)
from neuronpedia_inference_client.models.util_sae_vector_post_request import (
    UtilSaeVectorPostRequest,
)

from tests.conftest import (
    ABS_TOLERANCE,
    INVALID_SAE_SOURCE,
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/util/sae-vector"
FEATURE_INDEX_5 = 5
FEATURE_INDEX_6 = 6


def test_sae_vector_valid_request(client: TestClient):
    """
    Test the /util/sae-vector endpoint with valid parameters.
    Utilize multiple indexes to be sure that we have different vectors for different indices.
    """

    request_idx_5 = UtilSaeVectorPostRequest(
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        index=FEATURE_INDEX_5,
    )

    request_idx_6 = UtilSaeVectorPostRequest(
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        index=FEATURE_INDEX_6,
    )

    response_idx_5 = client.post(
        ENDPOINT,
        json=request_idx_5.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    response_idx_6 = client.post(
        ENDPOINT,
        json=request_idx_6.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response_idx_5.status_code == 200
    assert response_idx_6.status_code == 200

    data_idx_5 = response_idx_5.json()
    data_idx_6 = response_idx_6.json()

    response_model_idx_5 = UtilSaeVectorPost200Response(**data_idx_5)
    response_model_idx_6 = UtilSaeVectorPost200Response(**data_idx_6)

    test_data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_data",
        "sae_vector_integration_test_data.json",
    )
    with open(test_data_path) as f:
        test_data = json.load(f)
        expected_feature_vector_index_5 = test_data["expected_feature_vector_index_5"]
        expected_feature_vector_index_6 = test_data["expected_feature_vector_index_6"]

    # Compare vectors using absolute tolerance
    assert len(response_model_idx_5.vector) == len(expected_feature_vector_index_5)
    assert len(response_model_idx_6.vector) == len(expected_feature_vector_index_6)

    for actual, expected in zip(
        response_model_idx_5.vector, expected_feature_vector_index_5
    ):
        assert (
            abs(actual - expected) <= ABS_TOLERANCE
        ), f"Vector element difference {abs(actual - expected)} exceeds tolerance {ABS_TOLERANCE}"

    for actual, expected in zip(
        response_model_idx_6.vector, expected_feature_vector_index_6
    ):
        assert (
            abs(actual - expected) <= ABS_TOLERANCE
        ), f"Vector element difference {abs(actual - expected)} exceeds tolerance {ABS_TOLERANCE}"


def test_sae_vector_invalid_source(client: TestClient):
    """
    Test the /util/sae-vector endpoint with an invalid source.
    """
    request = UtilSaeVectorPostRequest(
        model=MODEL_ID,
        source=INVALID_SAE_SOURCE,
        index=FEATURE_INDEX_5,
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
