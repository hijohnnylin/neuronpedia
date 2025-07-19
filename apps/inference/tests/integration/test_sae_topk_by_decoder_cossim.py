from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_feature import NPFeature
from neuronpedia_inference_client.models.util_sae_topk_by_decoder_cossim_post200_response import (
    UtilSaeTopkByDecoderCossimPost200Response,
)
from neuronpedia_inference_client.models.util_sae_topk_by_decoder_cossim_post_request import (
    UtilSaeTopkByDecoderCossimPostRequest,
)

from tests.conftest import (
    INVALID_SAE_SOURCE,
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/util/sae-topk-by-decoder-cossim"
FEATURE_INDEX = 5
NUM_RESULTS = 3
COSSIM_ABS_TOLERANCE = 1e-5


def test_sae_topk_by_decoder_cossim_with_feature(client: TestClient):
    """
    Test the endpoint with a valid feature input.
    """
    request = UtilSaeTopkByDecoderCossimPostRequest(
        feature=NPFeature(
            source=SAE_SELECTED_SOURCES[0],
            index=FEATURE_INDEX,
            model=MODEL_ID,
        ),
        source=SAE_SELECTED_SOURCES[0],
        model=MODEL_ID,
        num_results=NUM_RESULTS,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200
    data = response.json()
    response_model = UtilSaeTopkByDecoderCossimPost200Response(**data)

    # Verify topk results
    assert len(response_model.topk_decoder_cossim_features) == NUM_RESULTS  # type: ignore

    expected_results = [
        {"index": 5, "similarity": 1.0},
        {"index": 15785, "similarity": 0.6101891398429871},
        {"index": 9213, "similarity": 0.5031709671020508},
    ]

    for i, (actual, expected) in enumerate(  # type: ignore
        zip(response_model.topk_decoder_cossim_features, expected_results)  # type: ignore
    ):
        assert actual.feature.source == SAE_SELECTED_SOURCES[0]
        assert actual.feature.model == MODEL_ID
        assert (
            actual.feature.index == expected["index"]
        ), f"Feature {i}: expected index {expected['index']}, got {actual.feature.index}"
        assert (
            abs(actual.cosine_similarity - expected["similarity"])
            < COSSIM_ABS_TOLERANCE
        ), f"Feature {i}: expected similarity {expected['similarity']}, got {actual.cosine_similarity}"


def test_sae_topk_by_decoder_cossim_with_vector(client: TestClient):
    """
    Test the endpoint with a valid vector input.
    """
    test_vector = [1.0] * 768

    request = UtilSaeTopkByDecoderCossimPostRequest(
        vector=test_vector,
        source=SAE_SELECTED_SOURCES[0],
        model=MODEL_ID,
        num_results=NUM_RESULTS,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200
    data = response.json()
    response_model = UtilSaeTopkByDecoderCossimPost200Response(**data)

    # Verify response structure
    assert response_model.feature is None  # No input feature
    assert len(response_model.topk_decoder_cossim_features) == NUM_RESULTS  # type: ignore

    # Check actual expected values for vector test
    expected_results = [
        {"index": 16087, "similarity": 0.7319665551185608},
        {"index": 20434, "similarity": 0.40199577808380127},
        {"index": 12002, "similarity": 0.05817627161741257},
    ]

    for i, (actual, expected) in enumerate(  # type: ignore
        zip(response_model.topk_decoder_cossim_features, expected_results)  # type: ignore
    ):
        assert actual.feature.source == SAE_SELECTED_SOURCES[0]
        assert actual.feature.model == MODEL_ID
        assert (
            actual.feature.index == expected["index"]
        ), f"Vector test feature {i}: expected index {expected['index']}, got {actual.feature.index}"
        assert (
            abs(actual.cosine_similarity - expected["similarity"])
            < COSSIM_ABS_TOLERANCE
        ), f"Vector test feature {i}: expected similarity {expected['similarity']}, got {actual.cosine_similarity}"


def test_sae_topk_by_decoder_cossim_input_validation_errors(client: TestClient):
    """
    Test input validation where both feature and vector are provided.
    """
    test_vector = [1.0] * 768

    # Test both feature and vector provided
    request_both = UtilSaeTopkByDecoderCossimPostRequest(
        feature=NPFeature(
            source=SAE_SELECTED_SOURCES[0],
            index=FEATURE_INDEX,
            model=MODEL_ID,
        ),
        vector=test_vector,
        source=SAE_SELECTED_SOURCES[0],
        model=MODEL_ID,
        num_results=NUM_RESULTS,
    )

    response_both = client.post(
        ENDPOINT,
        json=request_both.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response_both.status_code == 400
    assert (
        "exactly one of feature or vector must be provided"
        in response_both.json()["error"]
    )


def test_sae_topk_by_decoder_cossim_invalid_source(client: TestClient):
    """
    Test the endpoint with an invalid SAE source.
    """
    request = UtilSaeTopkByDecoderCossimPostRequest(
        feature=NPFeature(
            source=INVALID_SAE_SOURCE,
            index=FEATURE_INDEX,
            model=MODEL_ID,
        ),
        source=INVALID_SAE_SOURCE,
        model=MODEL_ID,
        num_results=NUM_RESULTS,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 400
    data = response.json()
    assert f"Invalid SAE ID or type: {INVALID_SAE_SOURCE}" in data["error"]
