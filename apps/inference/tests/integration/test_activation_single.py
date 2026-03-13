from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_single_post200_response import (
    ActivationSinglePost200Response,
)
from neuronpedia_inference_client.models.activation_single_post_request import (
    ActivationSinglePostRequest,
)

from tests.conftest import (
    BOS_TOKEN_STR,
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    TEST_PROMPT,
    X_SECRET_KEY,
)
from tests.utils.assertions import assert_activation_structure_stable

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

    reference_activations = [[134.71969604492188, 0.051671065390110016, 0.0, 0.0, 0.0]]
    assert_activation_structure_stable(
        [list(response_model.activation.values)],
        reference_activations,
        min_mean_cosine=0.99,
        min_mean_topk_overlap=1.0,
        top_k=2,
    )
    assert response_model.activation.max_value is not None
    assert response_model.activation.max_value_index is not None

    # Check tokens
    expected_tokens = [BOS_TOKEN_STR, "Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens


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

    reference_activations = [[5.4140625, 3.23828125, 1.9462890625, 1.671875]]
    assert_activation_structure_stable(
        [list(response_model.activation.values)],
        reference_activations,
        min_mean_cosine=0.99,
        min_mean_topk_overlap=1.0,
        top_k=2,
    )
    assert response_model.activation.max_value is not None
    assert response_model.activation.max_value_index is not None

    # Check token values
    expected_tokens = ["Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens
