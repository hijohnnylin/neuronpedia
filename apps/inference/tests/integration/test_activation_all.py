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
from tests.utils.assertions import assert_activation_structure_stable

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

    reference_activation_rows = [
        [0.0, 46.481327056884766, 11.279630661010742, 0.0, 0.0],
        [0.0, 0.0, 3.798774480819702, 6.36670446395874, 8.832769393920898],
        [0.0, 8.095728874206543, 3.749096632003784, 4.03702449798584, 6.3894195556640625],
        [0.0, 0.7275917530059814, 6.788952827453613, 5.938947677612305, 0.0],
        [0.0, 3.8083033561706543, 2.710123062133789, 6.348649501800537, 2.1380198001861572],
    ]

    # Verify we have the expected number of activations
    assert len(response_model.activations) == len(reference_activation_rows)

    activation_rows = [list(activation.values) for activation in response_model.activations]
    assert all(
        activation.source in SAE_SELECTED_SOURCES for activation in response_model.activations
    )
    assert all(len(row) == len(response_model.tokens) for row in activation_rows)
    assert_activation_structure_stable(
        activation_rows,
        reference_activation_rows,
        min_mean_cosine=0.90,
        min_mean_topk_overlap=0.50,
        top_k=3,
    )

    # Check expected tokens sequence
    expected_tokens = [BOS_TOKEN_STR, "Hello", ",", " world", "!"]
    assert response_model.tokens == expected_tokens
