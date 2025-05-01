from collections import Counter

from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_all_post200_response import (
    ActivationAllPost200Response,
)
from neuronpedia_inference_client.models.activation_all_post_request import (
    ActivationAllPostRequest,
)

from tests.conftest import BOS_TOKEN_STR, TEST_PROMPT, X_SECRET_KEY


def test_activation_all(client: TestClient):
    """
    Test basic functionality of the /activation/all endpoint with a simple request.
    """
    request = ActivationAllPostRequest(
        prompt=TEST_PROMPT,
        model="gpt2-small",
        source_set="res-jb",
        selected_sources=["7-res-jb"],
        sort_by_token_indexes=[],
        ignore_bos=True,
    )

    response = client.post(
        "/v1/activation/all",
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )

    assert response.status_code == 200

    # Validate the structure with Pydantic model
    # This will check all required fields are present with correct types
    data = response.json()
    response_model = ActivationAllPost200Response(**data)

    # Ensure we received non-empty results
    assert len(response_model.tokens) > 0, "No tokens returned"
    assert len(response_model.activations) > 0, "No activations returned"

    # Check for duplicate indices
    indices = [activation.index for activation in response_model.activations]
    index_counts = Counter(indices)
    duplicate_indices = [index for index, count in index_counts.items() if count > 1]
    assert len(duplicate_indices) == 0, f"Found duplicate indices: {duplicate_indices}"

    # Check expected tokens sequence
    assert (
        response_model.tokens[0] == BOS_TOKEN_STR
    ), f"First token is not '{BOS_TOKEN_STR}'"
    assert response_model.tokens[1] == "Hello", "Second token is not 'Hello'"
    assert response_model.tokens[2] == ",", "Third token is not ','"
    assert response_model.tokens[3] == " world", "Fourth token is not ' world'"
    assert response_model.tokens[4] == "!", "Fifth token is not '!'"
