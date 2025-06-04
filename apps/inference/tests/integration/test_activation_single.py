from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_single_post200_response import (
    ActivationSinglePost200Response,
)
from neuronpedia_inference_client.models.activation_single_post_request import (
    ActivationSinglePostRequest,
)

from tests.conftest import (
    MODEL_ID,
    TEST_PROMPT,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/activation/single"


# def test_activation_single_with_source_and_index(client: TestClient):
#     """
#     Test the /activation/single endpoint with source and index parameters.
#     """
#     request = ActivationSinglePostRequest(
#         prompt=TEST_PROMPT,
#         model=MODEL_ID,
#         source=SAE_SELECTED_SOURCES[0],
#         index="0",
#     )

#     response = client.post(
#         ENDPOINT,
#         json=request.model_dump(),
#         headers={"X-SECRET-KEY": X_SECRET_KEY},
#     )

#     assert response.status_code == 200

#     # Validate the structure with Pydantic model
#     data = response.json()
#     print("this is data", data)
#     response_model = ActivationSinglePost200Response(**data)

#     # Check response structure
#     assert response_model.activation is not None
#     assert response_model.tokens is not None

#     # Check activation values
#     assert len(response_model.activation.values) > 0
#     assert response_model.activation.max_value is not None
#     assert response_model.activation.max_value_index is not None

#     # Check tokens
#     assert len(response_model.tokens) > 0
#     assert response_model.tokens[0] == BOS_TOKEN_STR
#     assert response_model.tokens[1] == "Hello"


def test_activation_single_with_vector_and_hook(client: TestClient):
    """
    Test the /activation/single endpoint with vector and hook parameters.
    """
    # Create a test vector matching the attention head dimension (64)
    test_vector = [0.1] * 64
    test_hook = "blocks.0.attn.hook_v"

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

    # Check response structure
    assert response_model.activation is not None
    assert response_model.tokens is not None

    # Check activation values
    assert len(response_model.activation.values) > 0
    assert response_model.activation.max_value is not None
    assert response_model.activation.max_value_index is not None
