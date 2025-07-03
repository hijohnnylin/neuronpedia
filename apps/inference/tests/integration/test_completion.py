from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_vector import NPSteerVector
from neuronpedia_inference_client.models.steer_completion_post200_response import (
    SteerCompletionPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_request import (
    SteerCompletionRequest,
)

from tests.conftest import MODEL_ID, SAE_SELECTED_SOURCES, TEST_PROMPT, X_SECRET_KEY

ENDPOINT = "/v1/steer/completion"
N_COMPLETION_TOKENS = 10
TEMPERATURE = 0
STRENGTH = 10.0  # Steering mechanism (feature or vector) specific strength
STRENGTH_MULTIPLIER = 1.0  # Multiplier across all steering mechanisms
STEER_FEATURE_INDEX = 0
SEED = 42
FREQ_PENALTY = 0.0


def test_completion_steered_with_features_additive(client: TestClient):
    """
    Test steering using features with additive method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]


def test_completion_steered_with_vectors_additive(client: TestClient):
    """
    Test steering using vectors with additive method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        vectors=[
            NPSteerVector(
                steering_vector=[1000.0]
                * 768,  # We utilize a large vector to ensure the steering vector is impactful
                strength=STRENGTH,
                hook="blocks.7.hook_resid_post",
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]


def test_completion_steered_token_limit_exceeded(client: TestClient):
    """
    Test handling of a prompt that exceeds the token limit.
    """
    long_prompt = "This is a test prompt. " * 1000
    request = SteerCompletionRequest(
        prompt=long_prompt,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=0,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400

    data = response.json()
    expected_error_message = "Text too long: 6001 tokens, max is 500"
    assert data["error"] == expected_error_message


def test_completion_steered_with_features_orthogonal(client: TestClient):
    """
    Test steering using features with orthogonal decomposition method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.ORTHOGONAL_DECOMP,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]


def test_completion_steered_with_vectors_orthogonal(client: TestClient):
    """
    Test steering using vectors with orthogonal decomposition.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.ORTHOGONAL_DECOMP,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        vectors=[
            NPSteerVector(
                steering_vector=[1000.0]
                * 768,  # We utilize a large vector to ensure the steering vector is impactful
                strength=STRENGTH,
                hook="blocks.7.hook_resid_post",
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]
