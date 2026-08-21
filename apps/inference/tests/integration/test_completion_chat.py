"""``/v1/steer/completion-chat`` against the suite's base model, gpt2-small.

gpt2-small's tokenizer has no chat template, so a chat request to it is refused with a 400
that names the completion route. This file used to assert 200s here, pinning the output of a
generic ChatML render the endpoint substituted for the missing template -- text full of
`<|im_start|>` markers gpt2 has never seen. That fallback is gone (see
``NO_CHAT_TEMPLATE_ERROR`` in ``endpoints/steer/completion_chat.py``), so what a base model
can assert about this endpoint is the refusal and the request validation in front of it.

The coverage those 200s were standing in for lives where it can be real:

- steering math on gpt2-small (features/vectors, additive/orthogonal) -> ``test_completion.py``,
  over the raw-prompt route that suits a base model.
- chat generation end-to-end -> the instruct models in ``test_model_coverage.py``.
- the endpoint's token-limit guard -> ``tests/unit/test_completion_chat_token_limit.py``,
  which stubs a tokenizer that has a template so the guard is reachable on CPU CI.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from neuronpedia_inference.schemas import (
    NPSteerChatMessage,
    NPSteerFeature,
    NPSteerMethod,
    NPSteerType,
    NPSteerVector,
    SteerCompletionChatRequest,
)
from tests.conftest import (
    FREQ_PENALTY,
    MODEL_ID,
    N_COMPLETION_TOKENS,
    SAE_SELECTED_SOURCES,
    SEED,
    STEER_FEATURE_INDEX,
    STEER_SPECIAL_TOKENS,
    STRENGTH,
    STRENGTH_MULTIPLIER,
    TEMPERATURE,
    TEST_PROMPT,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/steer/completion-chat"

TEST_STEER_FEATURE = NPSteerFeature(
    model=MODEL_ID,
    source=SAE_SELECTED_SOURCES[0],
    index=STEER_FEATURE_INDEX,
    strength=STRENGTH,
)

TEST_STEER_VECTOR = NPSteerVector(
    steering_vector=[1000.0] * 768,
    strength=STRENGTH,
    hook="blocks.7.hook_resid_post",
)


def _chat_request(**overrides: Any) -> SteerCompletionChatRequest:
    """A valid feature-steered chat request; ``overrides`` swap individual fields."""
    fields: dict[str, Any] = {
        "prompt": [NPSteerChatMessage(content=TEST_PROMPT, role="user")],
        "model": MODEL_ID,
        "steer_method": NPSteerMethod.SIMPLE_ADDITIVE,
        "normalize_steering": False,
        "types": [NPSteerType.STEERED, NPSteerType.DEFAULT],
        "features": [TEST_STEER_FEATURE],
        "n_completion_tokens": N_COMPLETION_TOKENS,
        "temperature": TEMPERATURE,
        "strength_multiplier": STRENGTH_MULTIPLIER,
        "freq_penalty": FREQ_PENALTY,
        "seed": SEED,
        "steer_special_tokens": STEER_SPECIAL_TOKENS,
    }
    fields.update(overrides)
    return SteerCompletionChatRequest(**fields)


def _post(client: TestClient, request: SteerCompletionChatRequest):
    return client.post(ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY})


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="features-additive"),
        pytest.param({"features": None, "vectors": [TEST_STEER_VECTOR]}, id="vectors-additive"),
        pytest.param({"steer_method": NPSteerMethod.ORTHOGONAL_DECOMP}, id="features-orthogonal"),
    ],
)
def test_completion_chat_refused_without_a_chat_template(client: TestClient, overrides: dict[str, Any]):
    """A 200 for any of these means the ChatML fallback is back.

    Parametrized over the steering configurations this file used to generate text for: the
    refusal belongs to the model, so no steering setup gets around it.
    """
    response = _post(client, _chat_request(**overrides))

    assert response.status_code == 400
    data = response.json()
    assert "chat template" in data["error"]
    # The remedy is a different route on the same model, so the message has to name it
    # rather than leaving the caller thinking gpt2-small is unusable.
    assert "/v1/steer/completion" in data["error"]
    assert "outputs" not in data


def test_completion_chat_invalid_request_no_features_or_vectors(client: TestClient):
    """
    Test error handling when neither features nor vectors are provided.
    """
    response = _post(client, _chat_request(features=None))

    assert response.status_code == 400
    data = response.json()
    assert "exactly one of features or vectors must be provided" in data["error"]


def test_completion_chat_invalid_request_both_features_and_vectors(client: TestClient):
    """
    Test error handling when both features and vectors are provided.
    """
    response = _post(client, _chat_request(vectors=[TEST_STEER_VECTOR]))

    assert response.status_code == 400
    data = response.json()
    assert "exactly one of features or vectors must be provided" in data["error"]
