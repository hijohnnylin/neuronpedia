"""Chat requests against a model with no chat template are refused, not papered over.

Both endpoints used to substitute a generic ChatML render when the tokenizer had no
template of its own. That returned 200 carrying `<|im_start|>` markers the model has
never seen, tokenized as ordinary text — a failure indistinguishable from success
unless you read the output. The refusal is checked here with a stub tokenizer because
it has to hold for every template-less model, not just whichever base model happens to
be loaded.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest
from fastapi import Request

from neuronpedia_inference.endpoints.lens.prompt import lens_prompt
from neuronpedia_inference.endpoints.steer.completion_chat import completion_chat
from neuronpedia_inference.schemas import (
    LensChatMessage,
    LensPromptRequest,
    LensType,
    NPSteerChatMessage,
    NPSteerMethod,
    NPSteerType,
    NPSteerVector,
    SteerCompletionChatRequest,
)
from neuronpedia_inference.shared import Model


class StubTokenizer:
    """A base model's tokenizer: everything else intact, no chat template."""

    chat_template = None


class StubModel:
    def __init__(self):
        self.tokenizer = StubTokenizer()


@pytest.fixture(autouse=True)
def template_less_model() -> Iterator[None]:
    """Every test here is about a template-less model, so this is unconditional."""
    had_instance = hasattr(Model, "_instance")
    previous = getattr(Model, "_instance", None)
    Model.set_instance(StubModel())  # type: ignore[arg-type]
    try:
        yield
    finally:
        if had_instance:
            Model.set_instance(previous)  # type: ignore[arg-type]
        else:
            del Model._instance


def _body(response) -> dict:
    return json.loads(response.body)


def _http_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/lens/prompt", "headers": []})


def _steer_chat_request() -> SteerCompletionChatRequest:
    return SteerCompletionChatRequest(
        model="gpt2-small",
        prompt=[NPSteerChatMessage(role="user", content="What is 2+2?")],
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        vectors=[
            NPSteerVector(
                steering_vector=[0.0] * 8,
                strength=1.0,
                hook="blocks.0.hook_resid_post",
            )
        ],
        n_completion_tokens=8,
        temperature=0.0,
        strength_multiplier=1.0,
        freq_penalty=0.0,
        seed=16,
        steer_special_tokens=True,
    )


def _lens_request(**overrides) -> LensPromptRequest:
    fields = {
        "model": "gpt2-small",
        "type": [LensType.LOGIT_LENS],
        "chat": [LensChatMessage(role="user", content="What is 2+2?")],
        "num_completion_tokens": 0,
        "temperature": 0.0,
        "stream": False,
    }
    fields.update(overrides)
    return LensPromptRequest(**fields)


def test_steer_completion_chat_refuses_and_names_the_completion_route():
    """A 200 here means the ChatML fallback is back."""
    response = asyncio.run(completion_chat(_steer_chat_request()))

    assert response.status_code == 400
    # The remedy is a different route on the same model, so the message has to name it
    # rather than leaving the caller thinking the model itself is unusable.
    assert "chat template" in _body(response)["error"]
    assert "/v1/steer/completion" in _body(response)["error"]


def test_lens_prompt_refuses_chat_without_a_chat_template():
    """Read-outs over ChatML markers are real-looking numbers over a fiction."""
    response = asyncio.run(lens_prompt(_lens_request(), _http_request()))

    assert response.status_code == 400
    assert "chat template" in _body(response)["error"]
    assert "prompt" in _body(response)["error"]


def test_lens_prompt_still_accepts_raw_text_from_the_same_model():
    """The refusal is scoped to `chat` — these models' read-outs are otherwise fine.

    The stub is not a real backend, so the request is still rejected — but by the
    backend check further down, which is how we know it cleared the gate.
    """
    response = asyncio.run(lens_prompt(_lens_request(chat=None, prompt="What is 2+2?"), _http_request()))

    assert "backends" in _body(response)["error"]
    assert "chat template" not in _body(response)["error"]
