"""``/v1/steer/completion-chat``'s token limit, on a model that has a chat template.

The limit is checked against the *rendered* conversation, which makes it unreachable from the
suite's base model: gpt2-small has no chat template, so the request is refused before anything
is counted (``test_no_chat_template_endpoints.py``). The real instruct models are gated or
GPU-only, so the guard is pinned here against a stub tokenizer that renders a template — the
one arrangement that keeps it covered on CPU CI, where this endpoint otherwise only ever
produces its refusal.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.steer.completion_chat import completion_chat
from neuronpedia_inference.schemas import (
    NPSteerChatMessage,
    NPSteerMethod,
    NPSteerType,
    NPSteerVector,
    SteerCompletionChatRequest,
)
from neuronpedia_inference.shared import Model

TOKEN_LIMIT = 16


class StubTokenizer:
    """An instruct model's tokenizer, reduced to what the endpoint asks of it.

    One token per whitespace-separated word, and a render that wraps each turn, so a
    conversation's length is obvious from the message text in the test.
    """

    chat_template = "{# present; contents never evaluated here #}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,  # noqa: ARG002
        add_generation_prompt: bool = False,  # noqa: ARG002
        continue_final_message: bool = False,  # noqa: ARG002
    ) -> str:
        # `continue_final_message` is unused here but part of the signature the endpoint now
        # renders through: it goes via the engine's `Tokenize`, which always passes both flags,
        # so that a model whose chat format lives in code rather than in a template is served
        # by the same call.
        return " ".join(f"<turn> {m['content']} </turn>" for m in conversation)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,  # noqa: ARG002
    ) -> dict[str, list[int]]:
        return {"input_ids": [7] * len(text.split())}


class StubModel:
    def __init__(self):
        self.tokenizer = StubTokenizer()


@pytest.fixture(autouse=True)
def templated_model_with_small_limit() -> Iterator[None]:
    """A template-bearing backend and a ``token_limit`` a one-line prompt can exceed."""
    had_model = hasattr(Model, "_instance")
    previous_model = getattr(Model, "_instance", None)
    previous_config = Config._instance
    Model.set_instance(StubModel())  # type: ignore[arg-type]
    # The real Config would build the SAE directory, which this has no use for.
    with patch.object(Config, "_generate_sae_config", return_value=[]):
        Config._instance = Config(token_limit=TOKEN_LIMIT)
    try:
        yield
    finally:
        Config._instance = previous_config
        if had_model:
            Model.set_instance(previous_model)  # type: ignore[arg-type]
        else:
            del Model._instance


def _chat_request(content: str) -> SteerCompletionChatRequest:
    return SteerCompletionChatRequest(
        model="stub-instruct",
        prompt=[NPSteerChatMessage(role="user", content=content)],
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


def _body(response: Any) -> dict[str, Any]:
    return json.loads(response.body)


def test_chat_over_the_token_limit_is_refused():
    over = "word " * (TOKEN_LIMIT * 2)
    response = asyncio.run(completion_chat(_chat_request(over)))

    assert response.status_code == 400
    error = _body(response)["error"]
    assert "Text too long" in error
    assert f"max is {TOKEN_LIMIT}" in error


def test_chat_within_the_token_limit_clears_the_guard():
    """The stub is not a real backend, so this still fails — past the guard, at generation.

    Asserting where it fails is what distinguishes "the limit let it through" from "the limit
    happens to reject everything".
    """
    with pytest.raises(ValueError, match="only supports"):
        asyncio.run(completion_chat(_chat_request("hello there")))
