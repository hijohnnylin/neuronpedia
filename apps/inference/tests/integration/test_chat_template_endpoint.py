"""Integration test for the `/apply-chat-template` span-metadata endpoint.

Loads a small instruct model as the engine backend, wires the `Model`/`Config`
singletons, and calls the endpoint handler directly. Asserts the returned spans cover
the whole rendered sequence, tag message roles, and that the token ids match the
tokenizer's own chat-template output.

Downloads weights; auto-skips when the model can't be loaded (gated / offline). Run with:

    cd apps/inference && uv run pytest tests/integration/test_chat_template_endpoint.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

HF_ID = "Qwen/Qwen3-0.6B"


class _StubConfig:
    token_limit = 4096


@pytest.fixture(scope="module")
def loaded_engine_model():
    from interp_engine import EagerModel

    from neuronpedia_inference.config import Config
    from neuronpedia_inference.shared import Model

    try:
        model = EagerModel(HF_ID, dtype="float32", device="cpu")
    except Exception as exc:  # noqa: BLE001 - gated / uncached / offline
        pytest.skip(f"{HF_ID} unavailable: {type(exc).__name__}: {str(exc)[:160]}")

    prev_model = getattr(Model, "_instance", None)
    prev_config = getattr(Config, "_instance", None)
    Model.set_instance(model)
    Config._instance = _StubConfig()  # type: ignore[assignment]
    yield model
    Model._instance = prev_model  # type: ignore[assignment]
    Config._instance = prev_config


def _call(request: Any) -> Any:
    from neuronpedia_inference.endpoints.chat_template import apply_chat_template

    return asyncio.run(apply_chat_template(request))


def test_apply_chat_template_spans(loaded_engine_model: Any):
    from neuronpedia_inference.schemas import (
        ApplyChatTemplateRequest,
        ApplyChatTemplateResponse,
        ChatMessage,
    )

    req = ApplyChatTemplateRequest(
        messages=[
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(role="assistant", content="4"),
        ],
        add_generation_prompt=False,
    )
    resp = _call(req)
    assert isinstance(resp, ApplyChatTemplateResponse)

    # Spans cover every token position contiguously.
    assert [s.position for s in resp.spans] == list(range(len(resp.spans)))
    assert len(resp.tokens) == len(resp.spans)

    # Token ids match the tokenizer's own chat-template rendering exactly.
    # Recent transformers return a BatchEncoding (dict with input_ids/attention_mask)
    # from apply_chat_template(tokenize=True) rather than a bare id list, so pull out
    # input_ids when present.
    ref_ids = loaded_engine_model.tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        add_generation_prompt=False,
        tokenize=True,
    )
    if isinstance(ref_ids, dict) or hasattr(ref_ids, "input_ids"):
        ref_ids = ref_ids["input_ids"]
    assert resp.tokens == list(ref_ids)

    # Roles are tagged for both messages.
    roles = {s.role for s in resp.spans if s.message_index is not None}
    assert "user" in roles and "assistant" in roles


def test_apply_chat_template_thinking_kwarg(loaded_engine_model: Any):  # noqa: ARG001
    from neuronpedia_inference.schemas import (
        ApplyChatTemplateRequest,
        ChatMessage,
    )

    base = ApplyChatTemplateRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )
    nothink = ApplyChatTemplateRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    # The thinking switch flows through to the template and changes the token stream.
    assert _call(base).tokens != _call(nothink).tokens
