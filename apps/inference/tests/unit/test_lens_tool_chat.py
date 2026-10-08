"""Chat with tool calls renders through the real template, and is refused where tools are unread.

Uses the cached Qwen3 tokenizer (no weights), so the render is the one a served Qwen model sees.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest
from fastapi import Request
from interp_engine import Tokenize

from neuronpedia_inference.endpoints.lens.prompt import (
    NO_TOOLS_TEMPLATE_ERROR,
    _chat_args,
    build_token_ids,
    compute_prompt_spans,
    lens_prompt,
)
from neuronpedia_inference.schemas import LensChatMessage, LensPromptRequest, LensType
from neuronpedia_inference.shared import Model

transformers = pytest.importorskip("transformers")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


class StubModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tok = Tokenize(tokenizer, device="cpu")


@pytest.fixture(scope="module")
def qwen_tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen/Qwen3-0.6B tokenizer is not in the local Hugging Face cache")


@pytest.fixture
def use_model() -> Iterator:
    had_instance = hasattr(Model, "_instance")
    previous = getattr(Model, "_instance", None)

    def _set(model):
        Model.set_instance(model)  # type: ignore[arg-type]
        return model

    try:
        yield _set
    finally:
        if had_instance:
            Model.set_instance(previous)  # type: ignore[arg-type]
        else:
            del Model._instance


def _request(chat: list[LensChatMessage], **overrides) -> LensPromptRequest:
    fields = {
        "model": "qwen3-0.6b",
        "type": [LensType.LOGIT_LENS],
        "chat": chat,
        "num_completion_tokens": 0,
        "temperature": 0.0,
        "stream": False,
        "enable_thinking": False,
    }
    fields.update(overrides)
    return LensPromptRequest(**fields)


def _http_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/lens/prompt", "headers": []})


def _tool_chat(final_assistant: bool = True) -> list[LensChatMessage]:
    chat = [
        LensChatMessage(role="user", content="What is the weather in Paris?"),
        LensChatMessage.model_validate(
            {
                "role": "assistant",
                "content": "",
                "toolCalls": [{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "call_1"}],
            }
        ),
    ]
    if not final_assistant:
        chat += [
            LensChatMessage.model_validate({"role": "tool", "content": "Sunny, 21C", "toolCallId": "call_1"}),
            LensChatMessage(role="assistant", content="It is sunny and 21C in Paris."),
        ]
    return chat


def test_tool_calls_take_the_hugging_face_shape(qwen_tokenizer):
    messages, _, _, kwargs = _chat_args(StubModel(qwen_tokenizer).tok, _request(_tool_chat(False), tools=TOOLS))

    assert messages[1]["tool_calls"] == [
        {"type": "function", "id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
    assert messages[2]["tool_call_id"] == "call_1"
    assert kwargs["tools"] == TOOLS


def test_final_tool_calling_turn_is_closed_not_prefilled(qwen_tokenizer):
    """A prefill cuts the render at the end of the content, which would drop the calls."""
    tok = StubModel(qwen_tokenizer).tok
    _, add_generation_prompt, is_prefill, _ = _chat_args(tok, _request(_tool_chat()))
    assert (add_generation_prompt, is_prefill) == (False, False)

    plain = [LensChatMessage(role="user", content="Hi"), LensChatMessage(role="assistant", content="Hel")]
    _, add_generation_prompt, is_prefill, _ = _chat_args(tok, _request(plain))
    assert (add_generation_prompt, is_prefill) == (False, True)


def test_tool_transcript_renders_calls_results_and_tools_with_spans(qwen_tokenizer):
    model = StubModel(qwen_tokenizer)
    request = _request(_tool_chat(False), tools=TOOLS)

    ids = build_token_ids(model, request)
    text = qwen_tokenizer.decode(ids)
    assert '"name": "get_weather"' in text
    assert "<tool_call>" in text
    assert "<tool_response>\nSunny, 21C\n</tool_response>" in text

    spans, _ = compute_prompt_spans(model, request, ids)
    roles = {s.message_index: s.role for s in spans if s.message_index is not None}
    assert roles == {0: "user", 1: "assistant", 2: "tool", 3: "assistant"}


def test_lens_prompt_refuses_tools_for_a_template_that_does_not_read_them(qwen_tokenizer, use_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    use_model(StubModel(tokenizer))

    response = asyncio.run(lens_prompt(_request(_tool_chat(), tools=TOOLS), _http_request()))

    assert response.status_code == 400
    assert json.loads(bytes(response.body))["error"] == NO_TOOLS_TEMPLATE_ERROR
