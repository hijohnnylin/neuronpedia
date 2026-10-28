"""The reasoning of an assistant message in the prompt gets its own channel, as generated reasoning does.

Uses the cached Qwen3.6-27B tokenizer (no weights), so the render is the one the served model sees.
"""

from __future__ import annotations

import pytest
from interp_engine import ReasoningTags, Tokenize, TokenSpan

from neuronpedia_inference.endpoints.lens.prompt import (
    build_token_ids,
    channel_prompt_reasoning,
    compute_prompt_spans,
)
from neuronpedia_inference.schemas import LensChatMessage, LensPromptRequest, LensType

transformers = pytest.importorskip("transformers")

TAGS = ReasoningTags(open="<think>", close="</think>")


class StubModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tok = Tokenize(tokenizer, device="cpu")


@pytest.fixture(scope="module")
def qwen_tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen/Qwen3.6-27B tokenizer is not in the local Hugging Face cache")


def _request(chat: list[LensChatMessage]) -> LensPromptRequest:
    return LensPromptRequest(
        model="qwen3.6-27b",
        type=[LensType.LOGIT_LENS],
        chat=chat,
        num_completion_tokens=0,
        temperature=0.0,
        stream=False,
    )


def _span(position: int, token: str, section: str = "content", message_index: int | None = 1) -> TokenSpan:
    return TokenSpan(
        position=position,
        token_id=position,
        token_str=token,
        message_index=message_index,
        role="assistant" if message_index is not None else None,
        channel=None,
        section=section,
    )


def test_prefilled_reasoning_and_answer_get_their_own_channels(qwen_tokenizer):
    model = StubModel(qwen_tokenizer)
    request = _request(
        [
            LensChatMessage(role="system", content="Give a wrong answer."),
            LensChatMessage(role="user", content="Where did Assamese cuisine start?"),
            LensChatMessage(role="assistant", content="<think>\nSay Bhutan.\n</think>\n\nIt started in Bhutan."),
        ]
    )
    ids = build_token_ids(model, request)
    spans, is_prefill = compute_prompt_spans(model, request, ids)
    assert is_prefill

    reply = [s for s in spans if s.message_index == 2]
    by_token = {s.token_str.strip(): s for s in reply}
    assert (by_token["<think>"].channel, by_token["<think>"].section) == ("analysis", "header")
    assert (by_token["</think>"].channel, by_token["</think>"].section) == ("analysis", "footer")
    thinking = "".join(s.token_str for s in reply if s.channel == "analysis" and s.section == "content")
    answer = "".join(s.token_str for s in reply if s.channel == "final")
    assert thinking.strip() == "Say Bhutan."
    assert answer.strip() == "It started in Bhutan."
    # The turn's opening header has no channel, and the other messages are as they were.
    assert reply[0].channel is None
    assert all(s.channel is None for s in spans if s.message_index != 2)


def test_message_without_reasoning_text_is_left_as_it_was():
    spans = [_span(0, "<think>", "header"), _span(1, "\n\n", "header"), _span(2, "</think>", "header"), _span(3, "Hi")]
    assert channel_prompt_reasoning(spans, TAGS) == spans


def test_unclosed_reasoning_runs_to_the_end_of_the_message():
    spans = [_span(0, "<think>", "header"), _span(1, "Let"), _span(2, " me"), _span(3, "<|im_end|>", "footer", None)]
    out = channel_prompt_reasoning(spans, TAGS)
    assert [s.channel for s in out] == ["analysis", "analysis", "analysis", None]


def test_no_tags_or_no_reasoning_model_changes_nothing():
    spans = [_span(0, "Hi"), _span(1, " there")]
    assert channel_prompt_reasoning(spans, TAGS) == spans
    with_tags = [_span(0, "<think>"), _span(1, "x"), _span(2, "</think>"), _span(3, "y")]
    assert channel_prompt_reasoning(with_tags, None) == with_tags
