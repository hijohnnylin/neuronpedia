"""Persona capture pools activations over per-message token spans; these pin the spans.

The spans come from the engine's ``Tokenize.message_partition`` rather than being computed
here, so these tests drive the real engine helper through stub tokenizers. Two properties
matter to persona: the spans partition the rendered sequence 1:1 with the conversation, and the
ids are real ``int``s (transformers 5 returns a ``BatchEncoding`` from the tokenizing path, and
iterating that yields the string key ``"input_ids"`` — the failure that showed up in pod logs).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import torch
from interp_engine import Address, Tokenize, VLLMModel, to_address

from neuronpedia_inference.inference_utils.persona.capture_engine import (
    _per_message_spans,
    capture_turn_means_vllm,
    turn_means_from_generation_capture,
)


class _BatchEncoding(dict):
    """Minimal stand-in for transformers' BatchEncoding (dict + attribute access)."""

    @property
    def input_ids(self):
        return self["input_ids"]


class _BatchEncodingTokenizer:
    """Returns a BatchEncoding from ``apply_chat_template(tokenize=True)``."""

    # Present but never evaluated: the engine reads it only to decide that this model renders
    # chat from a template rather than from a code formatter.
    chat_template = "{# present; contents never evaluated here #}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,  # noqa: ARG002
        continue_final_message: bool = False,  # noqa: ARG002
    ):
        # Growing prefix: 2 tokens per message (simulates a chat template).
        ids = list(range(1, 2 * len(messages) + 1))
        if not tokenize:
            return " ".join(m["content"] for m in messages)
        return _BatchEncoding(input_ids=ids)


class _ListTokenizer:
    chat_template = "{# present; contents never evaluated here #}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,  # noqa: ARG002
        add_generation_prompt: bool = False,  # noqa: ARG002
        continue_final_message: bool = False,  # noqa: ARG002
    ):
        return list(range(1, 2 * len(messages) + 1))


def _tok(tokenizer: Any) -> Tokenize:
    return Tokenize(tokenizer, device="cpu")


def test_per_message_spans_with_batch_encoding():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    full_ids, spans = _per_message_spans(_tok(_BatchEncodingTokenizer()), msgs)
    assert full_ids == [1, 2, 3, 4]
    assert spans == [(0, 2), (2, 4)]
    # Must be real ints, not dict keys — this is the failure mode from the pod logs.
    assert all(isinstance(t, int) for t in full_ids)


def test_per_message_spans_with_list():
    msgs = [{"role": "user", "content": "hi"}]
    full_ids, spans = _per_message_spans(_tok(_ListTokenizer()), msgs)
    assert full_ids == [1, 2]
    assert spans == [(0, 2)]


def test_per_message_spans_partition_the_whole_sequence():
    """Persona indexes its projection per message, so a gap or overlap misattributes a turn."""
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(4)]
    full_ids, spans = _per_message_spans(_tok(_ListTokenizer()), msgs)

    assert len(spans) == len(msgs)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(full_ids)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_turn_means_from_generation_capture_pools_prompt_and_generated():
    msgs = [{"role": "user", "content": "hi"}]
    # The prompt is the rendered message (ids 1,2) plus a generation-header token (9);
    # the capture then adds a row per generated token.
    prompt_ids = [1, 2, 9]
    acts = torch.tensor(
        [[0.0, 0.0], [2.0, 4.0], [1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
    )
    means = turn_means_from_generation_capture(_tok(_ListTokenizer()), msgs, prompt_ids, acts)
    assert means is not None
    assert means.shape == (2, 2)
    torch.testing.assert_close(means[0], torch.tensor([1.0, 2.0]))
    # The generated turn runs from the header token through the last captured row.
    torch.testing.assert_close(means[1], torch.tensor([3.0, 3.0]))


def test_turn_means_from_generation_capture_rejects_misaligned_prompt():
    msgs = [{"role": "user", "content": "hi"}]
    assert turn_means_from_generation_capture(_tok(_ListTokenizer()), msgs, [7, 7, 7], torch.zeros(5, 2)) is None


def test_turn_means_from_generation_capture_rejects_short_capture():
    msgs = [{"role": "user", "content": "hi"}]
    assert turn_means_from_generation_capture(_tok(_ListTokenizer()), msgs, [1, 2, 9], torch.zeros(2, 2)) is None


def test_capture_turn_means_vllm_reads_back_the_point_it_asked_for():
    """``capture`` keys its result by ``Address``, so a second spelling of the point is a KeyError.

    A ``("resid_post", layer)`` tuple survived the ``Point`` -> ``Address`` migration here, and
    every steered assistant-axis turn runs through this function -- a steered generation's own
    means are post-cap, so the pre-cap ones always come from a second capture like this one.
    """
    asked: list[Any] = []

    class _Backend:
        tokenizer = _ListTokenizer()
        tok = _tok(_ListTokenizer())

        async def capture(self, token_ids: Sequence[int], points: Sequence[Any], *, steering_spec: Any = None):  # noqa: ARG002
            asked.extend(points)
            # Keyed the way the real capture keys it: parsed back from the worker's wire key.
            return {to_address(p): torch.tensor([[1.0, 2.0], [3.0, 4.0]]) for p in points}

    means = asyncio.run(capture_turn_means_vllm(cast(VLLMModel, _Backend()), [{"role": "user", "content": "hi"}], 40))

    assert asked == [Address("resid_post", 40)]
    torch.testing.assert_close(means, torch.tensor([[2.0, 3.0]]))
