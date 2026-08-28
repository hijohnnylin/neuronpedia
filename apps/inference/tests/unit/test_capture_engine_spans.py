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
        **template_kwargs: Any,  # noqa: ARG002
    ):
        return list(range(1, 2 * len(messages) + 1))


class _DateTokenizer:
    """Renders the date into the prefix, the way Llama 3.1's template does.

    Two tokens per message plus one for the date, so a render with the wrong date produces a
    different sequence rather than merely a different string.
    """

    chat_template = "{# present; contents never evaluated here #}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,  # noqa: ARG002
        add_generation_prompt: bool = False,  # noqa: ARG002
        continue_final_message: bool = False,  # noqa: ARG002
        date_string: str = "today",
    ):
        return [900 + len(date_string), *range(1, 2 * len(messages) + 1)]


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


class _CapturingBackend:
    """Stub vLLM backend recording the points it was asked for."""

    tokenizer = _ListTokenizer()

    def __init__(self):
        self.tok = _tok(_ListTokenizer())
        self.asked: list[Any] = []
        self.calls = 0

    async def capture(self, token_ids: Sequence[int], points: Sequence[Any], *, steering_spec: Any = None):  # noqa: ARG002
        self.asked.extend(points)
        self.calls += 1
        # Keyed the way the real capture keys it: parsed back from the worker's wire key.
        return {to_address(p): torch.tensor([[1.0, 2.0], [3.0, 4.0]]) for p in points}


def test_capture_turn_means_vllm_reads_back_the_point_it_asked_for():
    """``capture`` keys its result by ``Address``, so a second spelling of the point is a KeyError.

    A ``("resid_post", layer)`` tuple survived the ``Point`` -> ``Address`` migration here, and
    every steered readout turn runs through this function -- a steered generation's own means
    are post-cap, so the pre-cap ones always come from a second capture like this one.
    """
    backend = _CapturingBackend()
    means = asyncio.run(capture_turn_means_vllm(cast(VLLMModel, backend), [{"role": "user", "content": "hi"}], [40]))

    assert backend.asked == [Address("resid_post", 40)]
    torch.testing.assert_close(means[40], torch.tensor([[2.0, 3.0]]))


def test_pinned_template_kwargs_reach_the_render():
    """An axis that pins a template argument has to be measured against that rendering.

    Llama 3.1's template injects the current date, and the 8B trait fits pinned it. Rendering
    the conversation here without the pin would put the fit's date in the prompt that was
    generated from and today's date in the spans pooled over it -- a silent mismatch that grows
    as the calendar moves.
    """
    msgs = [{"role": "user", "content": "hi"}]
    pinned, _spans = _per_message_spans(_tok(_DateTokenizer()), msgs, {"date_string": "26 Jul 2024"})
    unpinned, _spans = _per_message_spans(_tok(_DateTokenizer()), msgs)
    assert pinned[0] == 900 + len("26 Jul 2024")
    assert pinned != unpinned


def test_a_generation_capture_pooled_with_the_wrong_render_is_refused():
    """The prefix check is what stops a mismatched render from being pooled over.

    The prompt was rendered with the pin; re-rendering without it yields ids that are not a
    prefix of it, so the pooling declines rather than attributing tokens to the wrong turns.
    """
    msgs = [{"role": "user", "content": "hi"}]
    tok = _tok(_DateTokenizer())
    prompt_ids = [900 + len("26 Jul 2024"), 1, 2, 9]
    acts = torch.zeros(len(prompt_ids) + 2, 2)

    assert turn_means_from_generation_capture(tok, msgs, prompt_ids, acts) is None
    assert turn_means_from_generation_capture(tok, msgs, prompt_ids, acts, {"date_string": "26 Jul 2024"}) is not None


def test_capture_turn_means_vllm_asks_for_every_layer_in_one_call():
    """Axes at different layers must not cost one forward each.

    A model can ship six axes across five layers, so looping per axis here would turn one
    extra pass into five. The layers are deduplicated and requested together.
    """
    backend = _CapturingBackend()
    means = asyncio.run(
        capture_turn_means_vllm(cast(VLLMModel, backend), [{"role": "user", "content": "hi"}], [19, 13, 19])
    )

    assert backend.calls == 1
    assert backend.asked == [Address("resid_post", 13), Address("resid_post", 19)]
    assert sorted(means) == [13, 19]
