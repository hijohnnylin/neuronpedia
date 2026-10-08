"""Vector capture pools activations over per-message token spans; these pin the spans.

The spans come from the engine's ``Tokenize.message_partition`` rather than being computed
here, so these tests drive the real engine helper through stub tokenizers. Two properties
matter to a readout: the spans partition the rendered sequence 1:1 with the conversation, and the
ids are real ``int``s (transformers 5 returns a ``BatchEncoding`` from the tokenizing path, and
iterating that yields the string key ``"input_ids"`` — the failure that showed up in pod logs).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import torch
from interp_engine import Address, AddSpec, InterpModel, LayerSteeringSpec, SteeringSpec, Tokenize, to_address
from interp_engine.steer import active_steering

from neuronpedia_inference.inference_utils.vectors.capture_engine import (
    _per_message_spans,
    _pool_spans,
    capture_turn_means,
)
from neuronpedia_inference.inference_utils.vectors.vector_data import CaptureKey, Pooling


def _key(layer: int, pool: Pooling = "mean") -> CaptureKey:
    return CaptureKey(point="resid_post", layer=layer, pool=pool)


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
    """A readout indexes its projection per message, so a gap or overlap misattributes a turn."""
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(4)]
    full_ids, spans = _per_message_spans(_tok(_ListTokenizer()), msgs)

    assert len(spans) == len(msgs)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(full_ids)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


class _CapturingBackend:
    """Stub backend recording the points it was asked for, and the steering open at the time."""

    tokenizer = _ListTokenizer()

    def __init__(self):
        self.tok = _tok(_ListTokenizer())
        self.asked: list[Any] = []
        self.calls = 0
        self.steered: list[Any] = []

    async def capture(self, token_ids: Sequence[int], points: Sequence[Any], *, detach: bool = True):  # noqa: ARG002
        self.asked.extend(points)
        self.calls += 1
        self.steered.append(active_steering(cast(InterpModel, self)))
        # Keyed the way the real capture keys it: parsed back from the worker's wire key.
        return {to_address(p): torch.tensor([[1.0, 2.0], [3.0, 4.0]]) for p in points}


def _means(backend: _CapturingBackend, keys: list[CaptureKey], specs: Any = None) -> dict[CaptureKey, torch.Tensor]:
    return asyncio.run(capture_turn_means(cast(InterpModel, backend), [{"role": "user", "content": "hi"}], keys, specs))


def test_capture_turn_means_reads_back_the_point_it_asked_for():
    """``capture`` keys its result by ``Address``, so a second spelling of the point is a KeyError.

    A ``("resid_post", layer)`` tuple survived the ``Point`` -> ``Address`` migration here, and
    every readout turn runs through this function.
    """
    backend = _CapturingBackend()
    means = _means(backend, [_key(40)])

    assert backend.asked == [Address("resid_post", 40)]
    torch.testing.assert_close(means[_key(40)], torch.tensor([[2.0, 3.0]]))


def test_pinned_template_kwargs_reach_the_render():
    """A vector that pins a template argument has to be measured against that rendering.

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


def test_capture_turn_means_asks_for_every_layer_in_one_call():
    """Axes at different layers must not cost one forward each.

    A model can ship six reads across five layers, so looping per vector here would turn one
    extra pass into five. The layers are deduplicated and requested together.
    """
    backend = _CapturingBackend()
    means = _means(backend, [_key(19), _key(13), _key(19)])

    assert backend.calls == 1
    assert backend.asked == [Address("resid_post", 13), Address("resid_post", 19)]
    assert sorted(key.layer for key in means) == [13, 19]


def test_capture_turn_means_asks_once_for_a_layer_read_two_ways():
    """Two poolings of one layer are one capture, not two.

    What keying the captures by their reduction is for: a vector that pools differently needs its
    own result, and getting it by capturing that layer again would double the cost of the pass.
    """
    backend = _CapturingBackend()
    means = _means(backend, [_key(19), _key(19, "last")])

    assert backend.asked == [Address("resid_post", 19)]
    assert sorted(key.pool for key in means) == ["last", "mean"]
    # The stub's two rows are [1,2] and [3,4] over one message span, so the mean and the last
    # token are different numbers -- which is the point of asking for both.
    torch.testing.assert_close(means[_key(19)], torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(means[_key(19, "last")], torch.tensor([[3.0, 4.0]]))


def test_a_post_cap_capture_runs_under_the_specs_it_was_given():
    """The post-cap read has to see the steer the text was generated under, on any backend.

    The specs open one ``steer()`` block around the capture, which is what a served backend's
    ``capture`` reads its request steering from; nothing is open once it returns.
    """
    spec = SteeringSpec(layers={3: LayerSteeringSpec(operations=[AddSpec(vector=torch.ones(2), scale=1.0)])})
    backend = _CapturingBackend()

    _means(backend, [_key(3)], specs=[spec])
    _means(backend, [_key(3)])

    assert [None if s is None else s.specs for s in backend.steered] == [(spec,), None]
    assert active_steering(cast(InterpModel, backend)) is None


def test_pooling_a_span_the_three_implemented_ways():
    """Each mode reduces the same span differently; a new mode raises rather than defaulting."""
    acts = torch.tensor([[0.0, 6.0], [2.0, 4.0], [1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    spans = [(0, 2), (2, 5)]

    by_pool = {pool: _pool_spans(acts, spans, pool) for pool in cast(list[Pooling], ["mean", "last", "max"])}
    torch.testing.assert_close(by_pool["mean"][0], torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(by_pool["last"][0], torch.tensor([2.0, 4.0]))
    # Per dimension, so the result need not be any one token's activation.
    torch.testing.assert_close(by_pool["max"][0], torch.tensor([2.0, 6.0]))
