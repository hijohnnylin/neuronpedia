"""Per-message activation capture for vector readouts, on whichever backend is loaded.

Captures the requested point at one or more layers for the chat-templated conversation and pools
per message, through the protocol's ``capture``. Every capture here is keyed by `CaptureKey`,
which carries the reduction as well as the point, because the pooling happens on this side of
the boundary. Two reads at one layer that pool differently are two results off one forward.

The projection onto a direction (`project_vector`) and the token selection are
backend-agnostic and stay in the caller; this module only returns the per-message
pooled activations, aligned 1:1 with the input conversation.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from interp_engine import Address, InterpModel, SteeringSpec, Tokenize
from interp_engine import steer as engine_steer

from neuronpedia_inference.inference_utils.vectors.vector_data import CaptureKey, Pooling


def _role(msg: Any) -> str:
    return msg.role if hasattr(msg, "role") else msg["role"]


def _content(msg: Any) -> str:
    return msg.content if hasattr(msg, "content") else msg["content"]


def _per_message_spans(
    tok: Tokenize,
    msgs: list[dict],
    template_kwargs: dict[str, str] | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Contiguous ``[start, end)`` token spans, one per message, from the engine.

    Each message's span is the block of tokens the model's chat renderer adds when that message
    is appended, so the spans partition the full rendered sequence and align 1:1 with ``msgs``
    (including any system message).

    ``template_kwargs`` has to be whatever the endpoint rendered the generation prompt with, or
    this renders a different conversation than the one that was generated from. Llama 3.1's
    template injects a date into the system block, so a fit that pins ``date_string`` would
    otherwise be measured against today's date here and the fit's date there.

    ``Tokenize.message_partition`` keeps the prefix-delta arithmetic this used to do inline,
    verbatim, for a model rendered by a Jinja template -- so every model served today pools over
    exactly the same token ranges as before. It reports exact message blocks instead for a model
    whose chat format lives in Python (DeepSeek-V4), where the prefix-delta assumption does not
    hold: dropping historical reasoning rewrites earlier turns once a later user turn exists, so
    appending a message is not purely additive and the deltas would land in the wrong places.
    """
    return tok.message_partition(msgs, **(template_kwargs or {}))


def _reduce_span(span: torch.Tensor, pool: Pooling) -> torch.Tensor:
    """One message's token activations collapsed to a single vector.

    Exhaustive over ``Pooling`` on purpose: a member added to that alias without a branch here
    raises rather than falling back to the mean, which would read the new rule as the old one.
    """
    if pool == "mean":
        return span.mean(dim=0)
    if pool == "last":
        return span[-1]
    if pool == "max":
        return span.max(dim=0).values
    raise ValueError(f"pooling {pool!r} is not implemented")


def _pool_spans(acts: torch.Tensor, spans: list[tuple[int, int]], pool: Pooling = "mean") -> torch.Tensor:
    seq, hidden = acts.shape
    pooled: list[torch.Tensor] = []
    for start, end in spans:
        end = min(end, seq)
        if start < end:
            pooled.append(_reduce_span(acts[start:end], pool))
        else:
            # Empty/truncated span (e.g. a blank system turn): contribute zeros so
            # row indices stay aligned with the conversation.
            pooled.append(torch.zeros(hidden, dtype=acts.dtype, device=acts.device))
    return torch.stack(pooled).float().cpu()


async def capture_turn_means(
    model: InterpModel,
    conversation: list[Any],
    keys: list[CaptureKey],
    specs: Sequence[SteeringSpec] | None = None,
    template_kwargs: dict[str, str] | None = None,
) -> dict[CaptureKey, torch.Tensor]:
    """Pooled activations per conversation message -> ``{key: [n_messages, hidden]}``.

    Several layers in one capture, not one per layer: vectors fitted at different depths are read
    off the same pass, so the cost of a second vector is memory rather than compute. Two keys
    differing only in ``pool`` likewise share the pass and pool the same tensor twice.

    Under ``specs`` the capture runs steered (post-cap activations); otherwise it is the unsteered
    (pre-cap) base model. The specs are one ``steer()`` block, which every backend's ``capture``
    honors -- eager through its hooks, a served backend by carrying it on the request.
    """
    msgs = [{"role": _role(m), "content": _content(m)} for m in conversation]
    full_ids, spans = _per_message_spans(model.tok, msgs, template_kwargs)

    # Addresses built once and used to ask and to read back: `capture` keys its result by
    # Address, so a second spelling of the same point here is a KeyError rather than a mismatch
    # anything warns about. Deduplicated by point rather than by key, since two keys that differ
    # only in pooling read the same captured activations.
    points = {(point, layer): Address(point, layer) for point, layer in sorted({(k.point, k.layer) for k in keys})}
    with engine_steer(model, list(specs), prompt_token_ids=full_ids) if specs else nullcontext():
        caps = await model.capture(full_ids, list(points.values()))
    return {key: _pool_spans(caps[points[(key.point, key.layer)]], spans, key.pool) for key in set(keys)}
