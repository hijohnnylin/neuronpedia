"""Eager (interp-engine / EagerModel) per-message activation capture for vector readouts.

The vLLM readout path captures mid-layer hidden states via the engine's
`VLLMModel.capture(...)`. This module provides the eager equivalent (same
per-message pooled activations) so readouts work on the EagerModel backend
(CUDA-eager / MPS / CPU): capture the requested site at one or more layers for the
chat-templated conversation and pool per message.

Every capture here is keyed by `CaptureKey`, which carries the reduction as well as the point,
because the pooling happens on this side of the boundary. Two reads at one layer that pool
differently are two results off one forward.

The projection onto a direction (`project_vector`) and the token selection are
backend-agnostic and stay in the caller; this module only returns the per-message
pooled activations, aligned 1:1 with the input conversation.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from interp_engine import Address, EagerModel, SteerSpec, Tokenize, VLLMModel, run_with_cache
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


def capture_turn_means_engine(
    model: EagerModel,
    conversation: list[Any],
    keys: list[CaptureKey],
    specs: list[SteerSpec] | None = None,
    template_kwargs: dict[str, str] | None = None,
) -> dict[CaptureKey, torch.Tensor]:
    """Pooled activations per conversation message -> ``{key: [n_messages, hidden]}``.

    Several layers in one forward, not one forward per layer: vectors fitted at different depths
    are read off the same pass, so the cost of a second vector is memory rather than compute.

    Two keys differing only in ``pool`` likewise share one forward and one cache entry, and pool
    that same tensor twice. The peak is the cache either way -- pooling more ways does not hold the
    activations any longer -- while what multiplies is the pooled result, which is a handful of rows.

    When ``specs`` is provided, the capture runs under engine steering (post-cap
    activations); otherwise it's the unsteered (pre-cap) base model.
    """
    msgs = [{"role": _role(m), "content": _content(m)} for m in conversation]

    full_ids, spans = _per_message_spans(model.tok, msgs, template_kwargs)
    tokens = torch.tensor(full_ids, dtype=torch.long, device=model.device).unsqueeze(0)

    wanted = sorted({(key.site, key.layer) for key in keys})
    ctx = engine_steer(model, specs) if specs else nullcontext()
    with ctx:
        cache = run_with_cache(model, tokens, list(wanted))
    return {key: _pool_spans(cache.get(key.site, key.layer)[0], spans, key.pool) for key in set(keys)}


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


def turn_means_from_generation_capture(
    tok: Tokenize,
    prompt_msgs: list[Any],
    prompt_token_ids: Sequence[int],
    acts: torch.Tensor,
    template_kwargs: dict[str, str] | None = None,
    pool: Pooling = "mean",
) -> torch.Tensor | None:
    """Per-message means from activations captured DURING generation.

    ``acts`` is the generation's own ``[prompt + generated - 1, hidden]`` capture and
    ``prompt_msgs`` is the conversation WITHOUT the turn that was generated, so the
    returned tensor is ``[len(prompt_msgs) + 1, hidden]`` -- aligned 1:1 with the
    conversation the caller projects, exactly like :func:`capture_turn_means_vllm`.

    The prompt was rendered with ``add_generation_prompt=True``, so the generated turn's
    span runs from the end of the previous messages' rendering (i.e. including the
    assistant header, as the re-capture path's template delta does) to the last captured
    row. It excludes the final sampled token, which is never forwarded and therefore has
    no activation -- the one place this differs from re-prefilling the finished text.

    Returns None when the per-message rendering is not a prefix of the prompt actually
    generated from, which is the assumption the spans rest on; the caller falls back to
    re-capturing rather than pooling over misaligned positions.
    """
    msgs = [{"role": _role(m), "content": _content(m)} for m in prompt_msgs]
    full_ids, spans = _per_message_spans(tok, msgs, template_kwargs)
    prompt_ids = [int(t) for t in prompt_token_ids]
    n_rows = int(acts.shape[0])
    if not spans or len(full_ids) >= min(len(prompt_ids), n_rows):
        return None
    if prompt_ids[: len(full_ids)] != full_ids:
        return None
    return _pool_spans(acts, [*spans, (len(full_ids), n_rows)], pool)


async def capture_turn_means_vllm(
    backend: VLLMModel,
    conversation: list[Any],
    keys: list[CaptureKey],
    steering_spec: object | None = None,
    template_kwargs: dict[str, str] | None = None,
) -> dict[CaptureKey, torch.Tensor]:
    """Pooled activations per conversation message on the engine VLLMModel.

    Native-extraction analogue of :func:`capture_turn_means_engine`, and likewise one request
    for every distinct point asked for. When ``steering_spec`` (an engine ``SteeringSpec``) is given,
    captures under steering (post-cap).
    """
    msgs = [{"role": _role(m), "content": _content(m)} for m in conversation]
    full_ids, spans = _per_message_spans(backend.tok, msgs, template_kwargs)

    # Addresses built once and used to ask and to read back: `capture` keys its result by
    # Address, so a second spelling of the same point here is a KeyError rather than a mismatch
    # anything warns about -- which is what a `("resid_post", layer)` tuple left over from the
    # old Point type did.
    #
    # Deduplicated by point rather than by key, since two keys that differ only in pooling read the
    # same captured activations; asking for that point twice would be a second forward.
    points = {(site, layer): Address(site, layer) for site, layer in sorted({(k.site, k.layer) for k in keys})}
    # Per-request steering: capture() registers the steering under the same request id,
    # so this is concurrency-safe (no global steering state clobbered by other requests).
    caps = await backend.capture(full_ids, list(points.values()), steering_spec=steering_spec)
    return {key: _pool_spans(caps[points[(key.site, key.layer)]], spans, key.pool) for key in set(keys)}
