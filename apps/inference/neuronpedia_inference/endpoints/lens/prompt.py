import asyncio
import json
import logging
import unicodedata
from collections.abc import AsyncIterator, Mapping
from contextlib import nullcontext
from typing import Any, NamedTuple

import numpy as np
import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from interp_engine import (
    EagerModel,
    GeneratedTurnSpans,
    NoChatTemplateError,
    ResidualBasisUnsupported,
    VLLMModel,
    lens_stream,
    steer,
)
from interp_engine.api import LensSpec
from interp_engine.steer_specs import (
    AblateSpec,
    LayerSteeringSpec,
    NormScaledAddSpec,
    SteeringSpec,
    SwapSpec,
)
from pydantic import BaseModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.lens.lens_loader import (
    JACOBIAN_LENS_KIND,
    JPP_LENS_KIND,
    JacobianLensStore,
    LensKind,
    LoadedJacobianLens,
)
from neuronpedia_inference.endpoints.lens.model_specific import (
    resolve_final_logit_softcap,
)
from neuronpedia_inference.endpoints.lens.residual_spec import (
    BLOCK_OUTPUT,
    LensResidualSpec,
    LensSpaceUnknown,
    block_output_point,
    resolve_residual_spec,
)
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    assert_residual_available,
    assert_steering_available,
    backend_name,
)
from neuronpedia_inference.memory_cost import lens_cost
from neuronpedia_inference.schemas import (
    LensChatMessage,
    LensErrorResponse,
    LensPromptRequest,
    LensSteerToken,
    LensType,
    PublicFrameSchema,
)
from neuronpedia_inference.shared import (
    REQUEST_LOCK_TIMEOUT,
    LoadedModel,
    Model,
    RequestBusy,
    RequestTooLarge,
    budget,
    limiter,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# `chat` used to be rendered through a generic ChatML fallback for a tokenizer with no
# template of its own. The read-outs that came back were computed over `<|im_start|>`
# scaffolding split into ordinary text, which is not a conversation the model has any
# representation of — and `role`/`section` were null throughout, because span metadata
# is derived from the real template. Refusing keeps the raw-text path, which is the one
# a completion model is for.
#
# "no chat template" is now the engine's verdict rather than a field on the tokenizer, so this
# fires only for a model with no chat format at all — not for one (DeepSeek-V4) that defines
# its format in code and used to be refused here for having no Jinja template.
NO_CHAT_TEMPLATE_ERROR = (
    "This model has no chat template, so it cannot accept `chat` input. Send `prompt` (raw text) instead."
)
NO_TOOLS_TEMPLATE_ERROR = (
    "This model's chat template does not read tool definitions, so it cannot accept `tools`. "
    "Send the chat without `tools`."
)

# The lens types that carry each layer through a fitted J_bar, and the lens kind each reads.
J_BAR_LENS_KINDS: dict[LensType, LensKind] = {
    LensType.JACOBIAN_LENS: JACOBIAN_LENS_KIND,
    LensType.JPP_LENS: JPP_LENS_KIND,
}


def _declared_residual(lenses: Mapping[LensType, LoadedJacobianLens]) -> LensResidualSpec | None:
    """The residual the request's lenses declare, else the first loaded lens's.

    Every type in one request reads the same activation, so two lenses that declare
    different ones cannot share it.
    """
    declared = [lens.residual for lens in lenses.values() if lens.residual is not None]
    if any(r != declared[0] for r in declared[1:]):
        raise LensSpaceUnknown("The requested lenses read different residuals; request them apart.")
    if declared:
        return declared[0]
    for kind in J_BAR_LENS_KINDS.values():
        lens = kind.store.get()
        if lens is not None and lens.residual is not None:
            return lens.residual
    return None


def _lens_spec(lens_type: LensType, layers: list[int]) -> LensSpec:
    """The engine's spec for one lens type: a J_bar type names its set, the logit lens none."""
    kind = J_BAR_LENS_KINDS.get(lens_type)
    if kind is None:
        return LensSpec(layers=layers)
    return LensSpec(layers=layers, jacobian=True, jacobian_set=kind.engine_set)


# --------------------------------------------------------------------------- #
# Streamed NDJSON frames
#
# These are not a response body FastAPI can document -- the endpoint emits them one JSON
# object per line -- so unlike the request models they stay here, next to the generator.
#
# They are `PublicFrameSchema` rather than `BaseSchema`, so their field names go out exactly
# as written instead of being camelCased. The webapp forwards these frames verbatim into
# `/api/lens/prompt` and into the stored share blobs, so the names below are the public
# contract; see the note on the base class. `test_lens_frame_contract.py` pins them.
# --------------------------------------------------------------------------- #


class LensTypeSlice(PublicFrameSchema):
    """Lens read-out for one (position, lens_type).

    All token references are STRINGS (decoded), never ids.
    """

    type: LensType
    # [n_layers, top_n]
    top_tokens: list[list[str]]
    top_probs: list[list[float]]


class LensMetaMessage(PublicFrameSchema):
    """First streamed message: the shared request context."""

    kind: str = "meta"
    model: str
    types: list[LensType]
    # Selected layers per lens type (identical for every position).
    layers_by_type: dict[str, list[int]]
    top_n: int
    prompt_len: int
    num_completion_tokens: int
    temperature: float
    prepend_bos: bool
    # Number of leading prompt positions whose read-outs were reused from the
    # client's cache (skipped this run). Token messages are only emitted for
    # positions >= reuse_len; the client keeps its prior results for the rest.
    reuse_len: int = 0
    # The layers `/v1/lens/oracle` reads on this server; empty when it has no oracle lens.
    oracle_layers: list[int] = []


class LensPromptToken(PublicFrameSchema):
    """A single chat-formatted prompt token (no lens read-out)."""

    position: int
    token: str
    # The token id, echoed so the client can send it back as `cached_token_ids`
    # on the next turn for prefix-reuse matching.
    id: int
    is_generated: bool = False
    # True on the 2nd..nth position of a character split across tokens, whose whole
    # glyph `token` repeats at every contributing position (see
    # `_decode_display_tokens`). Anything rebuilding TEXT from a token stream must
    # skip these, or one emoji comes back N times.
    is_char_continuation: bool = False
    # Per-token chat-span metadata (the single source of truth for message
    # boundaries, computed server-side by the engine's Tokenize.message_spans).
    # Null on raw-text / reproduction requests that carry no chat messages.
    message_index: int | None = None
    role: str | None = None
    channel: str | None = None
    section: str | None = None


class LensPromptTokensMessage(PublicFrameSchema):
    """Emitted right after `meta` and before inference begins.

    Carries the chat-formatted prompt tokens (no lens read-outs) so the client
    can render the full conversation structure (user turn + assistant scaffold)
    immediately, instead of waiting for generation to finish.
    """

    kind: str = "prompt"
    tokens: list[LensPromptToken]


class LensTokenMessage(PublicFrameSchema):
    """One per token position: the token plus its per-type lens slices."""

    kind: str = "token"
    position: int
    token: str
    # The token id, echoed so the client can send it back as `cached_token_ids`
    # on the next turn for prefix-reuse matching.
    id: int
    is_generated: bool
    results: list[LensTypeSlice]
    # See LensPromptToken: true where `token` is repeating a character this position
    # only holds part of, so text reconstruction skips it.
    is_char_continuation: bool = False
    # Per-token chat-span metadata (see LensPromptToken). Carried on token
    # messages too so a full re-render from `tokens` alone groups correctly (the
    # frontend replays shared runs from stored tokens, not the prompt message).
    message_index: int | None = None
    role: str | None = None
    channel: str | None = None
    section: str | None = None


class LensDoneMessage(PublicFrameSchema):
    """Final streamed message."""

    kind: str = "done"
    seq_len: int
    prompt_len: int
    vocab_size: int
    completion: str


class LensErrorMessage(PublicFrameSchema):
    """Emitted instead of `done` when the run fails partway through.

    A model rather than an inline dict so it is covered by the frame contract test like
    every other frame; the stream has already started by the time this is reached, so the
    failure cannot be reported as a status code.
    """

    kind: str = "error"
    error: str


# --------------------------------------------------------------------------- #
# Token helpers (ported from the jlens demo vis)
# --------------------------------------------------------------------------- #


def _decode_token(tokenizer, token_id: int, cache: dict[int, str]) -> str:
    """Decode a single token id to its string, memoised per request.

    We intentionally key identity internally by int id (distinct ids can decode
    to the same string), and only convert to strings at serialization time.
    """
    cached = cache.get(token_id)
    if cached is None:
        cached = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        cache[token_id] = cached
    return cached


# The Unicode replacement character produced when a token holds only part of a
# multi-byte (e.g. emoji) codepoint and is decoded in isolation.
_REPLACEMENT_CHAR = "\ufffd"
# Safety cap on how many adjacent tokens we'll join trying to complete a split
# multi-byte character before giving up.
_MAX_MULTI_TOKEN_CHAR = 8


class _DisplayToken(NamedTuple):
    """One position's display string, plus whether it is repeating the previous one's.

    ``continuation`` is what makes the repetition reversible: the string alone cannot be,
    because two adjacent split emoji look exactly like one repeated across its fragments.
    """

    token: str
    continuation: bool


def _decode_display_tokens(tokenizer, token_ids: list[int], cache: dict[int, str]) -> list[_DisplayToken]:
    """Per-position display strings, repairing characters split across tokens.

    A single emoji (or other multi-byte codepoint) is often split across several
    tokens; decoded individually each fragment is just a replacement char (`),
    so the glyph never shows. Here we detect a run of such fragments, decode the
    run together to recover the real character, and assign that combined string
    to EVERY position in the run (so the emoji shows at each contributing token
    rather than a row of `).

    Every position after the first in such a run is flagged a ``continuation``, so a
    consumer rebuilding text emits the character once while the chips still each show it.
    """
    n = len(token_ids)
    out: list[_DisplayToken] = [_DisplayToken("", False)] * n
    i = 0
    while i < n:
        solo = _decode_token(tokenizer, int(token_ids[i]), cache)
        if _REPLACEMENT_CHAR not in solo:
            out[i] = _DisplayToken(solo, False)
            i += 1
            continue
        # Broken fragment: greedily extend the run until it decodes cleanly.
        j = i
        combined = solo
        while _REPLACEMENT_CHAR in combined and j + 1 < n and (j - i) < _MAX_MULTI_TOKEN_CHAR:
            j += 1
            combined = tokenizer.decode(
                [int(token_ids[k]) for k in range(i, j + 1)],
                clean_up_tokenization_spaces=False,
            )
        if _REPLACEMENT_CHAR not in combined:
            for k in range(i, j + 1):
                out[k] = _DisplayToken(combined, k > i)
            i = j + 1
        else:
            # Unrecoverable; leave the lone replacement char for this position. It stands
            # for itself rather than continuing anything, so it is not a continuation.
            out[i] = _DisplayToken(solo, False)
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Non-word token filtering (mirrors the frontend `isWordLikeToken`)
# --------------------------------------------------------------------------- #


def _is_word_like_token(token: str) -> bool:
    """Whether ``token`` is "word-like" (kept when non-word filtering is on).

    This MUST mirror the frontend `isWordLikeToken` (jlens-token-popup.tsx): a
    token is word-like when, after trimming, it is non-empty, not a special
    token (``<|...|>`` or ``<...>``), and every Unicode character is a letter or
    number (categories ``L``/``N``) — with ``'``, ``-``, ``’`` allowed only in
    interior positions.
    """
    stripped = token.strip()
    if stripped == "":
        return False
    if "<|" in stripped or (stripped.startswith("<") and stripped.endswith(">")):
        return False
    chars = list(stripped)
    n = len(chars)
    for pos, ch in enumerate(chars):
        if unicodedata.category(ch)[0] in ("L", "N"):
            continue
        if 0 < pos < n - 1 and ch in ("'", "-", "\u2019"):
            continue
        return False
    return True


# Cache: id(tokenizer) -> CPU bool tensor ``[vocab]`` (True = word-like, keep).
# Built once per tokenizer (a full-vocab decode + classify) and reused across
# requests, mirroring `_DECODE_INDEX_CACHE`.
_WORD_MASK_CACHE: dict[int, torch.Tensor] = {}


def _readout_vocab_size(tokenizer, model=None) -> int:
    """Vocab dim of the model's logits / unembed, not the tokenizer's nominal size.

    Llama-3.x pads the embedding table to a multiple of 256 (``128256``) while
    ``tokenizer.vocab_size`` stays at ``128000``. The word-mask and done-message
    ``vocab_size`` must match the live logits dim. Prefer a model-reported size
    when available; otherwise take ``max(len(tokenizer), vocab_size)`` so added
    special tokens / padding are not dropped.
    """
    if model is not None:
        vs = getattr(model, "vocab_size", None)
        if isinstance(vs, int) and vs > 0:
            return int(vs)
        cfg = getattr(model, "config", None)
        if cfg is not None:
            text_cfg = getattr(cfg, "text_config", None) or cfg
            cfg_vs = getattr(text_cfg, "vocab_size", None)
            if isinstance(cfg_vs, int) and cfg_vs > 0:
                return int(cfg_vs)
    tok_vs = int(getattr(tokenizer, "vocab_size", 0) or 0)
    try:
        tok_len = int(len(tokenizer))
    except Exception:  # noqa: BLE001
        tok_len = 0
    size = max(tok_vs, tok_len)
    if size <= 0:
        raise ValueError("Could not resolve readout vocab size from tokenizer/model")
    return size


def _word_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """Bool tensor ``[vocab_size]`` marking word-like token ids (CPU, cached).

    Sized to the read-out's vocab dimension (which can exceed the tokenizer's
    nominal vocab due to padding); ids that fail to decode or are non-word are
    left ``False``.
    """
    key = id(tokenizer)
    cached = _WORD_MASK_CACHE.get(key)
    if cached is not None and cached.shape[0] == vocab_size:
        return cached
    flags = torch.zeros(vocab_size, dtype=torch.bool)
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:  # noqa: BLE001
            continue
        if _is_word_like_token(decoded):
            flags[token_id] = True
    _WORD_MASK_CACHE[key] = flags
    return flags


# --------------------------------------------------------------------------- #
# Tokenization (raw text or chat)
# --------------------------------------------------------------------------- #


def _encode_raw_text(tokenizer, text: str, prepend_bos: bool) -> list[int]:
    bos = tokenizer.bos_token
    if prepend_bos and bos and not text.startswith(bos):
        text = bos + text
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _coerce_token_ids(ids) -> list[int]:
    """Normalise the many shapes ``apply_chat_template`` can return into a flat
    ``list[int]``.

    Depending on the transformers version it may return a ``list[int]``, a
    (possibly batched) tensor, or a dict/``BatchEncoding`` (in which case
    ``list(ids)`` would wrongly yield the string keys, e.g. ``"input_ids"``).
    """
    # dict / BatchEncoding -> pull out input_ids
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    # tensor / ndarray -> python list (drop a leading batch dim if present)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    # batched nested list [[...]] -> first row
    if len(ids) > 0 and isinstance(ids[0], list | tuple):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _chat_template_kwargs(tok, request: LensPromptRequest) -> dict:
    """Select the template kwargs to pass for this request (only those this model reads).

    A renderer that doesn't know a kwarg will ignore or reject it, so each one is gated on the
    engine's answer for this model. We ask the engine rather than grepping
    ``tokenizer.chat_template`` ourselves because a model whose chat format lives in code
    (DeepSeek-V4) has no template source to grep — see ``Tokenize.accepted_template_kwargs``.
    Kept as a helper so ``build_token_ids`` and the span computation render with identical
    arguments (and therefore identical token ids/positions).
    """
    # gpt-oss (harmony) has no on/off thinking switch — it uses `reasoning_effort`
    # (low/medium/high). Map our boolean onto low/high, only where it is read.
    accepted = tok.accepted_template_kwargs(("enable_thinking", "preserve_thinking", "reasoning_effort"))
    kwargs: dict = {}
    if "enable_thinking" in accepted:
        kwargs["enable_thinking"] = request.enable_thinking
    if "preserve_thinking" in accepted:
        kwargs["preserve_thinking"] = request.preserve_thinking
    if "reasoning_effort" in accepted:
        kwargs["reasoning_effort"] = "high" if request.enable_thinking else "low"
    # The route refuses tools for a renderer that does not read them, so here they always go in.
    if request.tools:
        kwargs["tools"] = request.tools
    return kwargs


def _template_message(m: LensChatMessage) -> dict[str, Any]:
    """One chat message in the Hugging Face template shape (tool calls under ``function``)."""
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "type": "function",
                **({"id": c.id} if c.id else {}),
                "function": {"name": c.name, "arguments": c.arguments or {}},
            }
            for c in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    return out


def _chat_args(tok, request: LensPromptRequest) -> tuple[list[dict[str, Any]], bool, bool, dict]:
    """Return ``(messages, add_generation_prompt, continue_final_message, template_kwargs)``.

    If the final message is an assistant turn, treat it as a PREFILL: keep that turn open (no
    end-of-turn token, no fresh assistant scaffold) so generation continues from the prefilled
    text rather than starting a new assistant turn after it.

    A final assistant turn with tool calls is rendered closed instead: transformers keeps a turn
    open by cutting the render at the end of its content, which would cut off the calls.
    """
    messages = [_template_message(m) for m in (request.chat or [])]
    ends_with_assistant = len(messages) > 0 and messages[-1]["role"] == "assistant"
    is_prefill = ends_with_assistant and not messages[-1].get("tool_calls")
    return (
        messages,
        (not ends_with_assistant),
        is_prefill,
        _chat_template_kwargs(tok, request),
    )


def build_token_ids(model, request: LensPromptRequest) -> list[int]:
    """Build input token ids from either a raw prompt or a chat conversation.

    Chat rendering goes through the engine's ``Tokenize`` rather than the tokenizer directly:
    it is the layer that knows whether this model renders chat from a Jinja template or from a
    code formatter, and reading ``tokenizer.chat_template`` here would refuse a model that
    renders chat perfectly well.
    """
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized")

    if request.chat is not None:
        tok = model.tok
        if not tok.has_chat_template():
            raise NoChatTemplateError(NO_CHAT_TEMPLATE_ERROR)
        messages, add_generation_prompt, is_prefill, kwargs = _chat_args(tok, request)
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=is_prefill,
            **kwargs,
        )
        return _coerce_token_ids(ids)

    return _encode_raw_text(tokenizer, request.prompt or "", request.prepend_bos)


def compute_prompt_spans(model, request: LensPromptRequest, prompt_token_ids: list[int]) -> tuple[list, bool]:
    """Return ``(spans, is_prefill)`` for the chat prompt, or ``([], False)`` if unavailable.

    Uses the engine's ``Tokenize.message_spans`` (single source of truth for message boundaries),
    rendered with the SAME args ``build_token_ids`` used, then verified to align 1:1 with
    ``prompt_token_ids``. On any mismatch (raw-text request, truncation) we return no spans and
    the frontend renders the tokens plainly. A chat request against a model that cannot render
    chat never reaches here — it is rejected up front — so the check below is only a guard.
    """
    if request.chat is None:
        return [], False
    if model.tokenizer is None:
        return [], False
    try:
        tok = model.tok
        if not tok.has_chat_template():
            return [], False
        messages, add_generation_prompt, is_prefill, kwargs = _chat_args(tok, request)
        spans = tok.message_spans(
            messages,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=is_prefill,
            **kwargs,
        )
    except Exception:  # noqa: BLE001 - template rendering is tokenizer-dependent
        logger.exception("Failed to compute lens prompt spans")
        return [], False
    # Only trust the spans if they align exactly with the tokenized prompt.
    if [int(s.token_id) for s in spans] != [int(t) for t in prompt_token_ids]:
        return [], False
    return spans, is_prefill


# --------------------------------------------------------------------------- #
# Incremental generation + residual capture (KV-cached, forward hooks)
# --------------------------------------------------------------------------- #

# One streamed position: (token_id, is_generated, {layer: residual[d_model]}).
ResidualStep = tuple[int, bool, dict[int, torch.Tensor]]
# Positions that became available together (one prefill, or one decode-time drain).
# Batching is what the read-out is staged on: a batch's per-layer Jacobian transport
# is one matmul regardless of how many positions it holds, so handing out positions
# in groups instead of one at a time is the difference between reading every J_bar
# once and reading it once per position. A batch never delays a position: it holds
# exactly what the backend had ready at that moment.
ResidualBatch = list[ResidualStep]


# --------------------------------------------------------------------------- #
# Steering (readout-vector injection)
# --------------------------------------------------------------------------- #

# Cache: id(tokenizer) -> {exact decoded string: [token ids]}. Built once per
# tokenizer (a full-vocab decode) and reused across steer requests.
_DECODE_INDEX_CACHE: dict[int, dict[str, list[int]]] = {}


def _decoded_string_to_ids(tokenizer) -> dict[str, list[int]]:
    """Reverse map from a token's exact decoded string to the vocab id(s).

    Decoded with ``clean_up_tokenization_spaces=False`` so the keys match the
    read-out slice strings the client sends back verbatim (whitespace included).
    """
    cache_key = id(tokenizer)
    cached = _DECODE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    index: dict[str, list[int]] = {}
    for token_id in range(int(vocab_size)):
        try:
            decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:  # noqa: BLE001
            continue
        index.setdefault(decoded, []).append(token_id)
    _DECODE_INDEX_CACHE[cache_key] = index
    return index


# Longest prefix the closest-token search tries. It bounds the search cost for any
# input length, and is longer than any real vocab entry.
_MAX_SUGGEST_PREFIX_CHARS = 256


class SteerTokenNotFound(ValueError):
    """A steer or swap token string that is not exactly one vocab entry."""

    def __init__(self, token: str, suggestion: str | None) -> None:
        """Store the token and the closest vocab entry, and build the message."""
        message = f"{token!r} is not a single token in this model's vocabulary."
        if suggestion is not None:
            message += f" Closest token: {suggestion!r}."
        super().__init__(message)
        self.token = token
        self.suggestion = suggestion


def _suggest_steer_token(index: dict[str, list[int]], token: str) -> str | None:
    """Return the vocab entry that is the longest prefix of ``token``, or None.

    Tries ``token`` as typed, then with exactly one leading space. The score is the number
    of matched characters after the leading whitespace, so ``" ants"`` wins over ``"ant"``
    for input ``"ants"``. On a tie, the form as typed wins. Cost is at most
    ``2 * _MAX_SUGGEST_PREFIX_CHARS`` dict lookups.
    """
    if not token.strip():
        return None
    best: str | None = None
    best_score = 0
    for form in dict.fromkeys([token, " " + token.lstrip()]):
        lead = len(form) - len(form.lstrip())
        for end in range(min(len(form), _MAX_SUGGEST_PREFIX_CHARS), lead, -1):
            prefix = form[:end]
            if prefix in index:
                if end - lead > best_score:
                    best, best_score = prefix, end - lead
                break
    return best


def _resolve_steer_token_id(index: dict[str, list[int]], token: str) -> int:
    """Resolve an exact decoded string to a single vocab id.

    No near match is used, so the model steers on the token the caller named. When there
    is no exact match, raise ``SteerTokenNotFound`` with the closest token. True collisions
    (multiple ids -> same string) are rare; we take the lowest id (their unembedding
    directions are near-identical)."""
    ids = index.get(token)
    if not ids:
        raise SteerTokenNotFound(token, _suggest_steer_token(index, token))
    return int(min(ids))


async def _unembed_vectors_by_id(model: LoadedModel, token_ids: list[int]) -> dict[int, torch.Tensor]:
    """``{token_id: unembedding_row [d_model] float32}`` for jlens steering, from the protocol.

    ``unembed_rows`` is each backend's own ``W_U`` (``lm_head``, or the tied embedding where there is
    no head) and refuses an id outside the vocab before indexing anything on a device.
    """
    unique = list(dict.fromkeys(int(t) for t in token_ids))
    rows = (await model.unembed_rows(unique)).float()
    return {tid: rows[i] for i, tid in enumerate(unique)}


async def _build_steer_deltas(
    model,
    lenses: Mapping[LensType, LoadedJacobianLens],
    steer_tokens: list[LensSteerToken],
    steer_layers: list[int],
) -> dict[int, torch.Tensor]:
    """Build the per-layer unit direction to inject, summed across steer tokens.

    For a ``JACOBIAN_LENS`` or ``JPP_LENS`` token at a layer ``l`` its lens fits, the
    direction is ``J_bar_l^T @ w_t`` (equivalently ``w_t @ J_bar_l``), the residual-space
    direction whose readout is the token; otherwise the plain unembedding
    direction ``w_t``. Each per-layer direction is unit-normalized before
    summing so multiple tokens reinforce sensibly. The unembedding rows are the
    backend's ``unembed_rows``.

    Wherever ``J_bar`` lives is where the multiply happens: in the vLLM worker via
    ``lens_transport`` when the lens is resident there, otherwise on the lens's
    ``transport_device``. What it must not do is follow the unembedding rows, which arrive
    from vLLM as CPU tensors -- that meant widening a ``d_model**2`` ``J_bar`` to float32 on
    the host and reading it back once per layer per token, 9.3s before the response even
    started for a 63-layer swap on Qwen3.6-27B, which a swap pays twice (once for the source
    directions, once for the target). The matmul is done at the lens dtype, which the
    unit-normalization below makes moot anyway.
    """
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized")
    index = _decoded_string_to_ids(tokenizer)
    resolved = [(_resolve_steer_token_id(index, spec.token), spec.type) for spec in steer_tokens]
    w_by_id = await _unembed_vectors_by_id(model, [tid for tid, _ in resolved])
    if not w_by_id:
        return {}

    # Returned where the rows arrived, so the steering hooks see what they always saw.
    out_device = next(iter(w_by_id.values())).device

    # [n_tokens, d_model] in `resolved` order, so one round trip per lens covers every layer
    # when that lens is in the worker. `fitted` says which layers had a fitted J_bar; the rest
    # come back untouched, which is what a non-Jacobian token wants anyway.
    transported: dict[LensType, tuple[torch.Tensor, list[bool]]] = {}
    for lens_type, lens in lenses.items():
        if lens.worker_resident and any(t == lens_type for _, t in resolved):
            stacked = torch.stack([w_by_id[tid] for tid, _ in resolved], dim=0)
            engine_set = J_BAR_LENS_KINDS[lens_type].engine_set
            if engine_set == JACOBIAN_LENS_KIND.engine_set:
                transported[lens_type] = await model.lens_transport(stacked, steer_layers)
            else:
                transported[lens_type] = await model.lens_transport(stacked, steer_layers, jacobian_set=engine_set)

    deltas: dict[int, torch.Tensor] = {}
    for layer_index, layer in enumerate(steer_layers):
        acc: torch.Tensor | None = None
        for token_index, (token_id, lens_type) in enumerate(resolved):
            w = w_by_id[token_id]
            lens = lenses.get(lens_type)
            if lens is None:
                direction = w
            elif lens_type in transported:
                transported_by_layer, fitted = transported[lens_type]
                direction = transported_by_layer[layer_index][token_index] if fitted[layer_index] else w
            elif layer in lens.jacobians:
                j_bar = lens.jacobian_on(layer, lens.transport_device or out_device)
                direction = (w.to(device=j_bar.device, dtype=j_bar.dtype) @ j_bar).float()  # J_bar^T @ w
            else:
                direction = w
            norm = torch.linalg.vector_norm(direction)
            if norm > 0:
                direction = direction / norm
            acc = direction if acc is None else acc + direction.to(acc.device)
        if acc is not None:
            deltas[layer] = acc.to(out_device)
    return deltas


# Per-layer cap on the additive steering vector, as a fraction of the
# per-position residual norm. Steering is applied at every selected layer, so
# the effect compounds; capping each step keeps a strong/multi-layer request
# from driving the residual (and hence the logits) to inf/nan.
_MAX_STEER_INJECTION_FRACTION = 1.0


def _build_lens_steering_spec(
    steer_deltas: dict[int, torch.Tensor] | None,
    steer_strength: float,
    steer_ablate: bool,
    swap_deltas: dict[int, torch.Tensor] | None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
    n_streams: int = 1,
) -> SteeringSpec | None:
    """The lens intervention as the engine's own ``SteeringSpec``, or None when there is none.

    Swap wins over additive/ablation. The ops are the engine's -- ``NormScaledAddSpec`` is the
    lens's norm-scaled, capped steer, ``AblateSpec`` and ``SwapSpec`` the other two -- so every
    backend's ``steer()`` applies the same arithmetic. A zero-norm direction is skipped rather than
    refused by the engine.

    Every spec names the point the read-out is taken at, which is what lets an intervention reach a
    hyper-connection trunk: ``resid_post`` does not exist there, so a spec that said nothing had
    nothing to aim at. On a conventional trunk ``point_name`` returns exactly that default, so the two
    trunks share one code path rather than branching here.
    """
    swapping = bool(swap_deltas) and bool(steer_deltas)
    steering = bool(steer_deltas) and (steer_strength != 0.0 or steer_ablate)
    layers: dict[int, LayerSteeringSpec] = {}
    if swapping and steer_deltas is not None and swap_deltas is not None:
        for layer, tgt in swap_deltas.items():
            src = steer_deltas.get(layer)
            if src is not None and src.norm() != 0 and tgt.norm() != 0:
                layers[int(layer)] = LayerSteeringSpec(operations=[SwapSpec(vector=src.float(), target=tgt.float())])
    elif steering and steer_deltas is not None:
        for layer, delta in steer_deltas.items():
            if delta.norm() == 0:
                continue
            op = (
                AblateSpec(vector=delta.float())
                if steer_ablate
                else NormScaledAddSpec(
                    vector=delta.float(), strength=steer_strength, max_fraction=_MAX_STEER_INJECTION_FRACTION
                )
            )
            layers[int(layer)] = LayerSteeringSpec(operations=[op])
    if not layers:
        return None
    return SteeringSpec(layers=layers, point=residual.point_name(n_streams), stream=residual.write_stream)


def _lens_intervention(
    model: LoadedModel,
    prompt_token_ids: list[int],
    *,
    steer_deltas: dict[int, torch.Tensor] | None,
    steer_strength: float,
    steer_ablate: bool,
    swap_deltas: dict[int, torch.Tensor] | None,
    steer_generated: bool,
    bos_token_id: int | None,
    residual: LensResidualSpec,
):
    """The request's intervention as a ``steer()`` block to open around the capture, or a no-op.

    The spec from :func:`_build_lens_steering_spec`, BOS positions as the ``position_mask`` and
    ``generated=steer_generated`` -- the scoping the eager arm applies by hand. The engine carries
    all three to whichever backend runs the capture, so this is the one place the lens says what its
    intervention is.
    """
    n_streams = model.residual_basis.n_streams
    spec = _build_lens_steering_spec(steer_deltas, steer_strength, steer_ablate, swap_deltas, residual, n_streams)
    if spec is None:
        return nullcontext()
    bos_positions = [i for i, t in enumerate(prompt_token_ids) if bos_token_id is not None and t == bos_token_id]
    return steer(
        model,
        spec,
        prompt_token_ids=prompt_token_ids,
        position_mask=bos_positions or None,
        generated=steer_generated,
    )


def _common_prefix_len(token_ids: list[int], cached_token_ids: list[int]) -> int:
    """Length of the longest common leading run of two token-id lists."""
    n = 0
    for a, b in zip(token_ids, cached_token_ids):
        if a != b:
            break
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Slice assembly (ported from the jlens demo vis)
# --------------------------------------------------------------------------- #


def _slice(
    lens_type: LensType, tokenizer, decode_cache: dict[int, str], top_idx: torch.Tensor, top_probs: torch.Tensor
) -> LensTypeSlice:
    """One type's read-out at one position, from the engine's ``[n_layers, top_n]`` top-k."""
    top_tokens = [
        [_decode_token(tokenizer, int(token_id), decode_cache) for token_id in row]
        for row in top_idx.detach().cpu().tolist()
    ]
    # Rounded in float64, on the CPU (MPS has no float64): a rounded float32 widens to a noisy
    # float64 on `.tolist()`. 4 decimals is below what the client renders.
    top_probs_np = top_probs.detach().cpu().double().numpy()
    return LensTypeSlice(type=lens_type, top_tokens=top_tokens, top_probs=np.round(top_probs_np, 4).tolist())


def _select_layers(
    lens_type: LensType,
    n_layers: int,
    lens: LoadedJacobianLens | None,
    layers: list[int],
) -> list[int]:
    """Resolve the layers to read out for a lens type.

    Empty ``layers`` = all available layers; otherwise the intersection of the
    requested layers with the available ones. The final layer is ALWAYS included
    (decoded directly as the model's true output).
    """
    final_layer = n_layers - 1
    if lens_type in J_BAR_LENS_KINDS and lens is not None:
        available = list(lens.source_layers)
    else:
        available = list(range(n_layers))

    if layers:
        wanted = set(layers)
        selected = [layer for layer in available if layer in wanted]
    else:
        selected = list(available)

    if final_layer not in selected:
        selected.append(final_layer)
    return sorted(set(selected))


# --------------------------------------------------------------------------- #
# Message assembly
# --------------------------------------------------------------------------- #


async def _build_messages(
    model,
    request: LensPromptRequest,
    requested_types: list[LensType],
    lenses: Mapping[LensType, LoadedJacobianLens],
    softcap: float | None,
    layers_by_type: dict[LensType, list[int]],
    prompt_token_ids: list[int],
    reuse_len: int = 0,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    residual: LensResidualSpec = BLOCK_OUTPUT,
) -> AsyncIterator[BaseModel]:
    """Yield the ordered stream of messages: meta -> prompt tokens -> token* -> done.

    The read-out is the engine's ``generate_with_lens``: the prefill, then one step per generated
    token, each position's top-k as soon as it is read. What is left here is the messages: token
    text, chat spans, and multi-byte characters split across tokens.

    ``reuse_len`` is the number of leading prompt positions the client already has read-outs for
    (the token-id common prefix). The whole prompt is still prefilled, since later positions depend
    on it, but those positions are not read out.
    """
    tokenizer = model.tokenizer
    decode_cache: dict[int, str] = {}
    prompt_len = len(prompt_token_ids)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)

    # Per-token chat spans (single source of truth for message boundaries):
    # prompt positions come from the engine's message_spans (verified to align
    # with the tokenized prompt); generated positions from an incremental tracker
    # that follows the assistant turn (harmony channels + generic turn-end). Both
    # are None when there is no chat context (raw-text / reproduction requests),
    # in which case the frontend renders the tokens plainly.
    prompt_spans, is_prefill = compute_prompt_spans(model, request, prompt_token_ids)
    gen_message_index = (len(request.chat) - 1) if (is_prefill and request.chat) else None
    # ``for_prompt`` reads the prompt's trailing scaffold so a thinking-enabled template (which
    # ends on a dangling <think>) has its generated reasoning channelled correctly.
    gen_tracker = GeneratedTurnSpans.for_prompt(
        tokenizer,
        [span.token_str for span in prompt_spans],
        message_index=gen_message_index,
    )

    def _span_fields(pos: int, token_id: int, is_generated: bool, token_str: str) -> dict:
        """Span metadata for one position. Must be called at most once per
        generated position, in generation order (it advances the tracker)."""
        if is_generated:
            span = gen_tracker.process(pos, int(token_id), token_str)
        elif 0 <= pos < len(prompt_spans):
            span = prompt_spans[pos]
        else:
            return {}
        return {
            "message_index": span.message_index,
            "role": span.role,
            "channel": span.channel,
            "section": span.section,
        }

    from neuronpedia_inference.endpoints.lens.oracle import OracleStore

    yield LensMetaMessage(
        oracle_layers=list(OracleStore.layers()),
        model=request.model,
        types=requested_types,
        layers_by_type={t.value: layers_by_type[t] for t in requested_types},
        top_n=request.top_n,
        prompt_len=prompt_len,
        num_completion_tokens=request.num_completion_tokens,
        temperature=request.temperature,
        prepend_bos=request.prepend_bos,
        reuse_len=reuse_len,
    )

    # Emit the chat-formatted prompt tokens up-front, before running any
    # inference. This lets the client render the conversation structure (and the
    # assistant turn scaffold) right away rather than only after generation
    # completes. Decoding the already-tokenized prompt is cheap (no model
    # forward), so this first message arrives almost immediately. The full prompt
    # is always sent (including reused positions) so the client can render the
    # whole conversation; only the per-position lens read-out below is skipped.
    prompt_display = _decode_display_tokens(tokenizer, prompt_token_ids, decode_cache)
    yield LensPromptTokensMessage(
        tokens=[
            LensPromptToken(
                position=pos,
                token=prompt_display[pos].token,
                id=int(token_id),
                is_generated=False,
                is_char_continuation=prompt_display[pos].continuation,
                **_span_fields(pos, int(token_id), False, prompt_display[pos].token),
            )
            for pos, token_id in enumerate(prompt_token_ids)
        ]
    )

    completion_ids: list[int] = []
    # Buffer for a run of tokens that decode to lone replacement chars (the
    # fragments of one multi-byte char, e.g. an emoji split across tokens). We
    # hold their messages until the run decodes cleanly, then emit each with the
    # recovered character so the emoji shows at every contributing position.
    pending: list[LensTokenMessage] = []

    def _emit(entry: LensTokenMessage, token_str: str, *, continuation: bool = False) -> LensTokenMessage:
        entry.token = token_str
        entry.is_char_continuation = continuation
        return entry

    def _flush_pending_as_is() -> list[LensTokenMessage]:
        # An unrecoverable run: each position keeps its own lone replacement char, so none of
        # them is repeating a neighbour's character.
        flushed = [_emit(p, _decode_token(tokenizer, p.id, decode_cache)) for p in pending]
        pending.clear()
        return flushed

    specs = [_lens_spec(t, layers_by_type[t]) for t in requested_types]
    word_mask: torch.Tensor | None = None
    if request.filter_non_word_tokens:
        try:
            word_mask = _word_token_mask(tokenizer, _readout_vocab_size(tokenizer, model))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to build the word-token mask for the lens read-out")
    # A lens in the vLLM worker is already installed there, and so is a named set anywhere.
    # Otherwise the Jacobian lens's matrices go with the call, each layer where placement left it.
    jacobians = None
    jlens = lenses.get(LensType.JACOBIAN_LENS)
    if jlens is not None and not jlens.worker_resident:
        jacobians = jlens.placed_jacobians()
    intervention = _lens_intervention(
        model,
        prompt_token_ids,
        steer_deltas=steer_deltas,
        steer_strength=steer_strength,
        steer_ablate=steer_ablate,
        swap_deltas=swap_deltas,
        steer_generated=steer_generated,
        bos_token_id=bos_token_id,
        residual=residual,
    )

    position = reuse_len
    vocab_size = 0
    # The block is read when the engine registers the request, on the first step, so it is held
    # for the whole stream.
    with intervention:
        async for step in model.generate_with_lens(
            prompt_token_ids,
            specs,
            point=residual.point_name(model.residual_basis.n_streams),
            top_n=request.top_n,
            max_tokens=request.num_completion_tokens,
            temperature=request.temperature,
            word_mask=word_mask,
            skip_before=reuse_len,
            stream_reduce=residual.stream_reduce,
            stream_index=residual.stream_index,
            jacobians=jacobians,
            softcap=softcap,
        ):
            if vocab_size <= 0:
                vocab_size = _readout_vocab_size(tokenizer, model)
            pos, token_id, is_generated = step.position, step.token_id, step.is_generated
            results = [
                _slice(lens_type, tokenizer, decode_cache, step.top_ids[i], step.top_probs[i])
                for i, lens_type in enumerate(requested_types)
            ]
            solo = _decode_token(tokenizer, int(token_id), decode_cache)
            entry = LensTokenMessage(
                position=pos,
                token="",
                id=int(token_id),
                is_generated=is_generated,
                results=results,
                **_span_fields(pos, int(token_id), is_generated, solo),
            )
            if _REPLACEMENT_CHAR not in solo:
                # A self-contained token: flush any stuck fragment run first, then
                # emit this token normally.
                for flushed in _flush_pending_as_is():
                    yield flushed
                yield _emit(entry, solo)
            else:
                # A fragment: buffer it and see if the run now decodes cleanly.
                pending.append(entry)
                combined = tokenizer.decode([p.id for p in pending], clean_up_tokenization_spaces=False)
                if _REPLACEMENT_CHAR not in combined:
                    for run_index, p in enumerate(pending):
                        yield _emit(p, combined, continuation=run_index > 0)
                    pending.clear()
                elif len(pending) >= _MAX_MULTI_TOKEN_CHAR:
                    for flushed in _flush_pending_as_is():
                        yield flushed

            if is_generated:
                completion_ids.append(int(token_id))
            position = pos + 1

    # Any trailing fragments that never completed: emit them best-effort.
    for flushed in _flush_pending_as_is():
        yield flushed

    completion = tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False) if completion_ids else ""
    yield LensDoneMessage(
        seq_len=position,
        prompt_len=prompt_len,
        vocab_size=vocab_size,
        completion=completion,
    )


# --------------------------------------------------------------------------- #
# Startup warmup
# --------------------------------------------------------------------------- #


def warmup_lens() -> None:
    """Run one tiny (1-token) pass through the real lens code path at startup.

    Moves any one-time initialization on the lens read-out path to startup so the
    first *real* JACOBIAN_LENS request is correct/fast.

    Only runs when a Jacobian lens is loaded (LOGIT_LENS is always correct), and
    is fully best-effort: any failure is logged and swallowed so startup is never
    affected.
    """
    lens = JacobianLensStore.get()
    if lens is None:
        return

    try:
        model = Model.get_instance()
    except Exception:  # noqa: BLE001
        return

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return

    config = Config.get_instance()
    np_model_id = getattr(config, "model_id", None)
    hf_model_id = getattr(config, "custom_hf_model_id", None) or getattr(config, "override_model_id", None)

    # Backend-independent one-time work, warmed before the EagerModel gate below because
    # the vLLM path pays for all of it inline on its first request otherwise -- on the event
    # loop, so it stalls every other request too. Two full-vocab Python decodes (~0.9s each
    # at Qwen3.6-27B's 248k ids): the read-out's word mask, and the reverse index that
    # steer/swap resolves its token strings through. The softcap reads the HF config, a 2.7s
    # round trip cold since VLLMModel exposes no `.config` to read it from.
    try:
        _word_token_mask(tokenizer, _readout_vocab_size(tokenizer, model))
    except Exception:  # noqa: BLE001
        logger.exception("Word-mask warmup failed (non-fatal)")
    try:
        _decoded_string_to_ids(tokenizer)
    except Exception:  # noqa: BLE001
        logger.exception("Steer-token index warmup failed (non-fatal)")
    try:
        resolve_final_logit_softcap(model, np_model_id=np_model_id, hf_model_id=hf_model_id)
    except Exception:  # noqa: BLE001
        logger.exception("Softcap warmup failed (non-fatal)")

    if not isinstance(model, EagerModel):
        # The rest reads out in this process, which only the eager backend does at startup.
        logger.info("Lens warmup completed (shared paths only on the %s backend).", backend_name(model))
        return

    n_layers = config.num_layers
    if n_layers is None:
        return

    try:
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None:
            token_ids = [int(bos)]
        else:
            encoded = tokenizer("The", add_special_tokens=False)["input_ids"]
            token_ids = [int(t) for t in encoded[:1]]
        if not token_ids:
            return

        # Warm both types so the entire path (including JACOBIAN_LENS, the one
        # that needs it) is exercised; sharing one forward pass makes this cheap.
        requested_types = [LensType.JACOBIAN_LENS, LensType.LOGIT_LENS]
        # Resolved here too, so a lens that cannot be read out on this model says so at startup
        # rather than on someone's first request. The failure is caught and logged below, which
        # is the right severity: LOGIT_LENS alone is unaffected and the pod should still serve.
        residual = resolve_residual_spec(lens.residual, model.residual_basis)
        lenses = [
            LensSpec(layers=_select_layers(t, n_layers, lens, layers=[]), jacobian=t == LensType.JACOBIAN_LENS)
            for t in requested_types
        ]

        async def _read() -> None:
            async for _ in model.generate_with_lens(
                token_ids,
                lenses,
                point=residual.point_name(model.residual_basis.n_streams),
                top_n=1,
                stream_reduce=residual.stream_reduce,
                stream_index=residual.stream_index,
                jacobians=lens.placed_jacobians(),
            ):
                pass

        # Startup runs in an executor thread, which has no event loop of its own.
        asyncio.run(_read())

        logger.info("Lens warmup completed (%d token(s)).", len(token_ids))
    except Exception:  # noqa: BLE001
        logger.exception("Lens warmup failed (non-fatal)")


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


async def _acquire_request_lock(fail_if_busy: bool = False):
    """Acquire a request slot; return the primitive to release later (or None).

    Acquired in the route handler (not via a decorator) so we can return a proper
    HTTP status BEFORE the streaming response body starts; the slot is held for the
    lifetime of the stream and released in the generator's ``finally`` (Starlette
    iterates a StreamingResponse body after the handler returns, so a decorator-scoped
    slot would be released before generation even runs). With the per-request demux the
    lens hooks are per-request-safe, so this takes a NON-exclusive slot (concurrent on
    vLLM; still one-at-a-time off vLLM via the single mutex).

    Returns the acquired primitive on success, or ``None`` only when ``fail_if_busy``
    is set and no slot is immediately available (caller responds 429).
    """
    if fail_if_busy and limiter.is_busy(exclusive=False):
        return None
    if limiter.is_busy(exclusive=False):
        logger.warning("[LIMITER] Lens request waiting for a slot (another request in progress)...")
    return await limiter.acquire(exclusive=False, timeout=REQUEST_LOCK_TIMEOUT)


@router.post(
    "/lens/prompt",
    responses={400: {"model": LensErrorResponse, "description": "The request was refused before the stream started."}},
)
async def lens_prompt(request: LensPromptRequest, http_request: Request):
    config = Config.get_instance()
    model = Model.get_instance()

    # ---- validation (before the stream starts, so we can return proper 4xx) ---
    # A lens read-out is a capture, so a GENERATION_ONLY pod cannot serve one. Checked here for the
    # same reason as everything else in this block: once the stream has started, the only place left
    # to report an error is inside a frame.
    try:
        assert_residual_available(model, "The logit lens", point=block_output_point(model))
    except BackendUnsupported as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    use_input_token_ids = len(request.input_token_ids) > 0
    # When exact token ids are supplied we read out over them verbatim (no
    # tokenization, no generation), so `prompt`/`chat` are not required.
    if not use_input_token_ids and (request.prompt is None) == (request.chat is None):
        return JSONResponse(
            content={"error": "Provide exactly one of 'prompt' or 'chat'"},
            status_code=400,
        )

    # Checked here rather than left to `build_token_ids` so it reads as the client error
    # it is, instead of a logged tokenization failure. A missing tokenizer is a different
    # (server-side) fault and is left to `build_token_ids` to report.
    if request.chat is not None and model.tokenizer is not None and not model.tok.has_chat_template():
        return JSONResponse(content={"error": NO_CHAT_TEMPLATE_ERROR}, status_code=400)
    # Dropping the tools would silently analyze a different prompt, so refuse instead.
    if (
        request.chat is not None
        and request.tools
        and model.tokenizer is not None
        and "tools" not in model.tok.accepted_template_kwargs(("tools",))
    ):
        return JSONResponse(content={"error": NO_TOOLS_TEMPLATE_ERROR}, status_code=400)

    # De-duplicate the requested types while preserving order.
    requested_types: list[LensType] = list(dict.fromkeys(request.type))
    if not requested_types:
        return JSONResponse(
            content={"error": "Provide at least one lens type in 'type'"},
            status_code=400,
        )

    if request.temperature < 0:
        return JSONResponse(content={"error": "temperature must be >= 0"}, status_code=400)
    if request.num_completion_tokens < 0:
        return JSONResponse(content={"error": "num_completion_tokens must be >= 0"}, status_code=400)

    lenses: dict[LensType, LoadedJacobianLens] = {}
    for lens_type in requested_types:
        kind = J_BAR_LENS_KINDS.get(lens_type)
        if kind is None:
            continue
        lens = kind.store.get()
        if lens is None:
            return JSONResponse(
                content={
                    "error": f"{kind.label[0].upper()}{kind.label[1:]} is not available for this model",
                    "status": kind.store.status(),
                    "detail": kind.store.error(),
                },
                status_code=400,
            )
        lenses[lens_type] = lens

    try:
        if use_input_token_ids:
            # Read out over the exact ids; never generate (reproduction only).
            token_ids = [int(token_id) for token_id in request.input_token_ids]
            request.num_completion_tokens = 0
        else:
            token_ids = build_token_ids(model, request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to tokenize lens request")
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    if len(token_ids) == 0:
        return JSONResponse(content={"error": "Prompt produced zero tokens"}, status_code=400)

    # The lens endpoints use their own limit (config.lens_token_limit), separate
    # from config.token_limit used by the other endpoints. Reads-outs are
    # computed per position, so cost grows with sequence length; this caps the
    # conversation/prompt length to keep requests responsive.
    if len(token_ids) > config.lens_token_limit:
        return JSONResponse(
            content={
                "error": (
                    f"This conversation is too long ({len(token_ids)} tokens). "
                    f"The maximum is {config.lens_token_limit} tokens — please "
                    f"shorten your input or start a new conversation."
                )
            },
            status_code=400,
        )

    max_seq_len = request.max_seq_len or config.lens_token_limit
    token_ids = token_ids[:max_seq_len]

    # Clamp generation length to the memory-safe sequence budget (prompt + generation).
    request.num_completion_tokens = config.clamp_completion_tokens(len(token_ids), request.num_completion_tokens)

    # Longest common token-id prefix with what the client already has. Positions
    # in this prefix have identical preceding context (causal attention), so the
    # client's cached read-outs are still valid and we skip recomputing them.
    # Bounded to the prompt length (generation always recomputes).
    reuse_len = _common_prefix_len(token_ids, request.cached_token_ids)

    n_layers = config.num_layers
    if n_layers is None:
        return JSONResponse(content={"error": "Model layer count not initialized"}, status_code=500)

    layers_by_type = {
        lens_type: _select_layers(lens_type, n_layers, lenses.get(lens_type), request.layers)
        for lens_type in requested_types
    }

    # Which activation to read out, from the loaded lens's own declaration. Resolved from the store
    # even for a LOGIT_LENS-only request, and deliberately: the two types are shown side by side, so
    # reading the logit lens in the space the Jacobian lens was fitted in is what makes them
    # comparable -- and at the lens's target layer, where J is the identity, what makes them agree.
    try:
        residual_spec = resolve_residual_spec(_declared_residual(lenses), model.residual_basis)
    except (LensSpaceUnknown, ResidualBasisUnsupported) as exc:
        # 400 rather than 500: the missing fact lives in the artifact, and the message says how to
        # put it there. Nothing about the server or the request is wrong.
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    # ---- steering / swap: resolve readouts -> per-layer injection directions ----
    # SWAP replaces the source readout (steer_tokens[0]) with `swap_token`; it
    # needs the source directions too, so it reuses the steer-delta builder.
    swap_active = request.swap_token is not None and len(request.steer_tokens) > 0
    steer_active = len(request.steer_tokens) > 0 and (request.steer_strength != 0.0 or request.steer_ablate)
    steer_deltas: dict[int, torch.Tensor] = {}
    swap_deltas: dict[int, torch.Tensor] = {}
    if steer_active or swap_active:
        # An intervention writes into the forward, which a graph-replay pod cannot do without a
        # static write site -- and would not report, since a hook that never fires returns fluent,
        # unsteered text. Asked here rather than at registration so the answer is a 400 before the
        # stream opens, not an exception several frames into an RPC.
        try:
            assert_steering_available(model, "jlens steering, ablation and swap")
        except BackendUnsupported as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)
        # The client's explicit layer list is used verbatim: an empty list means
        # no steering/swap (e.g. the user deselected every layer).
        try:
            steer_deltas = await _build_steer_deltas(model, lenses, request.steer_tokens, request.steer_layers)
            if swap_active and request.swap_token is not None:
                swap_deltas = await _build_steer_deltas(model, lenses, [request.swap_token], request.steer_layers)
        except SteerTokenNotFound as exc:
            body = LensErrorResponse(error=str(exc), token=exc.token, suggested_token=exc.suggestion)
            return JSONResponse(content=body.model_dump(), status_code=400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to build steering/swap vectors")
            return JSONResponse(content={"error": str(exc)}, status_code=400)
        # The client's cached read-outs come from an unsteered run; they are no
        # longer valid once we steer/swap, so recompute every position.
        if steer_deltas or swap_deltas:
            reuse_len = 0

    softcap = resolve_final_logit_softcap(
        model,
        np_model_id=getattr(config, "model_id", None),
        hf_model_id=getattr(config, "custom_hf_model_id", None) or getattr(config, "override_model_id", None),
    )

    # ---- acquire the model lock up-front ----
    # Acquired here (not inside the streaming generator) so we can return a
    # proper HTTP status BEFORE the response body starts: 429 when the server is
    # busy and the client asked to fail fast (`fail_if_busy`, so it can try a
    # different server), or 503 on a lock-wait timeout. The lock is held for the
    # whole stream and released in the generator's `finally` once generation
    # completes (or the client disconnects).
    try:
        acquired = await _acquire_request_lock(fail_if_busy=request.fail_if_busy)
    except TimeoutError:
        logger.error("[LIMITER] Timeout waiting for a slot on lens request")
        return JSONResponse(
            content={"error": "Request timed out waiting for lock"},
            status_code=503,
        )
    if acquired is None:
        # Server is busy with another request and the client opted to fail fast
        # so it can fall back to another inference server for this model.
        return JSONResponse(
            content={"error": "Server is busy with another request", "busy": True},
            status_code=429,
        )

    # ---- reserve VRAM alongside the slot ----
    # Released in the generator's `finally`, together with the slot, so it covers the whole
    # stream. Note the slot must be released by hand on every path out of here, since the
    # generator that would normally do it never runs.
    #
    # TWO TERMS, and the second is the one that decides how many of these fit at once.
    #
    # The staged rows are device memory (see `lens_cost`), and only the positions this request
    # will actually read out count: a follow-up turn reusing a long cached prefix stages the
    # new tail, not the whole conversation. vLLM stages inside the worker, one read-out chunk
    # at a time rather than a whole transport batch, so its rows are a rounding error -- the
    # same fact about where its weights live that `_build_messages` branches on.
    #
    # The CAPTURE is not chunked, and it is charged over the whole sequence. Reuse buys
    # nothing here and `reuse_len` is deliberately not subtracted: `skip_before` drops cached
    # positions from the read-out, but the forward still runs over them and the hooks still
    # fire, so the harvest covers the conversation rather than its tail. That is why the
    # failure showed up on the SECOND turn of a chat -- the first fit, and the reservation
    # could not see the difference between them.
    staging_batch = lens_stream.READOUT_CHUNK if isinstance(model, VLLMModel) else lens_stream.STAGE_BATCH
    staged_positions = min(
        staging_batch,
        max(1, len(token_ids) - reuse_len + request.num_completion_tokens),
    )
    lens_bytes = lens_cost(
        staged_positions=staged_positions,
        layer_counts=[len(layers) for layers in layers_by_type.values()],
        d_model=int(getattr(model, "d_model", 0)) or max((lens.d_model for lens in lenses.values()), default=0),
        capture_positions=len(token_ids) + max(0, request.num_completion_tokens),
        # One capture site per DISTINCT layer: the types share a forward, so a layer both
        # lenses read is captured once.
        n_capture_points=len({layer for layers in layers_by_type.values() for layer in layers}),
        n_streams=int(getattr(getattr(model, "residual_basis", None), "n_streams", 1) or 1),
    )
    #
    # `fail_if_busy` covers this wait too, and has to. The slot check above is instant, so
    # without it a fail-fast request could still sit here for the full budget timeout -- and
    # a client that asked to fail fast did so in order to try a different pod, which it can
    # only do while it is still connected.
    try:
        budget_claim = await budget.acquire(lens_bytes, fail_if_busy=request.fail_if_busy)
    except RequestTooLarge as exc:
        acquired.release()
        logger.error("[BUDGET] lens request rejected: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except RequestBusy:
        acquired.release()
        return JSONResponse(
            content={"error": "Server has no free memory for another request", "busy": True},
            status_code=429,
        )
    except TimeoutError:
        acquired.release()
        logger.error("[BUDGET] Timeout waiting for VRAM on lens request")
        return JSONResponse(
            content={"error": "Request timed out waiting for available memory"},
            status_code=503,
        )

    # ---- streaming body: holds the model lock for its whole lifetime ----
    async def _ndjson_stream() -> AsyncIterator[str]:
        try:
            async for message in _build_messages(
                model,
                request,
                requested_types,
                lenses,
                softcap,
                layers_by_type,
                token_ids,
                reuse_len=reuse_len,
                steer_deltas=steer_deltas,
                steer_strength=request.steer_strength,
                steer_ablate=request.steer_ablate,
                swap_deltas=swap_deltas,
                steer_generated=request.steer_generated_tokens,
                residual=residual_spec,
            ):
                # Stop generating as soon as the client (or the proxy in front of
                # it) goes away — e.g. the user pressed "Stop", or it timed out and
                # moved to another pod. The `finally` below then releases the slot and
                # the VRAM so the next request isn't blocked behind a dead one.
                #
                # This only answers truthfully while every middleware is pure ASGI; see
                # the middleware section of server.py before adding one.
                if await http_request.is_disconnected():
                    logger.info("[LENS] Client disconnected; aborting generation.")
                    break
                yield json.dumps(message.model_dump(mode="json")) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error computing lens slice")
            # Reclaim cached blocks after a failure (e.g. CUDA OOM) so the next
            # request starts from a clean allocator state. Only on the error
            # path: empty_cache() forces re-allocation from the driver and would
            # add latency if called on every (successful) request.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield json.dumps(LensErrorMessage(error=str(exc)).model_dump(mode="json")) + "\n"
        finally:
            await budget.release(budget_claim)
            acquired.release()

    if request.stream:
        return StreamingResponse(_ndjson_stream(), media_type="application/x-ndjson")

    # Non-streaming: run the identical path, buffer messages into one object.
    meta: dict | None = None
    tokens: list[dict] = []
    done: dict | None = None
    error: dict | None = None
    async for line in _ndjson_stream():
        message = json.loads(line)
        kind = message.get("kind")
        if kind == "meta":
            meta = message
        elif kind == "token":
            tokens.append(message)
        elif kind == "done":
            done = message
        elif kind == "error":
            error = message

    if error is not None:
        return JSONResponse(content=error, status_code=500)
    return JSONResponse(content={"meta": meta, "tokens": tokens, "done": done})
