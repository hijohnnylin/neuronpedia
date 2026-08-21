import json
import logging
import unicodedata
from collections.abc import AsyncIterator, Callable, Iterator
from typing import NamedTuple, cast

import numpy as np
import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from interp_engine import (
    EagerModel,
    GeneratedTurnSpans,
    HookManager,
    NoChatTemplateError,
    ResidualBasisUnsupported,
    VLLMModel,
)
from interp_engine import decode_residuals as engine_decode_residuals
from pydantic import BaseModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.lens.lens_loader import (
    JacobianLensStore,
    LoadedJacobianLens,
)
from neuronpedia_inference.endpoints.lens.model_specific import (
    apply_final_logit_softcap,
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
    get_tokenize,
)
from neuronpedia_inference.memory_cost import lens_cost
from neuronpedia_inference.schemas import (
    LensPromptRequest,
    LensSteerToken,
    LensType,
    PublicFrameSchema,
)
from neuronpedia_inference.shared import (
    REQUEST_LOCK_TIMEOUT,
    STR_TO_DTYPE,
    Model,
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


def _resolve_eos_token_ids(model, tokenizer) -> set[int]:
    """Collect every token id that should stop generation for this model.

    A single ``tokenizer.eos_token_id`` is not enough for every family: gpt-oss
    (harmony) ends its assistant turn with ``<|return|>`` (or ``<|call|>`` for a
    tool call), while its plain ``eos_token`` (``<|endoftext|>``) is never
    emitted mid-conversation. The model's ``generation_config.eos_token_id``
    lists all of these (e.g. ``[<|endoftext|>, <|return|>, <|call|>]``), so we
    union it with the tokenizer's eos. Without this, gpt-oss generation runs
    past the assistant turn (emitting ``<|return|>`` then continuing) until the
    completion-token cap. NOTE: harmony's ``<|end|>`` (which closes the
    *analysis* channel before the *final* channel) is intentionally NOT a stop
    token — stopping there would truncate the response before its final answer.
    """
    ids: set[int] = set()

    def _add(value) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, list | tuple | set):
            for item in value:
                _add(item)

    _add(getattr(tokenizer, "eos_token_id", None))
    # EagerModel wraps the raw HF model as ``.hf_model`` (which carries the
    # ``generation_config``); best-effort lookup.
    hf = getattr(model, "hf_model", model)
    gen_cfg = getattr(hf, "generation_config", None)
    _add(getattr(gen_cfg, "eos_token_id", None))
    return ids


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
    return kwargs


def _chat_args(tok, request: LensPromptRequest) -> tuple[list[dict[str, str]], bool, bool, dict]:
    """Return ``(messages, add_generation_prompt, continue_final_message, template_kwargs)``.

    If the final message is an assistant turn, treat it as a PREFILL: keep that turn open (no
    end-of-turn token, no fresh assistant scaffold) so generation continues from the prefilled
    text rather than starting a new assistant turn after it.
    """
    messages = [{"role": m.role, "content": m.content} for m in (request.chat or [])]
    is_prefill = len(messages) > 0 and messages[-1]["role"] == "assistant"
    return (
        messages,
        (not is_prefill),
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
        tok = get_tokenize(model)
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
        tok = get_tokenize(model)
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


def _sample_token(logits_row: torch.Tensor, temperature: float) -> int:
    """Pick the next token id from a ``[vocab]`` logit row.

    ``temperature == 0`` is greedy (argmax); otherwise temperature sampling.

    Non-finite logits (``nan``/``inf``, e.g. when aggressive steering blows the
    residual up) are rejected before sampling: ``torch.multinomial`` on a
    probability tensor containing ``nan``/``inf`` triggers a device-side assert
    that poisons the process's CUDA context, so we raise a clean error instead.
    """
    if not torch.isfinite(logits_row).all():
        raise ValueError(
            "Non-finite logits during generation (nan/inf) — likely caused by "
            "steering that is too strong. Reduce the steer strength or the "
            "number of steered layers."
        )
    if temperature <= 0:
        return int(logits_row.argmax())
    probs = torch.softmax(logits_row.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


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


def _resolve_steer_token_id(index: dict[str, list[int]], token: str) -> int:
    """Resolve an exact (or, failing that, whitespace-trimmed) decoded string to
    a single vocab id. True collisions (multiple ids -> same string) are rare;
    we take the lowest id (their unembedding directions are near-identical)."""
    ids = index.get(token)
    if not ids:
        stripped = token.strip()
        for decoded, candidate_ids in index.items():
            if decoded.strip() == stripped:
                ids = candidate_ids
                break
    if not ids:
        raise ValueError(f"Could not resolve steer token to a vocab id: {token!r}")
    return int(min(ids))


def _check_token_id_in_range(token_id: int, vocab_size: int) -> None:
    """Raise a clear error for a token id outside ``[0, vocab_size)``.

    Guards against indexing the (un)embedding matrix out of bounds, which on a
    CUDA device raises a device-side assert that corrupts the process's CUDA
    context (all subsequent CUDA calls then fail until restart).
    """
    if not (0 <= token_id < vocab_size):
        raise ValueError(f"steer token_id {token_id} out of range for unembedding vocab size {vocab_size}")


def _unembed_vector(model, token_id: int) -> torch.Tensor:
    """Residual-space unembedding direction for ``token_id`` (float32).

    ``token_id`` is bounds-checked against the actual unembedding matrix on the
    host before indexing: an out-of-range id would otherwise trigger a
    device-side assert that poisons the entire CUDA context for the process.
    """
    weight = model.arch.lm_head.weight  # lm_head: [vocab, d_model]
    vocab_size = weight.shape[0]
    _check_token_id_in_range(token_id, vocab_size)
    return weight[token_id].detach().float()


async def _unembed_vectors_by_id(model, token_ids: list[int]) -> dict[int, torch.Tensor]:
    """Backend-aware ``{token_id: unembedding_row [d_model] float32}`` for jlens steering.

    EagerModel indexes ``arch.lm_head.weight`` directly; the vLLM backend fetches the rows
    via ``unembed_rows`` (``lm_head.weight``, or tied ``embed_tokens.weight`` on Gemma 2).
    """
    unique = list(dict.fromkeys(int(t) for t in token_ids))
    if isinstance(model, VLLMModel):
        rows = (await model.unembed_rows(unique)).float()
        return {tid: rows[i] for i, tid in enumerate(unique)}
    return {tid: _unembed_vector(model, tid) for tid in unique}


async def _build_steer_deltas(
    model,
    lens: LoadedJacobianLens | None,
    steer_tokens: list[LensSteerToken],
    steer_layers: list[int],
) -> dict[int, torch.Tensor]:
    """Build the per-layer unit direction to inject, summed across steer tokens.

    For a ``JACOBIAN_LENS`` token at a fitted layer ``l`` the direction is
    ``J_bar_l^T @ w_t`` (equivalently ``w_t @ J_bar_l``), the residual-space
    direction whose J-lens readout is the token; otherwise the plain unembedding
    direction ``w_t``.     Each per-layer direction is unit-normalized before
    summing so multiple tokens reinforce sensibly. The unembedding rows come from
    the eager model (``arch.lm_head``) or, on vLLM, the worker unembed weight
    (``lm_head`` or tied ``embed_tokens``).

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
    compute_device = (lens.transport_device if lens is not None else None) or out_device

    # [n_tokens, d_model] in `resolved` order, so one round trip covers every layer when the
    # lens is in the worker. `transported` says which layers had a fitted J_bar; the rest come
    # back untouched, which is what a non-Jacobian token wants anyway.
    transported_by_layer: torch.Tensor | None = None
    fitted: list[bool] = []
    if lens is not None and lens.worker_resident and any(t == LensType.JACOBIAN_LENS for _, t in resolved):
        stacked = torch.stack([w_by_id[tid] for tid, _ in resolved], dim=0)
        transported_by_layer, fitted = await model.lens_transport(stacked, steer_layers)

    deltas: dict[int, torch.Tensor] = {}
    for layer_index, layer in enumerate(steer_layers):
        acc: torch.Tensor | None = None
        for token_index, (token_id, lens_type) in enumerate(resolved):
            w = w_by_id[token_id]
            if lens_type != LensType.JACOBIAN_LENS or lens is None:
                direction = w
            elif transported_by_layer is not None:
                direction = transported_by_layer[layer_index][token_index] if fitted[layer_index] else w
            elif layer in lens.jacobians:
                j_bar = lens.jacobian_on(layer, compute_device)
                direction = (w.to(device=compute_device, dtype=j_bar.dtype) @ j_bar).float()  # J_bar^T @ w
            else:
                direction = w
            norm = torch.linalg.vector_norm(direction)
            if norm > 0:
                direction = direction / norm
            acc = direction if acc is None else acc + direction.to(acc.device)
        if acc is not None:
            deltas[layer] = acc.to(out_device)
    return deltas


def _bos_skip_mask(
    token_ids: list[int], bos_token_id: int | None, device, *, stacked: bool = False
) -> torch.Tensor | None:
    """Bool mask ``[1, seq, 1]`` marking BOS positions to leave unmodified.

    Returns ``None`` when there is no BOS id or the sequence contains none, so
    the caller skips masking entirely. Only the EXACT bos id is matched, so chat
    special tokens (turn markers, etc.) are still steered.

    ``stacked`` adds the trailing axis a hyper-connection trunk needs, giving ``[1, seq, 1, 1]``
    against a ``[batch, seq, streams, d_model]`` residual. The rank has to match the tensor rather
    than merely broadcast against it: torch aligns from the right, so the flat mask would line its
    sequence axis up with the STREAM axis and mask four positions' worth of nothing.
    """
    if bos_token_id is None:
        return None
    flags = [tid == bos_token_id for tid in token_ids]
    if not any(flags):
        return None
    shape = (1, -1, 1, 1) if stacked else (1, -1, 1)
    return torch.tensor(flags, dtype=torch.bool, device=device).view(*shape)


# Per-layer cap on the additive steering vector, as a fraction of the
# per-position residual norm. Steering is applied at every selected layer, so
# the effect compounds; capping each step keeps a strong/multi-layer request
# from driving the residual (and hence the logits) to inf/nan.
_MAX_STEER_INJECTION_FRACTION = 1.0
_STEER_NORM_EPS = 1e-12


def _apply_steer(
    tensor: torch.Tensor,
    delta: torch.Tensor,
    strength: float,
    ablate: bool = False,
    skip_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Steer each position's residual ``h`` along ``delta``.

    When ``ablate`` is true, project the (unit) readout direction OUT of the
    residual (``h <- h - (h.d_hat) d_hat``), fully removing that component
    regardless of ``strength``. Otherwise add ``strength * ||h|| * unit_delta``;
    scaling by the per-position residual norm keeps a given ``strength`` behaving
    consistently across layers/models (it's a fraction of the residual norm). The
    injected vector's norm is additionally capped to
    ``_MAX_STEER_INJECTION_FRACTION * ||h||`` so a large strength (or steering at
    many layers, which compounds) can't drive the residual to inf/nan.

    ``skip_mask`` (bool, broadcastable against ``tensor``) marks positions to
    leave UNCHANGED (e.g. the BOS token, whose huge attention-sink norm makes the
    intervention spuriously large there).
    """
    d = delta.to(device=tensor.device, dtype=tensor.dtype)
    if ablate:
        norm = torch.linalg.vector_norm(d)
        if norm == 0:
            return tensor
        d_hat = d / norm
        proj = (tensor * d_hat).sum(dim=-1, keepdim=True)
        steered = tensor - proj * d_hat
    else:
        # The injected vector is ``(strength * ||h||) * d``. Because steering is
        # applied at every selected layer on that layer's output, the effect
        # compounds across layers and can blow the residual up to inf/nan.
        # Cap the injected vector's norm to a fraction of the per-position
        # residual norm so a large strength (or many steered layers) can't push
        # the residual arbitrarily far in one step.
        scale = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
        injected = (strength * scale) * d
        injected_norm = torch.linalg.vector_norm(injected, dim=-1, keepdim=True)
        max_norm = _MAX_STEER_INJECTION_FRACTION * scale
        clamp_factor = torch.where(
            injected_norm > max_norm,
            max_norm / injected_norm.clamp_min(_STEER_NORM_EPS),
            torch.ones_like(injected_norm),
        )
        steered = tensor + injected * clamp_factor
    if skip_mask is not None:
        steered = torch.where(skip_mask, tensor, steered)
    return steered


def _apply_swap(
    tensor: torch.Tensor,
    src_delta: torch.Tensor,
    tgt_delta: torch.Tensor,
    skip_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Swap the source readout direction for the target at each position ``h``.

    Removes the residual's projection onto the (unit) source direction and adds
    back an equal-magnitude projection along the (unit) target direction:
    ``h <- h - (h.s_hat) s_hat + (h.s_hat) t_hat``. This is the causal
    "lens-vector swap" intervention (subtract the source readout, add the
    target with the same coefficient) and is parameter-free (the magnitude is
    the residual's own source projection).

    ``skip_mask`` (bool, broadcastable against ``tensor``) marks positions to
    leave UNCHANGED (e.g. the BOS token).
    """
    s = src_delta.to(device=tensor.device, dtype=tensor.dtype)
    t = tgt_delta.to(device=tensor.device, dtype=tensor.dtype)
    s_norm = torch.linalg.vector_norm(s)
    t_norm = torch.linalg.vector_norm(t)
    if s_norm == 0 or t_norm == 0:
        return tensor
    s_hat = s / s_norm
    t_hat = t / t_norm
    coeff = (tensor * s_hat).sum(dim=-1, keepdim=True)
    swapped = tensor - coeff * s_hat + coeff * t_hat
    if skip_mask is not None:
        swapped = torch.where(skip_mask, tensor, swapped)
    return swapped


async def _iter_residuals(
    model,
    prompt_token_ids: list[int],
    union_layers: list[int],
    *,
    num_completion_tokens: int,
    temperature: float,
    eos_token_ids: set[int] | None,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    bos_token_id: int | None = None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
):
    """Async-stream :data:`ResidualBatch` groups of ``(token_id, is_generated, residuals)``.

    Prompt positions come first (from the prefill), then the generated tokens.
    ``residuals`` maps each requested layer to its ``[d_model]`` residual -- one vector per layer
    even on a hyper-connection trunk, where ``residual`` says which function of the stream stack the
    lens was fitted on and both arms below collapse it before yielding. Each yielded batch is
    everything that became available at once -- the whole prefill, then one group per decode step --
    so the consumer can stage a batched read-out without ever holding a position back.

    EagerModel runs the incremental KV-cached loop eagerly (per-token streaming, with
    optional steer/swap write-hooks). The vLLM backend consumes the engine's
    decode-time capture as it is produced, so it streams per token too; lens
    INTERVENTION (steer/swap/ablate) is applied there by worker write-hooks.
    """
    if isinstance(model, EagerModel):
        for step in _iter_residuals_engine(
            model,
            prompt_token_ids,
            union_layers,
            num_completion_tokens=num_completion_tokens,
            temperature=temperature,
            eos_token_ids=eos_token_ids,
            steer_deltas=steer_deltas,
            steer_strength=steer_strength,
            steer_ablate=steer_ablate,
            swap_deltas=swap_deltas,
            steer_generated=steer_generated,
            bos_token_id=bos_token_id,
            residual=residual,
        ):
            yield step
        return
    if isinstance(model, VLLMModel):
        async for step in _iter_residuals_vllm(
            model,
            prompt_token_ids,
            union_layers,
            num_completion_tokens=num_completion_tokens,
            temperature=temperature,
            steer_deltas=steer_deltas,
            steer_strength=steer_strength,
            steer_ablate=steer_ablate,
            swap_deltas=swap_deltas,
            steer_generated=steer_generated,
            bos_token_id=bos_token_id,
            residual=residual,
        ):
            yield step
        return
    raise ValueError(
        f"Lens endpoint does not support model type {type(model).__name__} (only the interp-engine and vLLM backends)."
    )


def _build_vllm_lens_specs(
    steer_deltas: dict[int, torch.Tensor] | None,
    steer_strength: float,
    steer_ablate: bool,
    swap_deltas: dict[int, torch.Tensor] | None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
    n_streams: int = 1,
) -> list[dict]:
    """Build the vLLM worker lens-intervention specs (steer/ablate/swap) per layer.

    Mirrors the eager precedence in ``_iter_residuals_engine``: swap wins over
    additive/ablation. Empty when there is no active intervention.

    Every spec names the point the read-out is taken at, which is what lets an intervention reach a
    hyper-connection trunk: ``resid_post`` is the worker's default and does not exist there, so a
    spec that said nothing had nothing to aim at. On a conventional trunk ``point_name`` returns
    exactly that default, so the two trunks share one code path rather than branching here.
    """
    swapping = bool(swap_deltas) and bool(steer_deltas)
    steering = bool(steer_deltas) and (steer_strength != 0.0 or steer_ablate)
    site: dict = {"point": residual.point_name(n_streams)}
    if residual.write_stream is not None:
        site["stream"] = int(residual.write_stream)
    specs: list[dict] = []
    if swapping and steer_deltas is not None and swap_deltas is not None:
        for layer, tgt in swap_deltas.items():
            src = steer_deltas.get(layer)
            if src is not None:
                specs.append(
                    {
                        **site,
                        "layer": int(layer),
                        "op": "swap",
                        "delta": src.tolist(),
                        "tgt": tgt.tolist(),
                    }
                )
    elif steering and steer_deltas is not None:
        for layer, delta in steer_deltas.items():
            if steer_ablate:
                specs.append({**site, "layer": int(layer), "op": "ablate", "delta": delta.tolist()})
            else:
                specs.append(
                    {
                        **site,
                        "layer": int(layer),
                        "op": "steer",
                        "delta": delta.tolist(),
                        "strength": steer_strength,
                    }
                )
    return specs


def _lens_max_tokens(num_completion_tokens: int) -> int:
    """How many tokens to sample so that every requested position has a read-out.

    A position's read-out comes from the forward pass that has that token as its *input*, and
    nothing runs after the last token sampled. So sampling exactly ``n`` leaves the nth token
    with no row and no read-out, and a request for 3 generated tokens displayed 2. Sampling one
    extra runs that forward; the extra token itself sits past ``limit`` and is dropped by the
    emit cap, on the same path that already discards an engine overrun.

    ``num_completion_tokens == 0`` keeps its single throwaway token rather than growing to two:
    an intervention-only request generates one to run the intervened forward, and that position
    is not a result either.
    """
    return num_completion_tokens + 1 if num_completion_tokens > 0 else 1


async def _iter_readout_vllm(
    model: VLLMModel,
    prompt_token_ids: list[int],
    requested_types: list[LensType],
    layers_by_type: dict[LensType, list[int]],
    *,
    num_completion_tokens: int,
    temperature: float,
    top_n: int,
    softcap: float | None,
    word_mask: torch.Tensor | None,
    chunk_positions: int,
    skip_before: int,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    bos_token_id: int | None = None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
):
    """vLLM lens stream, read out in the worker: yields batches of per-position top-k.

    The residual-free counterpart of :func:`_iter_residuals_vllm`. Residuals are captured,
    Jacobian-transported and unembedded in the worker process, so what comes back per
    position is ``(top_idx, top_probs)`` per requested type rather than ``d_model`` floats
    per layer. Driving it the other way -- capture out to this process, transport here, ship
    the staged rows back for the unembed -- moved the same ~63 MB across ``collective_rpc``
    twice for a 96-position 64-layer read-out, and that transport, not the GPU, was what the
    endpoint's latency and its concurrency ceiling were made of.

    Yields the same ``(token_id, is_generated, payload)`` triples the residual iterators do,
    so the emit path downstream is shared; only the payload differs.

    ``residual`` is where the lens was fitted, which decides both the point captured and -- on a
    hyper-connection trunk, where the capture is a stream stack -- how the worker collapses it before
    the transport. It travels as part of the read-out spec rather than being applied here, because on
    this path the rows never leave the worker.
    """
    specs = [
        {"layers": layers_by_type[lens_type], "jacobian": lens_type == LensType.JACOBIAN_LENS}
        for lens_type in requested_types
    ]
    union_layers = sorted({layer for lens_type in requested_types for layer in layers_by_type[lens_type]})
    n_streams = model.residual_basis.n_streams
    points = [residual.address(layer, n_streams) for layer in union_layers]
    prompt_len = len(prompt_token_ids)
    intervention_specs = _build_vllm_lens_specs(
        steer_deltas, steer_strength, steer_ablate, swap_deltas, residual, n_streams
    )
    lens_intervention = None
    if intervention_specs:
        skip = [i for i, tid in enumerate(prompt_token_ids) if bos_token_id is not None and tid == bos_token_id]
        lens_intervention = {
            "specs": intervention_specs,
            "steer_generated": steer_generated,
            "skip_positions": skip,
            "prompt_len": prompt_len,
        }
    # As in `_iter_residuals_vllm`: an intervention-only request still generates one
    # throwaway token to run the intervened forward, and that position is not a result.
    limit = prompt_len if num_completion_tokens <= 0 else prompt_len + num_completion_tokens

    slices: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    emitted = max(0, int(skip_before))
    async for first_position, idx_list, probs_list, gen_ids in model.lens_capture_readout_stream(
        prompt_token_ids,
        points,
        specs,
        top_n=top_n,
        softcap=softcap,
        word_mask=word_mask,
        chunk_positions=chunk_positions,
        skip_before=skip_before,
        max_tokens=_lens_max_tokens(num_completion_tokens),
        temperature=temperature,
        lens_intervention=lens_intervention,
        stream_reduce=residual.stream_reduce,
        stream_index=residual.stream_index,
    ):
        # A yield may carry only the newly sampled ids (see `lens_capture_readout_stream`);
        # those still matter here, because an id is half of what a position needs.
        n_positions = idx_list[0].shape[0] // max(1, len(layers_by_type[requested_types[0]])) if idx_list else 0
        for offset in range(n_positions):
            per_type: list[tuple[torch.Tensor, torch.Tensor]] = []
            for spec_index, lens_type in enumerate(requested_types):
                n_layers = len(layers_by_type[lens_type])
                k = idx_list[spec_index].shape[-1]
                top_idx = idx_list[spec_index].view(n_positions, n_layers, k)[offset]
                top_probs = probs_list[spec_index].view(n_positions, n_layers, k)[offset]
                per_type.append((top_idx, top_probs))
            slices[first_position + offset] = per_type

        # A position needs BOTH its read-out and its token id; generation runs ahead of the
        # read-outs, so emit only where the two have met (and never past the request's cap).
        batch: list[tuple[int, bool, list[tuple[torch.Tensor, torch.Tensor]]]] = []
        while emitted < limit and emitted in slices and emitted < prompt_len + len(gen_ids):
            token_id = prompt_token_ids[emitted] if emitted < prompt_len else gen_ids[emitted - prompt_len]
            batch.append((int(token_id), emitted >= prompt_len, slices.pop(emitted)))
            emitted += 1
        if batch:
            yield batch


async def _iter_residuals_vllm(
    model: VLLMModel,
    prompt_token_ids: list[int],
    union_layers: list[int],
    *,
    num_completion_tokens: int,
    temperature: float,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    bos_token_id: int | None = None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
):
    """vLLM residual stream: engine decode-time capture of resid_post at every position,
    with optional jlens steer/ablate/swap intervention applied during generation.

    ``num_completion_tokens == 0`` with no intervention captures the prefill only; otherwise
    it generates and captures prompt + generated positions (the final sampled token is never
    processed, so it has no residual -- universal autoregressive behavior). Intervention is
    applied via decode-time worker write-hooks (norm-scaled+clamped steer / ablate / swap),
    matching the eager path; BOS positions in the prefill are skipped.

    Generation is consumed INCREMENTALLY (``capture_generation_stream``): every position is
    yielded as soon as its residual row and its token id are both known, so the read-out
    streams token-by-token like the eager path instead of waiting for the whole completion.
    Positions from one drain are yielded as a single :data:`ResidualBatch`; the engine
    outruns the read-out, so those batches naturally grow and the read-out stays batched
    without holding anything back.

    ``residual`` names the point to capture and, on a hyper-connection trunk, the reduction that
    turns each ``[n, n_streams, d_model]`` block into the ``[n, d_model]`` rows the consumer stages.
    Applied here rather than in the worker because this is the arm that ships residuals out; the
    fused read-out reduces there instead, from the same declaration.
    """
    n_streams = model.residual_basis.n_streams
    points = [residual.address(layer, n_streams) for layer in union_layers]
    addresses = dict(zip(union_layers, points, strict=True))
    prompt_len = len(prompt_token_ids)
    specs = _build_vllm_lens_specs(steer_deltas, steer_strength, steer_ablate, swap_deltas, residual, n_streams)
    lens_intervention = None
    if specs:
        skip = [i for i, tid in enumerate(prompt_token_ids) if bos_token_id is not None and tid == bos_token_id]
        lens_intervention = {
            "specs": specs,
            "steer_generated": steer_generated,
            "skip_positions": skip,
            "prompt_len": prompt_len,
        }

    if num_completion_tokens <= 0 and lens_intervention is None:
        caps = await model.capture(prompt_token_ids, points)
        reduced = {layer: residual.reduce(caps[addresses[layer]], n_streams) for layer in union_layers}
        yield [
            (
                int(prompt_token_ids[pos]),
                False,
                {layer: reduced[layer][pos] for layer in union_layers},
            )
            for pos in range(prompt_len)
        ]
        return

    # Captured rows in forward order: the prefill's prompt_len rows, then one per decode
    # step. Kept as per-row views so a position can be handed out the moment it lands.
    rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in union_layers}
    gen_ids: list[int] = []
    emitted = 0
    # An intervention-only request (num_completion_tokens == 0) still has to generate one
    # throwaway token to run the intervened forward; its position is not a result.
    limit = prompt_len if num_completion_tokens <= 0 else prompt_len + num_completion_tokens

    async for new_caps, token_ids in model.capture_generation_stream(
        prompt_token_ids,
        points,
        max_tokens=_lens_max_tokens(num_completion_tokens),
        temperature=temperature,
        lens_intervention=lens_intervention,
    ):
        for layer in union_layers:
            block = new_caps.get(addresses[layer])
            if block is not None:
                # Reduced per block rather than per row: on a stream stack this is an
                # `n_streams`-way mean, and one call over the whole drain beats one per position.
                rows[layer].extend(residual.reduce(block, n_streams).unbind(0))
        gen_ids = token_ids
        # A position needs BOTH its residual row and its token id; generation runs ahead
        # of the drains, so take the smaller of the two (and never past the request's cap).
        captured = min(len(rows[layer]) for layer in union_layers)
        available = min(captured, prompt_len + len(gen_ids), limit)
        batch: ResidualBatch = []
        while emitted < available:
            token_id = prompt_token_ids[emitted] if emitted < prompt_len else gen_ids[emitted - prompt_len]
            batch.append(
                (
                    int(token_id),
                    emitted >= prompt_len,
                    {layer: rows[layer][emitted] for layer in union_layers},
                )
            )
            emitted += 1
        if batch:
            yield batch


def _iter_residuals_engine(
    model: EagerModel,
    prompt_token_ids: list[int],
    union_layers: list[int],
    *,
    num_completion_tokens: int,
    temperature: float,
    eos_token_ids: set[int] | None,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    bos_token_id: int | None = None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
) -> Iterator[ResidualBatch]:
    """Engine (raw HF) residual stream: KV-cached prefill + decode, capturing resid_post
    per layer at every position, with optional additive-steer / swap interventions applied
    via forward write-hooks (before the capture read-hooks, so read-outs reflect steering).

    Yields the prefill's positions as one :data:`ResidualBatch`, then one per decode step.

    This path hooks the decoder layer's own output rather than resolving an address, because it
    interleaves capture with incremental decoding and re-installs its hooks per step. That tensor is
    the block output on either kind of trunk, so ``residual`` supplies only the stream reduction on
    the read and the stream to write on the intervention -- and any other capture point is refused
    rather than silently read as this one.
    """
    if residual.capture_point != "block_output":
        raise ValueError(
            "The eager read-out captures each decoder layer's own output, so it cannot serve a lens "
            f"fitted on {residual.capture_point!r}. Serve this lens on the vLLM backend, which "
            "resolves the point through the engine's address table."
        )
    device = model.device
    captures: dict[int, torch.Tensor] = {}
    layer_module = {layer: model.arch.decoder_layers[layer] for layer in union_layers}

    basis = model.residual_basis
    n_streams = basis.n_streams
    # What a write hook is handed here is the block's own output, which on a hyper-connection trunk
    # is the whole `[batch, seq, streams, d_model]` stack rather than one residual. A lens fitted on
    # one stream writes that stream; one fitted on the mean or the sum writes every stream at once.
    # See `LensResidualSpec.write_stream` for why the second is the intervention it describes and
    # not an approximation of it.
    write_stream = residual.write_stream if n_streams > 1 else None
    stacked_write = n_streams > 1 and write_stream is None

    skip_holder: dict = {"mask": _bos_skip_mask(prompt_token_ids, bos_token_id, device, stacked=stacked_write)}

    swapping = bool(swap_deltas) and bool(steer_deltas)
    steering = bool(steer_deltas) and (steer_strength != 0.0 or steer_ablate)

    def make_capture(layer: int):
        def _cap(tensor: torch.Tensor) -> None:
            captures[layer] = tensor.detach()

        return _cap

    def make_steer(delta: torch.Tensor):
        def _steer(tensor: torch.Tensor) -> torch.Tensor:
            return _apply_steer(
                tensor,
                delta,
                steer_strength,
                steer_ablate,
                skip_mask=skip_holder["mask"],
            )

        return _steer

    def make_swap(src: torch.Tensor, tgt: torch.Tensor):
        def _swap(tensor: torch.Tensor) -> torch.Tensor:
            return _apply_swap(tensor, src, tgt, skip_mask=skip_holder["mask"])

        return _swap

    def scoped(write: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        """Confine ``write`` to one stream of the stack, when the lens named one.

        Out of place via ``replace_stream``, because the hook runs on a tensor the model still
        holds. The selection is checked against the stream count rather than indexed directly:
        indexing the wrong axis succeeds too, and returns a tensor of an entirely believable shape.
        """
        if write_stream is None:
            return write
        stream = write_stream

        def _scoped(tensor: torch.Tensor) -> torch.Tensor:
            return basis.replace_stream(tensor, stream, write(basis.select_stream(tensor, stream)))

        return _scoped

    def install(hm: HookManager, apply_intervention: bool) -> None:
        # Write hooks first so the capture read hooks observe the modified residual.
        if apply_intervention and swapping and steer_deltas is not None and swap_deltas is not None:
            for layer, tgt in swap_deltas.items():
                src = steer_deltas.get(layer)
                if src is not None:
                    hm.write(
                        model.arch.decoder_layers[layer],
                        scoped(make_swap(src, tgt)),
                        point="output",
                    )
        elif apply_intervention and steering and steer_deltas is not None:
            for layer, delta in steer_deltas.items():
                hm.write(model.arch.decoder_layers[layer], scoped(make_steer(delta)), point="output")
        for layer in union_layers:
            hm.read(layer_module[layer], make_capture(layer), point="output")

    def rows_for(layer: int) -> torch.Tensor:
        """The layer's ``[seq, d_model]`` block for this forward, stream axis already collapsed."""
        return residual.reduce(captures[layer][0], n_streams)

    tokens = torch.tensor([prompt_token_ids], device=device)
    with HookManager() as hm:
        install(hm, apply_intervention=True)
        with torch.no_grad():
            out = model.hf_model(tokens, use_cache=True)
    past = out.past_key_values
    prefill = {layer: rows_for(layer) for layer in union_layers}
    yield [
        (
            int(token_id),
            False,
            {layer: prefill[layer][pos] for layer in union_layers},
        )
        for pos, token_id in enumerate(prompt_token_ids)
    ]

    last_logits = out.logits[0, -1, :]
    generated = 0
    while generated < num_completion_tokens:
        next_id = _sample_token(last_logits, temperature)
        generated += 1
        captures.clear()
        # Skip intervention on a generated BOS (its attention-sink residual norm is huge).
        skip_holder["mask"] = (
            torch.ones((1, 1, 1, 1) if stacked_write else (1, 1, 1), dtype=torch.bool, device=device)
            if (bos_token_id is not None and next_id == bos_token_id)
            else None
        )
        cur = torch.tensor([[next_id]], device=device)
        with HookManager() as hm:
            install(hm, apply_intervention=steer_generated)
            with torch.no_grad():
                out = model.hf_model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        yield [
            (
                int(next_id),
                True,
                {layer: rows_for(layer)[-1] for layer in union_layers},
            )
        ]
        last_logits = out.logits[0, -1, :]
        if eos_token_ids is not None and next_id in eos_token_ids:
            break


async def _decode_residuals(model, residuals_2d: torch.Tensor, softcap: float | None = None) -> torch.Tensor:
    """Decode ``[n_rows, d_model]`` residuals to ``[n_rows, vocab]`` logits using
    the model's own final norm + unembedding (no Jacobian; caller applies that).

    EagerModel decodes eagerly (real final_norm + lm_head); the vLLM backend decodes
    via the uniform worker ``compute_logits`` (reuses vLLM's own norm + lm_head).

    The softcap is applied here rather than by the caller because the two backends
    do not agree on whether it has already happened: vLLM applies the model's
    configured cap inside ``compute_logits``, the eager ``lm_head`` does not. The
    returned logits are softcapped exactly once either way.

    ``logit_multiplier`` rides along for the same reason and from the same asymmetry: on Cohere,
    Granite, Falcon-H1 and LLaDA the real forward scales its logits after ``lm_head``, and vLLM's
    ``LogitsProcessor`` does too, while the eager ``lm_head`` does not. Read off the model rather than
    passed in, because unlike the softcap there is no served-model override for it.
    """
    if isinstance(model, VLLMModel):
        return await model.decode_residuals(residuals_2d)
    with torch.no_grad():
        return engine_decode_residuals(model, residuals_2d, softcap=softcap, multiplier=model.logit_multiplier)


# --------------------------------------------------------------------------- #
# Per-engine layer logits (single-forward read-out; parity test + warmup)
# --------------------------------------------------------------------------- #

# Per-type layer logits: {lens_type: {layer: logits[seq_len, vocab]}}.
LayerLogitsByType = dict[LensType, dict[int, torch.Tensor]]


def _compute_logits_for_types(
    model,
    token_ids: list[int],
    layers_by_type: dict[LensType, list[int]],
    lens: LoadedJacobianLens | None,
    softcap: float | None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
) -> LayerLogitsByType:
    """Read out per-layer logits for every requested lens type in ONE forward pass.

    The residual stream is captured once and reused: a LOGIT_LENS row decodes the
    residual directly, a JACOBIAN_LENS row first transports it with ``J_bar``. So
    requesting both types only costs the extra per-layer projections, not a second
    model forward pass. The final layer (not fitted) is always decoded directly,
    giving the model's true output.
    """
    if isinstance(model, EagerModel):
        return _compute_logits_for_types_engine(model, token_ids, layers_by_type, lens, softcap, residual)
    raise ValueError(
        f"Lens endpoint does not support model type {type(model).__name__} (only the interp-engine backend)."
    )


def _union_layers(layers_by_type: dict[LensType, list[int]]) -> list[int]:
    return sorted({layer for layers in layers_by_type.values() for layer in layers})


def _common_prefix_len(token_ids: list[int], cached_token_ids: list[int]) -> int:
    """Length of the longest common leading run of two token-id lists."""
    n = 0
    for a, b in zip(token_ids, cached_token_ids):
        if a != b:
            break
        n += 1
    return n


def _compute_logits_for_types_engine(
    model: EagerModel,
    token_ids: list[int],
    layers_by_type: dict[LensType, list[int]],
    lens: LoadedJacobianLens | None,
    softcap: float | None,
    residual: LensResidualSpec = BLOCK_OUTPUT,
) -> LayerLogitsByType:
    """Engine read-out: capture the lens's point once, decode each layer via real norm+lm_head.

    JACOBIAN_LENS rows first transport the residual with the fitted lens (only for
    layers the lens was fit on).

    Unlike :func:`_iter_residuals_engine`, this goes through the engine's address table, so it can
    serve any capture point ``residual`` names -- and collapses the stream stack where the trunk
    carries one.
    """
    from interp_engine import run_with_cache

    device = model.device
    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)
    union = _union_layers(layers_by_type)
    n_streams = model.residual_basis.n_streams
    point = residual.point_name(n_streams)
    cache = run_with_cache(model, tokens, [(point, layer) for layer in union])
    residuals = {layer: residual.reduce(cache.get(point, layer)[0], n_streams) for layer in union}

    out: LayerLogitsByType = {}
    for lens_type, layers in layers_by_type.items():
        use_jacobian = lens_type == LensType.JACOBIAN_LENS and lens is not None
        layer_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            rows = residuals[layer]
            if use_jacobian and lens is not None and layer in lens.jacobians:
                rows = lens.transport(rows.float(), layer)
            logits = engine_decode_residuals(model, rows, multiplier=model.logit_multiplier)
            logits = apply_final_logit_softcap(logits, softcap)
            layer_logits[layer] = logits.detach()
        out[lens_type] = layer_logits
    return out


# --------------------------------------------------------------------------- #
# Slice assembly (ported from the jlens demo vis)
# --------------------------------------------------------------------------- #


class _TypeReadoutState:
    """Stateful, per-position read-out for one lens type.

    Each ``process(...)`` call takes the ``[n_layers, vocab]`` logits at a single
    position and returns the ``LensTypeSlice`` for that position.
    """

    def __init__(
        self,
        lens_type: LensType,
        tokenizer,
        vocab_size: int,
        *,
        top_n: int,
        decode_cache: dict[int, str],
        filter_non_word: bool = False,
    ) -> None:
        self.lens_type = lens_type
        self.tokenizer = tokenizer
        self.top_n = top_n
        self.decode_cache = decode_cache
        self.vocab_size = vocab_size
        self.filter_non_word = filter_non_word
        # Word-mask, lazily moved to the logits' device on first use (None until
        # then, or when filtering is disabled).
        self._filter_mask: torch.Tensor | None = None

    def _word_mask_on(self, logits: torch.Tensor) -> torch.Tensor:
        if self._filter_mask is None or self._filter_mask.device != logits.device:
            mask = _word_token_mask(self.tokenizer, int(logits.shape[-1]))
            self._filter_mask = mask.to(logits.device)
        return self._filter_mask

    def process(self, logits: torch.Tensor) -> LensTypeSlice:
        """logits: ``[n_layers, vocab]`` for ONE position."""
        # float32 for logsumexp / topk stability, matching the vLLM worker's
        # `worker_lens_readout`. A bf16 read-out is not merely rounded: `log_z`
        # sums ~256k terms and `top_logits - log_z` cancels two large nearly
        # equal values, both in a format with 8 mantissa bits. The result is a
        # systematic overestimate that reports probabilities of exactly 1.0 and
        # top-k rows summing above 1.
        logits = logits.float()
        # `log_z` is computed over the FULL (unmasked) vocab, so the reported
        # probabilities stay the model's real probabilities; non-word filtering
        # only changes WHICH tokens are selected into the top-n.
        log_z = logits.logsumexp(dim=-1, keepdim=True)
        if self.filter_non_word:
            mask = self._word_mask_on(logits)
            # Preserve ONLY the FINAL (output) layer's true top-1 — the model's
            # actual next-token prediction — even when it is a non-word token.
            # Intermediate-layer top-1s are NOT preserved: at those layers the
            # argmax is frequently a lens artifact (e.g. ``<|endoftext|>`` /
            # special tokens dominating the early-decoding basis mid-sequence),
            # not a meaningful read-out, so we let the non-word filter drop them.
            # The final layer is always the LAST row (``_select_layers`` sorts
            # ascending and the final layer is the max), so index -1 is it.
            final_top1 = int(logits[-1].argmax())
            final_logit = float(logits[-1, final_top1])
            logits.masked_fill_(~mask, torch.finfo(logits.dtype).min)
            logits[-1, final_top1] = final_logit
        top_idx = logits.topk(self.top_n, dim=-1).indices  # [n_layers, top_n]
        top_logits = logits.gather(-1, top_idx)
        top_probs = (top_logits - log_z).exp()

        top_idx_np = top_idx.cpu().numpy()
        # Round in float64 (not float32): rounding a float32 leaves the value at
        # the nearest float32 bit pattern, which then widens to a noisy float64
        # on `.tolist()` (e.g. 0.0591 -> 0.059112560003995895). Rounding a
        # float64 lands on a clean decimal whose shortest round-trip repr is
        # short, so the serialized payload actually shrinks. The tensor is tiny
        # ([n_layers, top_n]) so the double cast is negligible.
        top_probs_np = top_probs.double().cpu().numpy()

        top_tokens = [
            [_decode_token(self.tokenizer, int(token_id), self.decode_cache) for token_id in row] for row in top_idx_np
        ]

        return LensTypeSlice(
            type=self.lens_type,
            top_tokens=top_tokens,
            # Round to 4 decimals (0.01% resolution) to cut serialized payload
            # size. The client only renders integer percentages and normalized
            # per-layer heatmap weights, so extra precision is never visible.
            top_probs=np.round(top_probs_np, 4).tolist(),
        )

    def from_topk(self, top_idx: torch.Tensor, top_probs: torch.Tensor) -> LensTypeSlice:
        """Build a slice from worker-side top-k ids/probs (``[n_layers, top_n]``)."""
        top_idx_np = top_idx.detach().cpu().numpy()
        top_probs_np = top_probs.detach().double().cpu().numpy()
        top_tokens = [
            [_decode_token(self.tokenizer, int(token_id), self.decode_cache) for token_id in row] for row in top_idx_np
        ]
        return LensTypeSlice(
            type=self.lens_type,
            top_tokens=top_tokens,
            top_probs=np.round(top_probs_np, 4).tolist(),
        )


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
    if lens_type == LensType.JACOBIAN_LENS and lens is not None:
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


# What a position carries from the iterator to the emit path. Raw residuals on the eager
# backend, where the read-out happens here; on vLLM the read-out itself -- one
# ``(top_idx, top_probs)`` per requested type -- because the worker did it already.
PositionPayload = dict[int, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]

# How many positions to decode per batched read-out matmul. The per-layer
# unembedding matmul against the (large) vocab re-streams the ``lm_head`` weight
# from HBM, so it is memory-bound when decoding one position at a time. Batching
# ``chunk_size * n_layers`` rows into a single matmul amortizes that weight read
# across positions, crossing into compute-bound territory (the win saturates once
# ``chunk_size * n_layers`` exceeds the GPU's FLOP:byte ridge, ~150 on A100 / ~300
# on H100). It is kept small because the intermediate it bounds is vocab-sized:
# ``chunk_size * n_layers * vocab``, which is 213 MB per chunk at gemma-2-2b's
# 26 layers x 256k vocab. Both backends use it: eager decodes a chunk per matmul
# here, and vLLM passes it to the worker as ``chunk_positions``.
_READOUT_CHUNK_SIZE = 8

# How many positions to STAGE (Jacobian-transport) per read-out batch, ahead of
# splitting into read-out chunks. Sized independently of the chunk above because
# the two are bound by different things: the transport's cost is re-reading each
# layer's ``J_bar`` (10.6 MB at d_model=2304 in bf16, so ~265 MB for a 25-layer
# sweep), and its intermediate is only ``batch * n_layers * d_model`` -- 31 MB
# here, three orders of magnitude below the vocab-sized one. Staging at the
# read-out chunk's granularity instead re-read every J_bar once per 8 positions,
# which cost 0.7s of a 1.3s 398-token gemma-2-2b request (measured when J_bar was
# held fp32, so twice the bytes below); staging 128 at a time makes it 0.05s.
#
# The re-read is from HBM now that the lens is device-resident, so the same 16x
# reduction in sweeps is worth ~72ms rather than ~650ms on a 128-position batch. Still
# worth batching, but it no longer gates the first read-out chunk on anything a user
# would notice, which is what makes staging this far ahead of emission acceptable.
#
# The eager backend's, since it is the one that stages here. On vLLM this only sets how
# many already-decoded positions the emit path assembles messages for at a time.
_TRANSPORT_BATCH_SIZE = 128


def _stack_chunk_residuals(
    lens_type: LensType,
    layers: list[int],
    residuals_list: list[dict[int, torch.Tensor]],
    lens: LoadedJacobianLens | None,
) -> torch.Tensor:
    """Stack per-position per-layer residuals to ``[n_positions * n_layers, d_model]``.

    For JACOBIAN_LENS, fitted layers are first transported with ``J_bar``; the
    final (unfitted) layer is left as-is (``J = I``), giving the model's true output.

    The transport is batched per LAYER (one ``[n_positions, d_model] @ [d_model,
    d_model]`` matmul) rather than per position. ``J_bar`` is ``d_model**2``
    values -- 10.6 MB at ``d_model=2304`` in bf16, far past any cache -- so a
    per-position matvec re-streams the whole matrix for every position and the loop
    is bound by memory bandwidth, not arithmetic.

    Being bandwidth-bound is also why ``LoadedJacobianLens.transport`` casts the
    residual down to the lens dtype instead of widening ``J_bar``: the residual block
    is ``n_positions * d_model`` and the matrix is ``d_model**2``.

    Everything is staged on the lens's ``transport_device`` -- including the layers that
    are NOT transported, which have to share a device to stack with the ones that are.

    The EAGER backend's staging. vLLM does the equivalent inside the worker, in
    ``worker_lens_capture_readout``, because that is where its residuals are; a lens whose
    matrices went to the worker leaves ``transport_device`` unset and never reaches here.
    """
    use_jacobian = lens_type == LensType.JACOBIAN_LENS and lens is not None
    stage_device = lens.transport_device if use_jacobian and lens is not None else None
    # [n_layers][n_positions, d_model], each block sharing one J_bar.
    blocks: list[torch.Tensor] = []
    for layer in layers:
        # Uniform float32 so transported and directly-decoded layers stack
        # together; the decode path recasts to the param dtype for the matmul.
        block = torch.stack([residuals[layer] for residuals in residuals_list], dim=0).float()
        if stage_device is not None and block.device != stage_device:
            block = block.to(stage_device)
        if use_jacobian and lens is not None and layer in lens.jacobians:
            block = lens.transport(block, layer)
        blocks.append(block)
    # [n_positions, n_layers, d_model] -> rows ordered position-major, which is
    # the layout `_chunk_position_logits` / `_chunk_position_slices_vllm` unfold.
    return torch.stack(blocks, dim=1).reshape(-1, blocks[0].shape[-1])


async def _chunk_position_logits(
    model,
    staged_rows: torch.Tensor,
    n_layers: int,
    softcap: float | None,
) -> list[torch.Tensor]:
    """Decode a chunk of already-staged rows in ONE batched matmul.

    ``staged_rows`` is ``[n_positions * n_layers, d_model]`` in position-major
    order (as produced by :func:`_stack_chunk_residuals`, already transported for
    JACOBIAN_LENS). Returns one ``[n_layers, vocab]`` logit tensor per position.
    All rows are decoded together so the unembedding weight is read from HBM once
    for the whole chunk instead of once per position. The chunk is kept in the
    model dtype: the unembedding matmul is already bf16, so widening every row at
    once only doubles the vocab-sized tensor's bandwidth. The softmax does need
    float32 to be correct, so ``_TypeReadoutState.process`` upcasts the one
    position it is reading out rather than the whole chunk.

    Used by the EagerModel path. The vLLM path uses
    :func:`_chunk_position_slices_vllm` instead so vocab-sized logits never cross
    ``collective_rpc``.
    """
    n_positions = staged_rows.shape[0] // n_layers
    logits = await _decode_residuals(model, staged_rows, softcap)  # [n_rows, vocab]
    logits = logits.view(n_positions, n_layers, logits.shape[-1])
    return [logits[pos] for pos in range(n_positions)]


def _to_wire_dtype(staged_rows: torch.Tensor) -> torch.Tensor:
    """Cast staged rows to the dtype the worker will use, where that dtype is known.

    ``model_dtype`` is ``"auto"`` by default and only resolves to something concrete inside
    vLLM, so an unrecognised name keeps the float32 rather than guessing: casting down to a
    dtype the model does not use would lose precision for real, where casting to the one it
    does use loses none.
    """
    wire = STR_TO_DTYPE.get(str(Config.get_instance().model_dtype))
    return staged_rows if wire is None else staged_rows.to(wire)


async def _chunk_position_slices_vllm(
    model: VLLMModel,
    staged_rows: torch.Tensor,
    n_layers: int,
    softcap: float | None,
    *,
    state: _TypeReadoutState,
    word_mask: torch.Tensor | None,
) -> list[LensTypeSlice]:
    """vLLM lens readout with the residuals shipped from here: one RPC per type x chunk.

    The fallback path, used only when the lens could not be made worker-resident (see
    :func:`place_jacobian_lens_on_worker`), because then this process is the only one that
    can apply ``J_bar``. ``staged_rows`` is ``[n_positions * n_layers, d_model]``,
    position-major and already transported. Only the norm + unembed + top-k run on the
    worker, so the RPC returns ``[n_rows, top_n]`` instead of ``[n_rows, vocab]``.

    Sent at the model dtype rather than the float32 the staging produces. ``worker_lens_readout``
    opens with ``.to(param.device, param.dtype)``, so the extra precision is discarded on
    arrival -- shipping it just doubles a payload that is 10 MB per call at 64 layers x 8
    positions x d_model 5120. That payload is why this is the fallback and not the norm: it
    made the RPC the read-out's serial resource, holding measured throughput near 0.5 req/s
    with the GPU at ~18%.
    """
    n_positions = staged_rows.shape[0] // n_layers
    top_idx, top_probs = await model.decode_residuals_topk(
        _to_wire_dtype(staged_rows),
        top_n=state.top_n,
        softcap=softcap,
        word_mask=word_mask,
        rows_per_group=n_layers,
    )
    # [n_positions * n_layers, k] -> per position [n_layers, k] (k may be < top_n
    # only if vocab is tiny; real models always have vocab >> top_n).
    k = int(top_idx.shape[-1])
    top_idx = top_idx.view(n_positions, n_layers, k)
    top_probs = top_probs.view(n_positions, n_layers, k)
    return [state.from_topk(top_idx[pos], top_probs[pos]) for pos in range(n_positions)]


async def _build_messages(
    model,
    request: LensPromptRequest,
    requested_types: list[LensType],
    lens: LoadedJacobianLens | None,
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
    """Yield the ordered stream of messages: meta -> token* -> done.

    A plain (synchronous) generator; the route wraps it to manage the model lock
    and NDJSON serialization. Residuals are produced incrementally (prefill, then
    one KV-cached decode step per generated token) and each position's lens slice
    is emitted as soon as it is computed — token-by-token streaming, with the
    read-out batched over whatever positions the backend already has ready.

    ``reuse_len`` is the number of leading prompt positions the client already
    has read-outs for (the token-id common prefix). The model is still prefilled
    over the FULL prompt (later positions' residuals depend on the earlier ones),
    but the per-layer read-out and the token message are skipped for those
    positions — the bulk of the cost — so a follow-up turn only recomputes the
    new tokens.
    """
    tokenizer = model.tokenizer
    decode_cache: dict[int, str] = {}
    prompt_len = len(prompt_token_ids)
    union_layers = _union_layers(layers_by_type)
    eos_token_ids = _resolve_eos_token_ids(model, tokenizer)
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

    yield LensMetaMessage(
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

    states: dict[LensType, _TypeReadoutState] = {}
    position = 0
    vocab_size = 0
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

    # Word-mask for the vLLM top-k path (built once; the eager path builds it
    # lazily inside `_TypeReadoutState` from logits.shape[-1]). Must be sized to
    # the model's logits vocab (padded embedding), not tokenizer.vocab_size alone.
    vllm_word_mask: torch.Tensor | None = None
    if isinstance(model, VLLMModel) and request.filter_non_word_tokens:
        try:
            vs = _readout_vocab_size(tokenizer, model)
            vllm_word_mask = _word_token_mask(tokenizer, vs)
            vocab_size = vs
        except Exception:  # noqa: BLE001
            logger.exception("Failed to build word-token mask for vLLM lens readout")
            vllm_word_mask = None

    async def _emit_chunk(
        buf: list[tuple[int, int, bool, PositionPayload]],
    ) -> "AsyncIterator[LensTokenMessage]":
        """Emit token messages for a batch of positions, in order (with multi-byte-char repair).
        An empty batch is a no-op.

        On vLLM the read-out already happened, in the worker, so each position arrives holding
        its per-type top-k and there is nothing here but message assembly.

        On the eager path the work is here. The Jacobian transport is staged ONCE per lens type
        for the WHOLE batch, so each layer's ``J_bar`` is read once no matter how many positions
        arrived, and the vocab-sized read-out then walks the batch in ``_READOUT_CHUNK_SIZE``
        chunks. Each chunk's messages are emitted as soon as that chunk is decoded --
        chunk-major, not type-major. Ordering the loops the other way would hold the first
        message until the entire batch (up to ``_TRANSPORT_BATCH_SIZE`` positions x every type)
        had been decoded, which cost 0.18s of time-to-first-read-out on a 398-token gemma-2-2b
        prompt.
        """
        nonlocal vocab_size
        if not buf:
            return

        # Read by the payload rather than the backend: which iterator ran is the thing that
        # decides this, and the two payload shapes are what tell them apart.
        precomputed = isinstance(buf[0][3], list)
        for lens_type in requested_types:
            if lens_type in states:
                continue
            if vocab_size <= 0:
                vocab_size = _readout_vocab_size(tokenizer, model)
            states[lens_type] = _TypeReadoutState(
                lens_type,
                tokenizer,
                vocab_size,
                top_n=request.top_n,
                decode_cache=decode_cache,
                filter_non_word=request.filter_non_word_tokens,
            )

        staged_by_type: dict[LensType, torch.Tensor] = {}
        if not precomputed:
            residuals_list = [cast("dict[int, torch.Tensor]", payload) for (_, _, _, payload) in buf]
            staged_by_type = {
                lens_type: _stack_chunk_residuals(lens_type, layers_by_type[lens_type], residuals_list, lens)
                for lens_type in requested_types
            }

        step = len(buf) if precomputed else _READOUT_CHUNK_SIZE
        for start in range(0, len(buf), step):
            end = min(start + step, len(buf))
            chunk_results: list[list[LensTypeSlice]] = [[] for _ in range(end - start)]
            if precomputed:
                for i in range(end - start):
                    payload = cast("list[tuple[torch.Tensor, torch.Tensor]]", buf[start + i][3])
                    for spec_index, lens_type in enumerate(requested_types):
                        top_idx, top_probs = payload[spec_index]
                        chunk_results[i].append(states[lens_type].from_topk(top_idx, top_probs))
            else:
                for lens_type in requested_types:
                    state = states[lens_type]
                    n_layers = len(layers_by_type[lens_type])
                    rows = staged_by_type[lens_type][start * n_layers : end * n_layers]
                    if isinstance(model, VLLMModel):
                        slices = await _chunk_position_slices_vllm(
                            model,
                            rows,
                            n_layers,
                            softcap,
                            state=state,
                            word_mask=vllm_word_mask if request.filter_non_word_tokens else None,
                        )
                        for i, sl in enumerate(slices):
                            chunk_results[i].append(sl)
                        continue
                    logits_list = await _chunk_position_logits(model, rows, n_layers, softcap)
                    if vocab_size <= 0:
                        vocab_size = int(logits_list[0].shape[-1])
                        state.vocab_size = vocab_size
                    for i, logits in enumerate(logits_list):
                        chunk_results[i].append(state.process(logits))

            for i, (pos, token_id, is_generated, _residuals) in enumerate(buf[start:end]):
                solo = _decode_token(tokenizer, int(token_id), decode_cache)
                entry = LensTokenMessage(
                    position=pos,
                    token="",
                    id=int(token_id),
                    is_generated=is_generated,
                    results=chunk_results[i],
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

    # Positions are staged in batches of up to ``_TRANSPORT_BATCH_SIZE`` and read out
    # a batch at a time. A batch is flushed at the end of every group the backend hands
    # over (the prefill, then each decode-time drain), so nothing is ever held waiting
    # on a later forward pass -- generated tokens still stream as they are produced,
    # and only positions that were ALREADY available get batched together.
    chunk_buf: list[tuple[int, int, bool, PositionPayload]] = []
    # The worker can only read out what it can transport, so a lens that did not fit there
    # keeps the older route: residuals out to this process, ``J_bar`` applied here, staged rows
    # back for the unembed. Nothing to transport (logit lens alone) needs no lens at all.
    worker_reads_out = isinstance(model, VLLMModel) and (
        LensType.JACOBIAN_LENS not in requested_types or (lens is not None and lens.worker_resident)
    )
    if worker_reads_out:
        assert isinstance(model, VLLMModel)
        # The worker reads out as it captures, so positions arrive already decoded -- and the
        # ones below `reuse_len` are dropped there, before the unembed, rather than being
        # shipped here to be discarded. The counter therefore starts where the stream does.
        position = reuse_len
        batches = _iter_readout_vllm(
            model,
            prompt_token_ids,
            requested_types,
            layers_by_type,
            num_completion_tokens=request.num_completion_tokens,
            temperature=request.temperature,
            top_n=request.top_n,
            softcap=softcap,
            word_mask=vllm_word_mask if request.filter_non_word_tokens else None,
            chunk_positions=_READOUT_CHUNK_SIZE,
            skip_before=reuse_len,
            steer_deltas=steer_deltas,
            steer_strength=steer_strength,
            steer_ablate=steer_ablate,
            swap_deltas=swap_deltas,
            steer_generated=steer_generated,
            bos_token_id=bos_token_id,
            residual=residual,
        )
    else:
        batches = _iter_residuals(
            model,
            prompt_token_ids,
            union_layers,
            num_completion_tokens=request.num_completion_tokens,
            temperature=request.temperature,
            eos_token_ids=eos_token_ids,
            steer_deltas=steer_deltas,
            steer_strength=steer_strength,
            steer_ablate=steer_ablate,
            swap_deltas=swap_deltas,
            steer_generated=steer_generated,
            bos_token_id=bos_token_id,
            residual=residual,
        )

    async for batch in batches:
        for token_id, is_generated, payload in batch:
            # Skip the read-out + emission for positions the client already has
            # (matched token-id prefix). Generated positions are always past the
            # prompt, so they are never skipped.
            if position < reuse_len:
                if is_generated:
                    completion_ids.append(int(token_id))
                position += 1
                continue

            chunk_buf.append((position, int(token_id), is_generated, payload))
            position += 1
            if len(chunk_buf) >= _TRANSPORT_BATCH_SIZE:
                async for msg in _emit_chunk(chunk_buf):
                    yield msg
                chunk_buf = []

        async for msg in _emit_chunk(chunk_buf):
            yield msg
        chunk_buf = []

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
        # The rest needs `run_with_cache`, which only the eager backend has.
        logger.info("Lens warmup completed (shared paths only; %s has no in-process forward).", type(model).__name__)
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
        layers_by_type = {
            lens_type: _select_layers(lens_type, n_layers, lens, layers=[]) for lens_type in requested_types
        }
        softcap = resolve_final_logit_softcap(model, np_model_id=np_model_id, hf_model_id=hf_model_id)

        _compute_logits_for_types(
            model,
            token_ids,
            layers_by_type,
            lens,
            softcap,
            # Resolved here too, so a lens that cannot be read out on this model says so at startup
            # rather than on someone's first request. The failure is caught and logged below, which
            # is the right severity: LOGIT_LENS alone is unaffected and the pod should still serve.
            resolve_residual_spec(lens.residual, model.residual_basis),
        )

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


@router.post("/lens/prompt")
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
    if request.chat is not None and model.tokenizer is not None and not get_tokenize(model).has_chat_template():
        return JSONResponse(content={"error": NO_CHAT_TEMPLATE_ERROR}, status_code=400)

    # De-duplicate the requested types while preserving order.
    requested_types: list[LensType] = list(dict.fromkeys(request.type))
    if not requested_types:
        return JSONResponse(
            content={"error": "Provide at least one lens type in 'type'"},
            status_code=400,
        )

    if not isinstance(model, EagerModel | VLLMModel):
        return JSONResponse(
            content={"error": ("The lens endpoint is only supported on the interp-engine and vLLM backends.")},
            status_code=400,
        )

    if request.temperature < 0:
        return JSONResponse(content={"error": "temperature must be >= 0"}, status_code=400)
    if request.num_completion_tokens < 0:
        return JSONResponse(content={"error": "num_completion_tokens must be >= 0"}, status_code=400)

    lens: LoadedJacobianLens | None = None
    if LensType.JACOBIAN_LENS in requested_types:
        lens = JacobianLensStore.get()
        if lens is None:
            return JSONResponse(
                content={
                    "error": "Jacobian lens is not available for this model",
                    "status": JacobianLensStore.status(),
                    "detail": JacobianLensStore.error(),
                },
                status_code=400,
            )

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
        lens_type: _select_layers(lens_type, n_layers, lens, request.layers) for lens_type in requested_types
    }

    # Which activation to read out, from the loaded lens's own declaration. Resolved from the store
    # even for a LOGIT_LENS-only request, and deliberately: the two types are shown side by side, so
    # reading the logit lens in the space the Jacobian lens was fitted in is what makes them
    # comparable -- and at the lens's target layer, where J is the identity, what makes them agree.
    declared = lens or JacobianLensStore.get()
    try:
        residual_spec = resolve_residual_spec(declared.residual if declared is not None else None, model.residual_basis)
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
            steer_deltas = await _build_steer_deltas(model, lens, request.steer_tokens, request.steer_layers)
            if swap_active and request.swap_token is not None:
                swap_deltas = await _build_steer_deltas(model, lens, [request.swap_token], request.steer_layers)
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
    # at a time rather than a whole transport batch, so its rows are a rounding error.
    #
    # The CAPTURE is not chunked, and it is charged over the whole sequence. Reuse buys
    # nothing here and `reuse_len` is deliberately not subtracted: `skip_before` drops cached
    # positions from the read-out, but the forward still runs over them and the hooks still
    # fire, so the harvest covers the conversation rather than its tail. That is why the
    # failure showed up on the SECOND turn of a chat -- the first fit, and the reservation
    # could not see the difference between them.
    staging_batch = _READOUT_CHUNK_SIZE if isinstance(model, VLLMModel) else _TRANSPORT_BATCH_SIZE
    staged_positions = min(
        staging_batch,
        max(1, len(token_ids) - reuse_len + request.num_completion_tokens),
    )
    lens_bytes = lens_cost(
        staged_positions=staged_positions,
        layer_counts=[len(layers) for layers in layers_by_type.values()],
        d_model=int(getattr(model, "d_model", 0)) or (lens.d_model if lens is not None else 0),
        capture_positions=len(token_ids) + max(0, request.num_completion_tokens),
        # One capture site per DISTINCT layer: the types share a forward, so a layer both
        # lenses read is captured once.
        n_capture_points=len({layer for layers in layers_by_type.values() for layer in layers}),
        n_streams=int(getattr(getattr(model, "residual_basis", None), "n_streams", 1) or 1),
    )
    try:
        budget_claim = await budget.acquire(lens_bytes)
    except RequestTooLarge as exc:
        acquired.release()
        logger.error("[BUDGET] lens request rejected: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=400)
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
                lens,
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
                # it) goes away — e.g. the user pressed "Stop". Checked once per
                # token; the `finally` below then releases the model lock so the
                # next request isn't blocked behind an abandoned generation.
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
