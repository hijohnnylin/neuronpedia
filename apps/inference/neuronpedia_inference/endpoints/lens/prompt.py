import asyncio
import json
import logging
import unicodedata
from collections.abc import Iterator
from enum import Enum

import nnsight
import numpy as np
import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from nnterp import StandardizedTransformer
from pydantic import BaseModel
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.lens.lens_loader import (
    JacobianLensStore,
    LoadedJacobianLens,
)
from neuronpedia_inference.endpoints.lens.model_specific import (
    apply_final_logit_softcap,
    resolve_final_logit_softcap,
)
from neuronpedia_inference.inference_utils.steering import apply_generic_chat_template
from neuronpedia_inference.shared import (
    REQUEST_LOCK_TIMEOUT,
    STR_TO_DTYPE,
    Model,
    request_lock,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class LensType(str, Enum):
    LOGIT_LENS = "LOGIT_LENS"
    JACOBIAN_LENS = "JACOBIAN_LENS"


class LensChatMessage(BaseModel):
    role: str
    content: str


class LensSteerToken(BaseModel):
    """A single readout to steer on.

    ``token`` is the EXACT decoded token string (whitespace preserved, e.g.
    ``" cat"``) as it appeared in a read-out slice; the server resolves it back
    to a vocab id via a cached reverse-decode map. ``type`` selects which lens's
    readout direction to use: ``JACOBIAN_LENS`` uses the J-lens direction
    ``J_bar_l^T @ w_t`` at each fitted layer (the residual-space direction whose
    J-lens readout is this token), ``LOGIT_LENS`` uses the plain unembedding
    direction ``w_t``.
    """

    token: str
    type: LensType


class LensPromptRequest(BaseModel):
    model: str
    # One or more lens types to compute. When both are requested, the model is
    # run only once (the residuals are shared), so adding LOGIT_LENS alongside
    # JACOBIAN_LENS is essentially free.
    type: list[LensType]
    # Provide exactly one of `prompt` (raw text) or `chat` (chat-formatted).
    prompt: str | None = None
    chat: list[LensChatMessage] | None = None
    top_n: int = 10
    # Layers to read out. Empty (default) = all available layers for the lens
    # type. The model's final layer is ALWAYS included (decoded directly as the
    # model's true output), regardless of this list.
    layers: list[int] = []
    max_seq_len: int | None = None
    prepend_bos: bool = True
    # Whether to enable "thinking" mode when applying a chat template (only
    # relevant for `chat` requests on models whose chat template supports it).
    enable_thinking: bool = False
    # Whether to preserve historical reasoning (`<think>`) blocks when applying a
    # chat template (only relevant for `chat` requests on models whose chat
    # template supports it, e.g. Qwen3.6). Keeping historical think blocks (the
    # default) stabilizes the chat-formatted token prefix across turns: without
    # it, templates strip prior `<think>` blocks from history, shifting token
    # positions every turn.
    preserve_thinking: bool = True
    # Stream results as NDJSON (one message per line). When false, the identical
    # path runs and all messages are buffered into a single JSON object.
    stream: bool = True
    # Sampling temperature for generated tokens. 0 = greedy (argmax).
    temperature: float = 1.0
    # Number of tokens to generate after the prompt. 0 = lens over the prompt
    # only (no generation).
    num_completion_tokens: int = 0
    # Token ids the client already has lens read-outs for (the previous
    # response's prompt + generated tokens, in order). The server computes the
    # longest common prefix with the freshly tokenized prompt and skips the
    # (expensive) per-layer read-out + emission for those positions, so a
    # follow-up turn only recomputes the new tokens. Position reuse is validated
    # by token id, so a divergent prefix simply reuses less (never wrong).
    cached_token_ids: list[int] = []
    # Exact input token ids to read out over, bypassing tokenization. When
    # provided (non-empty), `prompt`/`chat` are ignored, generation is disabled
    # (``num_completion_tokens`` is forced to 0), and the read-out runs over
    # exactly these ids. Used to faithfully reproduce a previously-computed run
    # (e.g. a shared jlens link) without depending on chat-template / tokenizer
    # drift — the lens read-out is a deterministic function of the token ids.
    input_token_ids: list[int] = []
    # Steering: readouts to additively inject (negatively, to suppress) into the
    # residual stream at every position, during prefill AND generation. Empty
    # (default) = no steering. When steering is active, prefix-reuse is disabled
    # (the cached read-outs from an unsteered run are no longer valid).
    steer_tokens: list[LensSteerToken] = []
    # Layers to inject the steering direction at. Empty = the read-out layers.
    steer_layers: list[int] = []
    # Signed steering strength as a fraction of each position's residual norm
    # (negative suppresses the readout). 0 = no steering.
    steer_strength: float = 0.0
    # When true, ABLATE the readout direction: project it out of the residual at
    # every steered layer/position (h <- h - (h.d_hat) d_hat) instead of
    # additively steering. Mutually exclusive with ``steer_strength`` (which is
    # ignored when ablating).
    steer_ablate: bool = False
    # SWAP: when set, replace the source readout (``steer_tokens[0]``) with this
    # target readout at every steered layer/position. The residual's projection
    # onto the source direction is removed and re-added (same magnitude) along
    # the target direction: ``h <- h - (h.s_hat) s_hat + (h.s_hat) t_hat``. This
    # is the causal "lens-vector swap" intervention and takes precedence over
    # ``steer_strength`` / ``steer_ablate`` when present. ``type`` should match
    # the source readout's lens type.
    swap_token: LensSteerToken | None = None
    # Whether to apply the steer/swap intervention to GENERATED tokens too. When
    # false (default), the intervention is applied only to the prompt positions
    # (prefill); generation then proceeds from the steered prompt context but the
    # newly generated positions are not themselves steered/swapped. When true,
    # the intervention is also applied at each generated position as it is
    # produced.
    steer_generated_tokens: bool = False
    # Whether to drop "non-word" tokens (punctuation / whitespace / symbol /
    # special tokens) from each position's per-layer read-out BEFORE selecting
    # the top-n, so the returned tokens are predominantly interesting word
    # tokens. The model's TRUE top-1 (output) token at each layer is always
    # preserved even when it is non-word. Probabilities are computed over the
    # FULL vocab (filtering only changes WHICH tokens are selected, not their
    # reported probabilities). Defaults to True.
    filter_non_word_tokens: bool = True
    # When true, if the server is already processing another request (the global
    # model lock is held), return HTTP 429 immediately instead of queueing and
    # waiting for the lock. This lets a client (e.g. the webapp) fail over to a
    # different inference server for this model. Defaults to False, preserving
    # the original behavior of waiting up to REQUEST_LOCK_TIMEOUT for the lock.
    fail_if_busy: bool = False


class LensTypeSlice(BaseModel):
    """Lens read-out for one (position, lens_type).

    All token references are STRINGS (decoded), never ids.
    """

    type: LensType
    # [n_layers, top_n]
    top_tokens: list[list[str]]
    top_probs: list[list[float]]


class LensMetaMessage(BaseModel):
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


class LensPromptToken(BaseModel):
    """A single chat-formatted prompt token (no lens read-out)."""

    position: int
    token: str
    # The token id, echoed so the client can send it back as `cached_token_ids`
    # on the next turn for prefix-reuse matching.
    id: int
    is_generated: bool = False


class LensPromptTokensMessage(BaseModel):
    """Emitted right after `meta` and before inference begins.

    Carries the chat-formatted prompt tokens (no lens read-outs) so the client
    can render the full conversation structure (user turn + assistant scaffold)
    immediately, instead of waiting for generation to finish.
    """

    kind: str = "prompt"
    tokens: list[LensPromptToken]


class LensTokenMessage(BaseModel):
    """One per token position: the token plus its per-type lens slices."""

    kind: str = "token"
    position: int
    token: str
    # The token id, echoed so the client can send it back as `cached_token_ids`
    # on the next turn for prefix-reuse matching.
    id: int
    is_generated: bool
    results: list[LensTypeSlice]


class LensDoneMessage(BaseModel):
    """Final streamed message."""

    kind: str = "done"
    seq_len: int
    prompt_len: int
    vocab_size: int
    completion: str


class LensPromptResponse(BaseModel):
    """Non-streaming response: the same messages buffered into one object."""

    meta: LensMetaMessage
    tokens: list[LensTokenMessage]
    done: LensDoneMessage


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


def _decode_display_tokens(tokenizer, token_ids: list[int], cache: dict[int, str]) -> list[str]:
    """Per-position display strings, repairing characters split across tokens.

    A single emoji (or other multi-byte codepoint) is often split across several
    tokens; decoded individually each fragment is just a replacement char (`),
    so the glyph never shows. Here we detect a run of such fragments, decode the
    run together to recover the real character, and assign that combined string
    to EVERY position in the run (so the emoji shows at each contributing token
    rather than a row of `).
    """
    n = len(token_ids)
    out: list[str] = [""] * n
    i = 0
    while i < n:
        solo = _decode_token(tokenizer, int(token_ids[i]), cache)
        if _REPLACEMENT_CHAR not in solo:
            out[i] = solo
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
                out[k] = combined
            i = j + 1
        else:
            # Unrecoverable; leave the lone replacement char for this position.
            out[i] = solo
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
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def build_token_ids(model, request: LensPromptRequest) -> list[int]:
    """Build input token ids from either a raw prompt or a chat conversation."""
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized")

    if request.chat is not None:
        messages = [{"role": m.role, "content": m.content} for m in request.chat]
        # If the final message is an assistant turn, treat it as a PREFILL: keep
        # that turn open (no end-of-turn token, no fresh assistant scaffold) so
        # generation continues from the prefilled text rather than starting a new
        # assistant turn after it.
        is_prefill = len(messages) > 0 and messages[-1]["role"] == "assistant"
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            kwargs = {}
            # Only pass `enable_thinking` if the chat template actually references
            # it; templates that don't will otherwise ignore (or reject) it.
            if "enable_thinking" in chat_template:
                kwargs["enable_thinking"] = request.enable_thinking
            # Likewise for `preserve_thinking` (e.g. Qwen3.6): only pass it when
            # the template references it, so older templates don't reject it.
            if "preserve_thinking" in chat_template:
                kwargs["preserve_thinking"] = request.preserve_thinking
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=not is_prefill,
                continue_final_message=is_prefill,
                **kwargs,
            )
            return _coerce_token_ids(ids)
        # Tokenizer has no chat template: fall back to a generic ChatML template.
        text = apply_generic_chat_template(
            messages,
            add_generation_prompt=not is_prefill,
            continue_final_message=is_prefill,
        )
        return _encode_raw_text(tokenizer, text, request.prepend_bos)

    return _encode_raw_text(tokenizer, request.prompt or "", request.prepend_bos)


# --------------------------------------------------------------------------- #
# Incremental generation + residual capture (KV-cached, forward hooks)
# --------------------------------------------------------------------------- #

# One streamed position: (token_id, is_generated, {layer: residual[d_model]}).
ResidualStep = tuple[int, bool, dict[int, torch.Tensor]]


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


def _make_capture_hook(captures: dict[int, torch.Tensor], layer: int):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captures[layer] = hidden.detach()

    return hook


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
            decoded = tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
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
        raise ValueError(
            f"steer token_id {token_id} out of range for unembedding vocab "
            f"size {vocab_size}"
        )


def _unembed_vector(model, token_id: int) -> torch.Tensor:
    """Residual-space unembedding direction for ``token_id`` (float32).

    ``token_id`` is bounds-checked against the actual unembedding matrix on the
    host before indexing: an out-of-range id would otherwise trigger a
    device-side assert that poisons the entire CUDA context for the process.
    """
    if isinstance(model, HookedTransformer):
        weight = model.W_U  # W_U: [d_model, vocab]
        vocab_size = weight.shape[1]
        _check_token_id_in_range(token_id, vocab_size)
        return weight[:, token_id].detach().float()
    hf = model._model
    _layers, _norm, head = _hf_decoder_modules(hf)
    weight = head.weight  # lm_head: [vocab, d_model]
    vocab_size = weight.shape[0]
    _check_token_id_in_range(token_id, vocab_size)
    return weight[token_id].detach().float()


def _build_steer_deltas(
    model,
    lens: LoadedJacobianLens | None,
    steer_tokens: list[LensSteerToken],
    steer_layers: list[int],
) -> dict[int, torch.Tensor]:
    """Build the per-layer unit direction to inject, summed across steer tokens.

    For a ``JACOBIAN_LENS`` token at a fitted layer ``l`` the direction is
    ``J_bar_l^T @ w_t`` (equivalently ``w_t @ J_bar_l``), the residual-space
    direction whose J-lens readout is the token; otherwise the plain unembedding
    direction ``w_t``. Each per-layer direction is unit-normalized before
    summing so multiple tokens reinforce sensibly.
    """
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized")
    index = _decoded_string_to_ids(tokenizer)
    resolved = [
        (_resolve_steer_token_id(index, spec.token), spec.type)
        for spec in steer_tokens
    ]

    deltas: dict[int, torch.Tensor] = {}
    for layer in steer_layers:
        acc: torch.Tensor | None = None
        for token_id, lens_type in resolved:
            w = _unembed_vector(model, token_id)  # [d_model]
            if (
                lens_type == LensType.JACOBIAN_LENS
                and lens is not None
                and layer in lens.jacobians
            ):
                j_bar = lens.jacobian_on(layer, w.device).to(torch.float32)
                direction = w @ j_bar  # J_bar^T @ w
            else:
                direction = w
            norm = torch.linalg.vector_norm(direction)
            if norm > 0:
                direction = direction / norm
            acc = direction if acc is None else acc + direction
        if acc is not None:
            deltas[layer] = acc
    return deltas


def _bos_skip_mask(
    token_ids: list[int], bos_token_id: int | None, device
) -> torch.Tensor | None:
    """Bool mask ``[1, seq, 1]`` marking BOS positions to leave unmodified.

    Returns ``None`` when there is no BOS id or the sequence contains none, so
    the caller skips masking entirely. Only the EXACT bos id is matched, so chat
    special tokens (turn markers, etc.) are still steered.
    """
    if bos_token_id is None:
        return None
    flags = [tid == bos_token_id for tid in token_ids]
    if not any(flags):
        return None
    return torch.tensor(flags, dtype=torch.bool, device=device).view(1, -1, 1)


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

    ``skip_mask`` (bool, broadcastable to ``[..., seq, 1]``) marks positions to
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


def _make_steer_hook_hf(
    delta: torch.Tensor,
    strength: float,
    ablate: bool = False,
    skip_holder: dict | None = None,
):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        skip = skip_holder["mask"] if skip_holder is not None else None
        steered = _apply_steer(hidden, delta, strength, ablate, skip_mask=skip)
        if isinstance(output, tuple):
            return (steered, *tuple(output[1:]))
        return steered

    return hook


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

    ``skip_mask`` (bool, broadcastable to ``[..., seq, 1]``) marks positions to
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


def _make_swap_hook_hf(
    src_delta: torch.Tensor,
    tgt_delta: torch.Tensor,
    skip_holder: dict | None = None,
):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        skip = skip_holder["mask"] if skip_holder is not None else None
        swapped = _apply_swap(hidden, src_delta, tgt_delta, skip_mask=skip)
        if isinstance(output, tuple):
            return (swapped, *tuple(output[1:]))
        return swapped

    return hook


def _hf_decoder_modules(hf):
    """Locate (decoder_layers, final_norm, lm_head) on a HF causal-LM.

    Handles the common decoder-only families (Llama/Qwen/Gemma/Mistral use
    ``model.layers`` + ``model.norm`` + ``lm_head``; GPT-2/Falcon use
    ``transformer.h`` + ``transformer.ln_f``; GPT-NeoX uses ``gpt_neox.*``).

    Also handles multimodal wrappers (e.g. ``Gemma3ForConditionalGeneration``)
    where the text decoder is nested one level deeper under
    ``model.language_model`` rather than directly on ``model``.
    """
    base = (
        getattr(hf, "model", None)
        or getattr(hf, "transformer", None)
        or getattr(hf, "gpt_neox", None)
    )

    def _decoder_in(obj):
        """Return (layers, norm) if ``obj`` directly exposes a decoder stack."""
        if obj is None:
            return None, None
        layers = getattr(obj, "layers", None) or getattr(obj, "h", None)
        norm = (
            getattr(obj, "norm", None)
            or getattr(obj, "ln_f", None)
            or getattr(obj, "final_layer_norm", None)
        )
        return layers, norm

    layers, norm = _decoder_in(base)
    # Multimodal wrappers (Gemma3/PaliGemma/etc.) nest the text decoder under
    # ``language_model`` (or another ``model``/``text_model`` attribute).
    if (layers is None or norm is None) and base is not None:
        for attr in ("language_model", "text_model", "model"):
            inner = getattr(base, attr, None)
            inner_layers, inner_norm = _decoder_in(inner)
            if inner_layers is not None and inner_norm is not None:
                layers, norm = inner_layers, inner_norm
                base = inner
                break

    head = (
        getattr(hf, "lm_head", None)
        or getattr(hf, "embed_out", None)
        or getattr(base, "lm_head", None)
    )
    if layers is None or norm is None or head is None:
        raise ValueError(
            f"Could not locate decoder layers / final norm / lm_head on "
            f"{type(hf).__name__} for the lens read-out."
        )
    return layers, norm, head


def _iter_residuals(
    model,
    prompt_token_ids: list[int],
    union_layers: list[int],
    *,
    num_completion_tokens: int,
    temperature: float,
    eos_token_id: int | None,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    bos_token_id: int | None = None,
) -> Iterator[ResidualStep]:
    """Stream ``(token_id, is_generated, residuals)`` one position at a time.

    Prompt positions are yielded first (from the prefill), then each generated
    token as it is produced by a KV-cached decode step. Generation stops at
    ``num_completion_tokens`` or EOS (whichever first). ``residuals`` maps each
    requested layer to its ``[d_model]`` residual at that position.

    When ``steer_deltas`` is provided and ``steer_strength`` is non-zero, each
    layer's residual is additively steered (at every position, prefill +
    generation) before being captured/propagated, so both the generated tokens
    and the read-outs reflect the steering. When ``swap_deltas`` is provided,
    a SWAP intervention is applied instead (``steer_deltas`` holds the per-layer
    source directions, ``swap_deltas`` the targets); swap takes precedence over
    additive steering / ablation.
    """
    if isinstance(model, HookedTransformer):
        yield from _iter_residuals_tlens(
            model,
            prompt_token_ids,
            union_layers,
            num_completion_tokens=num_completion_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
            steer_deltas=steer_deltas,
            steer_strength=steer_strength,
            steer_ablate=steer_ablate,
            swap_deltas=swap_deltas,
            steer_generated=steer_generated,
            bos_token_id=bos_token_id,
        )
        return
    if isinstance(model, StandardizedTransformer):
        yield from _iter_residuals_hf(
            model,
            prompt_token_ids,
            union_layers,
            num_completion_tokens=num_completion_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
            steer_deltas=steer_deltas,
            steer_strength=steer_strength,
            steer_ablate=steer_ablate,
            swap_deltas=swap_deltas,
            steer_generated=steer_generated,
            bos_token_id=bos_token_id,
        )
        return
    raise ValueError(
        f"Lens endpoint does not support model type {type(model).__name__} "
        "(only TransformerLens and nnsight/nnterp models)."
    )


def _iter_residuals_tlens(
    model: HookedTransformer,
    prompt_token_ids: list[int],
    union_layers: list[int],
    *,
    num_completion_tokens: int,
    temperature: float,
    eos_token_id: int | None,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    bos_token_id: int | None = None,
) -> Iterator[ResidualStep]:
    device = model.cfg.device
    captures: dict[int, torch.Tensor] = {}
    name_to_layer = {f"blocks.{layer}.hook_resid_post": layer for layer in union_layers}

    def hook_fn(tensor, hook):
        captures[name_to_layer[hook.name]] = tensor.detach()

    # Never intervene on the BOS token: its attention-sink residual has a huge
    # norm, so the (un-normalized) steer/swap projection is spuriously large
    # there and injects artifacts the read-out (which normalizes) never showed.
    # We skip only the exact BOS id, so chat special tokens are still steered.
    # ``skip_holder["mask"]`` is updated per forward (prompt mask, then per
    # generated token) and read by the steer/swap hooks below.
    skip_holder: dict = {"mask": _bos_skip_mask(prompt_token_ids, bos_token_id, device)}
    gen_bos_mask = (
        torch.ones((1, 1, 1), dtype=torch.bool, device=device) if bos_token_id is not None else None
    )

    # Steer/swap hooks run BEFORE the capture hooks at the same resid_post point,
    # so the modified residual is both captured (read out) and propagated forward.
    # They are kept separate from the capture hooks so we can drop them during
    # generation when ``steer_generated`` is false (intervene on the prompt only).
    swapping = bool(swap_deltas) and bool(steer_deltas)
    steering = bool(steer_deltas) and (steer_strength != 0.0 or steer_ablate)
    steer_fwd_hooks: list = []
    if swapping and steer_deltas is not None and swap_deltas is not None:

        def make_swap_hook(src: torch.Tensor, tgt: torch.Tensor):
            def swap_hook(tensor, hook):  # noqa: ARG001
                return _apply_swap(tensor, src, tgt, skip_mask=skip_holder["mask"])

            return swap_hook

        for layer, tgt in swap_deltas.items():
            src = steer_deltas.get(layer)
            if src is None:
                continue
            steer_fwd_hooks.append((f"blocks.{layer}.hook_resid_post", make_swap_hook(src, tgt)))
    elif steering and steer_deltas is not None:

        def make_steer_hook(delta: torch.Tensor):
            def steer_hook(tensor, hook):  # noqa: ARG001
                return _apply_steer(tensor, delta, steer_strength, steer_ablate, skip_mask=skip_holder["mask"])

            return steer_hook

        for layer, delta in steer_deltas.items():
            steer_fwd_hooks.append((f"blocks.{layer}.hook_resid_post", make_steer_hook(delta)))
    capture_fwd_hooks = [(name, hook_fn) for name in name_to_layer]
    # Prefill always applies the intervention to the prompt; generation only
    # applies it when steer_generated is set.
    prefill_fwd_hooks = steer_fwd_hooks + capture_fwd_hooks
    gen_fwd_hooks = (steer_fwd_hooks if steer_generated else []) + capture_fwd_hooks
    cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1)

    tokens = torch.tensor([prompt_token_ids], device=device)
    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=prefill_fwd_hooks,
            past_kv_cache=cache,
        )
    for pos, token_id in enumerate(prompt_token_ids):
        yield (
            int(token_id),
            False,
            {layer: captures[layer][0, pos] for layer in union_layers},
        )

    last_logits = logits[0, -1, :]
    generated = 0
    while generated < num_completion_tokens:
        next_id = _sample_token(last_logits, temperature)
        generated += 1
        captures.clear()
        # Skip the intervention if this generated token is itself a BOS.
        skip_holder["mask"] = (
            gen_bos_mask if (bos_token_id is not None and next_id == bos_token_id) else None
        )
        with torch.no_grad():
            logits = model.run_with_hooks(
                torch.tensor([[next_id]], device=device),
                return_type="logits",
                fwd_hooks=gen_fwd_hooks,
                past_kv_cache=cache,
            )
        yield (
            next_id,
            True,
            {layer: captures[layer][0, -1] for layer in union_layers},
        )
        if eos_token_id is not None and next_id == eos_token_id:
            break
        last_logits = logits[0, -1, :]


def _iter_residuals_hf(
    model: StandardizedTransformer,
    prompt_token_ids: list[int],
    union_layers: list[int],
    *,
    num_completion_tokens: int,
    temperature: float,
    eos_token_id: int | None,
    steer_deltas: dict[int, torch.Tensor] | None = None,
    steer_strength: float = 0.0,
    steer_ablate: bool = False,
    swap_deltas: dict[int, torch.Tensor] | None = None,
    steer_generated: bool = False,
    bos_token_id: int | None = None,
) -> Iterator[ResidualStep]:
    # Reach through to the underlying HF model, bypassing nnsight tracing. nnsight
    # loads weights lazily on the meta device, so ensure they are materialized.
    if not getattr(model, "dispatched", False):
        model.dispatch()
    hf = model._model
    layer_modules, _norm, _head = _hf_decoder_modules(hf)
    device = next(hf.parameters()).device

    captures: dict[int, torch.Tensor] = {}
    # Never intervene on the BOS token (attention-sink norm makes the projection
    # spuriously large). Only the exact BOS id is skipped, so chat special tokens
    # are still steered. The holder is updated per forward (prompt, then per
    # generated token) and read by the steer/swap hooks.
    skip_holder: dict = {"mask": _bos_skip_mask(prompt_token_ids, bos_token_id, device)}
    gen_bos_mask = (
        torch.ones((1, 1, 1), dtype=torch.bool, device=device) if bos_token_id is not None else None
    )

    # Steer/swap hooks are registered BEFORE the capture hooks so the modified
    # output is what gets captured (read out) and propagated to later layers.
    swapping = bool(swap_deltas) and bool(steer_deltas)
    steering = bool(steer_deltas) and (steer_strength != 0.0 or steer_ablate)
    if swapping and steer_deltas is not None and swap_deltas is not None:
        steer_handles = [
            layer_modules[layer].register_forward_hook(
                _make_swap_hook_hf(steer_deltas[layer], tgt, skip_holder=skip_holder)
            )
            for layer, tgt in swap_deltas.items()
            if layer in steer_deltas
        ]
    elif steering and steer_deltas is not None:
        steer_handles = [
            layer_modules[layer].register_forward_hook(
                _make_steer_hook_hf(delta, steer_strength, steer_ablate, skip_holder=skip_holder)
            )
            for layer, delta in steer_deltas.items()
        ]
    else:
        steer_handles = []
    capture_handles = [
        layer_modules[layer].register_forward_hook(_make_capture_hook(captures, layer))
        for layer in union_layers
    ]
    handles = steer_handles + capture_handles
    try:
        input_ids = torch.tensor([prompt_token_ids], device=device)
        with torch.no_grad():
            out = hf(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        for pos, token_id in enumerate(prompt_token_ids):
            yield (
                int(token_id),
                False,
                {layer: captures[layer][0, pos] for layer in union_layers},
            )

        # The intervention always applies to the prompt (prefill above). For
        # generation it only applies when steer_generated is set; otherwise drop
        # the steer/swap hooks here so the generated positions are not modified
        # (the capture hooks stay so we still read out the generated tokens).
        if not steer_generated:
            for handle in steer_handles:
                handle.remove()
            steer_handles = []
            handles = capture_handles

        last_logits = out.logits[0, -1, :]
        generated = 0
        while generated < num_completion_tokens:
            next_id = _sample_token(last_logits, temperature)
            generated += 1
            captures.clear()
            # Skip the intervention if this generated token is itself a BOS.
            skip_holder["mask"] = (
                gen_bos_mask if (bos_token_id is not None and next_id == bos_token_id) else None
            )
            with torch.no_grad():
                out = hf(
                    input_ids=torch.tensor([[next_id]], device=device),
                    past_key_values=past,
                    use_cache=True,
                )
            past = out.past_key_values
            yield (
                next_id,
                True,
                {layer: captures[layer][0, -1] for layer in union_layers},
            )
            if eos_token_id is not None and next_id == eos_token_id:
                break
            last_logits = out.logits[0, -1, :]
    finally:
        for handle in handles:
            handle.remove()


def _decode_residuals(model, residuals_2d: torch.Tensor) -> torch.Tensor:
    """Decode ``[n_rows, d_model]`` residuals to ``[n_rows, vocab]`` logits using
    the model's own final norm + unembedding (no Jacobian; caller applies that)."""
    with torch.no_grad():
        if isinstance(model, HookedTransformer):
            param_dtype = model.W_U.dtype
            return model.unembed(model.ln_final(residuals_2d.to(param_dtype)))
        # nnsight/nnterp: use the dispatched HF modules directly.
        hf = model._model
        _layers, norm, head = _hf_decoder_modules(hf)
        param = next(hf.parameters())
        return head(norm(residuals_2d.to(device=param.device, dtype=param.dtype)))


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
) -> LayerLogitsByType:
    """Read out per-layer logits for every requested lens type in ONE forward pass.

    The residual stream is captured once and reused: a LOGIT_LENS row decodes the
    residual directly, a JACOBIAN_LENS row first transports it with ``J_bar``. So
    requesting both types only costs the extra per-layer projections, not a second
    model forward pass. The final layer (not fitted) is always decoded directly,
    giving the model's true output.
    """
    if isinstance(model, HookedTransformer):
        return _compute_logits_for_types_tlens(
            model, token_ids, layers_by_type, lens, softcap
        )
    if isinstance(model, StandardizedTransformer):
        return _compute_logits_for_types_nnsight(
            model, token_ids, layers_by_type, lens, softcap
        )
    raise ValueError(
        f"Lens endpoint does not support model type {type(model).__name__} "
        "(only TransformerLens and nnsight/nnterp models)."
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


def _compute_logits_for_types_tlens(
    model: HookedTransformer,
    token_ids: list[int],
    layers_by_type: dict[LensType, list[int]],
    lens: LoadedJacobianLens | None,
    softcap: float | None,
) -> LayerLogitsByType:
    device = model.cfg.device
    param_dtype = model.W_U.dtype
    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)

    union = _union_layers(layers_by_type)
    wanted = {f"blocks.{layer}.hook_resid_post" for layer in union}
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name in wanted
        )
    residuals = {layer: cache[f"blocks.{layer}.hook_resid_post"][0] for layer in union}

    out: LayerLogitsByType = {}
    for lens_type, layers in layers_by_type.items():
        use_jacobian = lens_type == LensType.JACOBIAN_LENS and lens is not None
        layer_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = residuals[layer]  # [seq, d_model]
            if use_jacobian and lens is not None and layer in lens.jacobians:
                residual = lens.transport(residual.float(), layer)
            residual = residual.to(param_dtype)
            with torch.no_grad():
                logits = model.unembed(model.ln_final(residual))  # [seq, vocab]
            logits = apply_final_logit_softcap(logits, softcap)
            # Keep on-device: the vocab-sized ranking in the read-out runs on
            # the GPU. Moving to CPU here would force that work onto the CPU.
            # Keep the model dtype (no float32 upcast) to halve the resident
            # vocab-sized tensors.
            layer_logits[layer] = logits.detach()
        out[lens_type] = layer_logits
    return out


def _compute_logits_for_types_nnsight(
    model: StandardizedTransformer,
    token_ids: list[int],
    layers_by_type: dict[LensType, list[int]],
    lens: LoadedJacobianLens | None,
    softcap: float | None,
) -> LayerLogitsByType:
    config = Config.get_instance()
    model_dtype = STR_TO_DTYPE.get(config.model_dtype, torch.float32)
    device = torch.device(getattr(model, "device", None) or config.device or "cpu")
    tokens = torch.tensor(token_ids)
    union = _union_layers(layers_by_type)

    saves: dict[tuple[LensType, int], object] = {}
    with model.trace(tokens):
        hiddens = {layer: model.layers_output[layer] for layer in union}
        for lens_type, layers in layers_by_type.items():
            use_jacobian = lens_type == LensType.JACOBIAN_LENS and lens is not None
            for layer in layers:
                hidden = hiddens[layer]  # [1, seq, d_model]
                if use_jacobian and lens is not None and layer in lens.jacobians:
                    j_bar = lens.jacobian_on(layer, device).to(torch.float32)
                    hidden = (hidden.to(torch.float32) @ j_bar.T).to(model_dtype)
                logits = model.project_on_vocab(hidden)  # [1, seq, vocab]
                if softcap is not None:
                    logits = softcap * torch.tanh(logits / softcap)
                # Use nnsight.save(...) (not the .save() method): the package
                # disables nnsight PYMOUNT, so the method-based save is unavailable.
                saves[(lens_type, layer)] = nnsight.save(logits)

    out: LayerLogitsByType = {}
    for lens_type, layers in layers_by_type.items():
        layer_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            saved = saves[(lens_type, layer)]
            tensor = saved[0] if saved.dim() == 3 else saved  # type: ignore[attr-defined]
            # Keep on-device (see _compute_logits_for_types_tlens): the ranking
            # in the read-out is GPU-resident. Keep the model dtype (no float32
            # upcast) to halve the resident vocab-sized tensors.
            layer_logits[layer] = tensor.detach()
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
            [_decode_token(self.tokenizer, int(token_id), self.decode_cache) for token_id in row]
            for row in top_idx_np
        ]

        return LensTypeSlice(
            type=self.lens_type,
            top_tokens=top_tokens,
            # Round to 4 decimals (0.01% resolution) to cut serialized payload
            # size. The client only renders integer percentages and normalized
            # per-layer heatmap weights, so extra precision is never visible.
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


# How many prompt positions to decode per batched read-out matmul. The
# per-layer unembedding matmul against the (large) vocab re-streams the
# ``lm_head`` weight from HBM, so it is memory-bound when decoding one position
# at a time. Batching ``chunk_size * n_layers`` rows into a single matmul
# amortizes that weight read across positions, crossing into compute-bound
# territory (the win saturates once ``chunk_size * n_layers`` exceeds the GPU's
# FLOP:byte ridge, ~150 on A100 / ~300 on H100). Generated tokens are produced
# one at a time and are decoded individually (an effective chunk of 1).
_READOUT_CHUNK_SIZE = 8


def _chunk_position_logits(
    model,
    lens_type: LensType,
    layers: list[int],
    residuals_list: list[dict[int, torch.Tensor]],
    lens: LoadedJacobianLens | None,
    softcap: float | None,
) -> list[torch.Tensor]:
    """Decode several positions' per-layer residuals in ONE batched matmul.

    Returns one ``[n_layers, vocab]`` logit tensor per input position. All
    ``len(positions) * n_layers`` rows are stacked and decoded together so the
    unembedding weight is read from HBM once for the whole chunk instead of once
    per position. The read-out logits are kept in the model dtype (no float32
    upcast): the unembedding matmul is already bf16, so widening afterwards only
    doubles the vocab-sized tensor's bandwidth without recovering precision.

    For JACOBIAN_LENS, fitted layers are first transported with ``J_bar``; the
    final (unfitted) layer is decoded directly (``J = I``), giving the model's
    true output.
    """
    use_jacobian = lens_type == LensType.JACOBIAN_LENS and lens is not None
    n_layers = len(layers)
    rows: list[torch.Tensor] = []
    for residuals in residuals_list:
        for layer in layers:
            residual = residuals[layer]  # [d_model]
            if use_jacobian and lens is not None and layer in lens.jacobians:
                residual = lens.transport(residual.float(), layer)
            # Keep a uniform dtype across rows so transported (float32) and
            # directly-decoded (model-dtype) layers can be stacked together;
            # `_decode_residuals` recasts to the param dtype for the matmul, so
            # this is numerically a no-op.
            rows.append(residual.float())
    stacked = torch.stack(rows, dim=0)  # [n_positions * n_layers, d_model]
    logits = _decode_residuals(model, stacked)  # [n_positions * n_layers, vocab]
    logits = apply_final_logit_softcap(logits, softcap)
    vocab = logits.shape[-1]
    logits = logits.view(len(residuals_list), n_layers, vocab)
    return [logits[pos] for pos in range(len(residuals_list))]


def _build_messages(
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
) -> Iterator[BaseModel]:
    """Yield the ordered stream of messages: meta -> token* -> done.

    A plain (synchronous) generator; the route wraps it to manage the model lock
    and NDJSON serialization. Residuals are produced incrementally (prefill, then
    one KV-cached decode step per generated token) and each position's lens slice
    is emitted as soon as it is computed — true token-by-token streaming.

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
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)

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
                token=prompt_display[pos],
                id=int(token_id),
                is_generated=False,
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

    def _emit(entry: LensTokenMessage, token_str: str) -> LensTokenMessage:
        entry.token = token_str
        return entry

    def _flush_pending_as_is() -> list[LensTokenMessage]:
        flushed = [_emit(p, _decode_token(tokenizer, p.id, decode_cache)) for p in pending]
        pending.clear()
        return flushed

    def _emit_chunk(
        buf: list[tuple[int, int, bool, dict[int, torch.Tensor]]],
    ) -> Iterator[LensTokenMessage]:
        """Decode a chunk of buffered positions in ONE batched read-out matmul,
        then emit each position's token message in order (with multi-byte-char
        repair). An empty chunk is a no-op."""
        nonlocal vocab_size
        if not buf:
            return

        residuals_list = [residuals for (_, _, _, residuals) in buf]
        # Per-position results, computed type-by-type with a single batched
        # read-out matmul per type across the whole chunk (the win in #2).
        per_pos_results: list[list[LensTypeSlice]] = [[] for _ in buf]
        for lens_type in requested_types:
            logits_list = _chunk_position_logits(
                model, lens_type, layers_by_type[lens_type], residuals_list, lens, softcap
            )
            state = states.get(lens_type)
            if state is None:
                vocab_size = int(logits_list[0].shape[-1])
                state = _TypeReadoutState(
                    lens_type,
                    tokenizer,
                    vocab_size,
                    top_n=request.top_n,
                    decode_cache=decode_cache,
                    filter_non_word=request.filter_non_word_tokens,
                )
                states[lens_type] = state
            for i, logits in enumerate(logits_list):
                per_pos_results[i].append(state.process(logits))

        for i, (pos, token_id, is_generated, _residuals) in enumerate(buf):
            entry = LensTokenMessage(
                position=pos,
                token="",
                id=int(token_id),
                is_generated=is_generated,
                results=per_pos_results[i],
            )
            solo = _decode_token(tokenizer, int(token_id), decode_cache)
            if _REPLACEMENT_CHAR not in solo:
                # A self-contained token: flush any stuck fragment run first, then
                # emit this token normally.
                for flushed in _flush_pending_as_is():
                    yield flushed
                yield _emit(entry, solo)
            else:
                # A fragment: buffer it and see if the run now decodes cleanly.
                pending.append(entry)
                combined = tokenizer.decode(
                    [p.id for p in pending], clean_up_tokenization_spaces=False
                )
                if _REPLACEMENT_CHAR not in combined:
                    for p in pending:
                        yield _emit(p, combined)
                    pending.clear()
                elif len(pending) >= _MAX_MULTI_TOKEN_CHAR:
                    for flushed in _flush_pending_as_is():
                        yield flushed

            if is_generated:
                completion_ids.append(int(token_id))

    # Prompt positions are buffered and decoded in chunks of ``_READOUT_CHUNK_SIZE``
    # (one batched matmul per chunk); generated tokens arrive one at a time and are
    # decoded individually so they keep streaming token-by-token.
    chunk_buf: list[tuple[int, int, bool, dict[int, torch.Tensor]]] = []
    for token_id, is_generated, residuals in _iter_residuals(
        model,
        prompt_token_ids,
        union_layers,
        num_completion_tokens=request.num_completion_tokens,
        temperature=request.temperature,
        eos_token_id=eos_token_id,
        steer_deltas=steer_deltas,
        steer_strength=steer_strength,
        steer_ablate=steer_ablate,
        swap_deltas=swap_deltas,
        steer_generated=steer_generated,
        bos_token_id=bos_token_id,
    ):
        # Skip the read-out + emission for positions the client already has
        # (matched token-id prefix). Generated positions are always past the
        # prompt, so they are never skipped.
        if position < reuse_len:
            if is_generated:
                completion_ids.append(int(token_id))
            position += 1
            continue

        if is_generated:
            # Flush any buffered prompt positions first (to preserve order), then
            # emit this generated token on its own.
            yield from _emit_chunk(chunk_buf)
            chunk_buf = []
            yield from _emit_chunk([(position, int(token_id), True, residuals)])
            position += 1
            continue

        chunk_buf.append((position, int(token_id), False, residuals))
        position += 1
        if len(chunk_buf) >= _READOUT_CHUNK_SIZE:
            yield from _emit_chunk(chunk_buf)
            chunk_buf = []

    # Flush any remaining buffered prompt positions (last partial chunk).
    yield from _emit_chunk(chunk_buf)
    chunk_buf = []

    # Any trailing fragments that never completed: emit them best-effort.
    for flushed in _flush_pending_as_is():
        yield flushed

    completion = (
        tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False)
        if completion_ids
        else ""
    )
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

    nnsight/nnterp lazily initializes internal state on the first ``model.trace``
    that applies the Jacobian transport; on that very first trace the transported
    layers come back degenerate (every transported layer collapses to uniform
    ``1/vocab`` logits, while the directly-decoded final layer is fine). Doing a
    throwaway pass here moves that one-time initialization to startup so the first
    *real* JACOBIAN_LENS request is correct.

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
    if not isinstance(model, (HookedTransformer, StandardizedTransformer)):
        return

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return

    config = Config.get_instance()
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
            lens_type: _select_layers(lens_type, n_layers, lens, layers=[])
            for lens_type in requested_types
        }
        softcap = resolve_final_logit_softcap(
            model,
            np_model_id=getattr(config, "model_id", None),
            hf_model_id=getattr(config, "custom_hf_model_id", None)
            or getattr(config, "override_model_id", None),
        )

        _compute_logits_for_types(model, token_ids, layers_by_type, lens, softcap)

        # Pre-build the non-word-token mask (a one-time full-vocab decode) so the
        # first request that enables filtering doesn't pay for it inline.
        try:
            vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
            _word_token_mask(tokenizer, int(vocab_size))
        except Exception:  # noqa: BLE001
            logger.exception("Word-mask warmup failed (non-fatal)")

        logger.info("Lens warmup completed (%d token(s)).", len(token_ids))
    except Exception:  # noqa: BLE001
        logger.exception("Lens warmup failed (non-fatal)")


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


async def _acquire_request_lock(fail_if_busy: bool = False) -> bool:
    """Acquire the global model lock with the configured timeout.

    Acquired in the route handler (not via a decorator) so we can return a
    proper HTTP status BEFORE the streaming response body starts; the lock is
    then held for the lifetime of the stream and released in the generator's
    ``finally`` (Starlette iterates a StreamingResponse body after the handler
    returns, so a decorator-scoped lock would be released before generation even
    runs).

    Returns ``True`` if the lock was acquired. Returns ``False`` only when
    ``fail_if_busy`` is set and the lock is already held, so the caller can fail
    fast (e.g. respond 429 and let the client try a different server). The
    ``locked()`` check and the (non-blocking, since the lock is free) acquire
    below run with no ``await`` between them, so this reliably reports "busy"
    without racing another request.
    """
    if request_lock.locked():
        if fail_if_busy:
            return False
        logger.warning(
            "[LOCK] Lens request waiting for lock (another request in progress)..."
        )
    if REQUEST_LOCK_TIMEOUT > 0:
        await asyncio.wait_for(request_lock.acquire(), timeout=REQUEST_LOCK_TIMEOUT)
    else:
        await request_lock.acquire()
    return True


@router.post("/lens/prompt")
async def lens_prompt(request: LensPromptRequest, http_request: Request):
    config = Config.get_instance()
    model = Model.get_instance()

    # ---- validation (before the stream starts, so we can return proper 4xx) ---
    use_input_token_ids = len(request.input_token_ids) > 0
    # When exact token ids are supplied we read out over them verbatim (no
    # tokenization, no generation), so `prompt`/`chat` are not required.
    if not use_input_token_ids and (request.prompt is None) == (request.chat is None):
        return JSONResponse(
            content={"error": "Provide exactly one of 'prompt' or 'chat'"},
            status_code=400,
        )

    # De-duplicate the requested types while preserving order.
    requested_types: list[LensType] = list(dict.fromkeys(request.type))
    if not requested_types:
        return JSONResponse(
            content={"error": "Provide at least one lens type in 'type'"},
            status_code=400,
        )

    if not isinstance(model, (HookedTransformer, StandardizedTransformer)):
        return JSONResponse(
            content={
                "error": (
                    "The lens endpoint is only supported on TransformerLens and "
                    "nnsight/nnterp models (not vLLM/chatspace)."
                )
            },
            status_code=400,
        )

    if request.temperature < 0:
        return JSONResponse(
            content={"error": "temperature must be >= 0"}, status_code=400
        )
    if request.num_completion_tokens < 0:
        return JSONResponse(
            content={"error": "num_completion_tokens must be >= 0"}, status_code=400
        )

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
        return JSONResponse(
            content={"error": "Prompt produced zero tokens"}, status_code=400
        )

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

    # Longest common token-id prefix with what the client already has. Positions
    # in this prefix have identical preceding context (causal attention), so the
    # client's cached read-outs are still valid and we skip recomputing them.
    # Bounded to the prompt length (generation always recomputes).
    reuse_len = _common_prefix_len(token_ids, request.cached_token_ids)

    n_layers = config.num_layers
    if n_layers is None:
        return JSONResponse(
            content={"error": "Model layer count not initialized"}, status_code=500
        )

    layers_by_type = {
        lens_type: _select_layers(lens_type, n_layers, lens, request.layers)
        for lens_type in requested_types
    }

    # ---- steering / swap: resolve readouts -> per-layer injection directions ----
    # SWAP replaces the source readout (steer_tokens[0]) with `swap_token`; it
    # needs the source directions too, so it reuses the steer-delta builder.
    swap_active = request.swap_token is not None and len(request.steer_tokens) > 0
    steer_active = len(request.steer_tokens) > 0 and (
        request.steer_strength != 0.0 or request.steer_ablate
    )
    steer_deltas: dict[int, torch.Tensor] = {}
    swap_deltas: dict[int, torch.Tensor] = {}
    if steer_active or swap_active:
        # The client's explicit layer list is used verbatim: an empty list means
        # no steering/swap (e.g. the user deselected every layer).
        try:
            steer_deltas = _build_steer_deltas(
                model, lens, request.steer_tokens, request.steer_layers
            )
            if swap_active and request.swap_token is not None:
                swap_deltas = _build_steer_deltas(
                    model, lens, [request.swap_token], request.steer_layers
                )
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
        hf_model_id=getattr(config, "custom_hf_model_id", None)
        or getattr(config, "override_model_id", None),
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
    except asyncio.TimeoutError:
        logger.error("[LOCK] Timeout waiting for lock on lens request")
        return JSONResponse(
            content={"error": "Request timed out waiting for lock"},
            status_code=503,
        )
    if not acquired:
        # Server is busy with another request and the client opted to fail fast
        # so it can fall back to another inference server for this model.
        return JSONResponse(
            content={"error": "Server is busy with another request", "busy": True},
            status_code=429,
        )

    # ---- streaming body: holds the model lock for its whole lifetime ----
    async def _ndjson_stream() -> Iterator[str]:
        try:
            for message in _build_messages(
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
            yield json.dumps({"kind": "error", "error": str(exc)}) + "\n"
        finally:
            request_lock.release()

    if request.stream:
        return StreamingResponse(
            _ndjson_stream(), media_type="application/x-ndjson"
        )

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
