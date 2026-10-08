"""``/v1/lens/oracle``: the activation at one or more positions, in words, at each layer.

The oracle lens is an adapter trained to describe one residual vector (see
``interp_engine.oracle``). A read captures the base model's ``resid_post`` at the requested
layers for its positions (one forward pass for all of them), then the adapter-on model writes
up to ``max_bullets`` bullets per position and layer. On vLLM all of these run together, and
each frame goes out as its read ends (with ``partial``, also its text per token).

The adapter is chosen at startup (``ORACLE_LENS``), before the model loads, because on vLLM it
decides how the engine is built: an oracle read is an embeds prompt with a LoRA request. Reads
are greedy, so a read depends (up to bf16 noise at near ties) on the token prefix, the layer and
the bullet cap, and a small in-process cache serves repeats.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any

import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from interp_engine import EagerModel, MLXModel, VLLMModel
from interp_engine.oracle import (
    ALL_LAYERS,
    FAST_LAYERS,
    OracleAdapter,
    OraclePartial,
    OracleRead,
    stream_oracle,
)

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.lens.prompt import LensErrorMessage, _acquire_request_lock
from neuronpedia_inference.engine_adapter import BackendUnsupported, assert_residual_available
from neuronpedia_inference.memory_cost import lens_cost
from neuronpedia_inference.schemas import LensOracleRequest, PublicFrameSchema
from neuronpedia_inference.shared import Model, RequestBusy, RequestTooLarge, budget

logger = logging.getLogger(__name__)

router = APIRouter()

#: The adapter ``ORACLE_LENS=auto`` loads for each HF model id.
ORACLE_REGISTRY: dict[str, str] = {
    "Qwen/Qwen3.6-27B": "neuronpedia/jacobian-lens:qwen3.6-27b/oracle",
    "Qwen/Qwen3.5-0.8B": "neuronpedia/jacobian-lens:qwen3.5-0.8b/oracle",
    "Qwen/Qwen3.5-2B": "neuronpedia/jacobian-lens:qwen3.5-2b/oracle",
    "Qwen/Qwen3-4B": "neuronpedia/jacobian-lens:qwen3-4b/oracle",
}

MAX_BULLETS = 5
MAX_TOKENS = 256
# Positions in one request. vLLM holds 64 oracle sequences at once on the 27B A100 pod
# (4 blocks each), so more than ~12 positions x 5 layers only queue.
MAX_POSITIONS = 64
_CACHE_SIZE = 4096


# --------------------------------------------------------------------------- #
# Streamed NDJSON frames (public names; see the note in prompt.py)
# --------------------------------------------------------------------------- #


class OracleMetaMessage(PublicFrameSchema):
    """First frame: what is being read. ``position`` and ``token`` are the first of ``positions``."""

    kind: str = "meta"
    model: str
    adapter: str
    position: int
    token: str
    positions: list[int]
    tokens: list[str]
    layers: list[int]
    max_bullets: int


class OracleReadMessage(PublicFrameSchema):
    """One layer's read at one position, sent as it ends, in the order the reads end."""

    kind: str = "read"
    position: int
    layer: int
    bullets: list[str]
    text: str
    # "bullets" (the cap), "eos", or "length".
    finish: str
    cached: bool = False


class OraclePartialMessage(PublicFrameSchema):
    """One layer's text so far, while its read runs (``partial`` requests). Its ``read`` comes after."""

    kind: str = "partial"
    position: int
    layer: int
    text: str


class OracleDoneMessage(PublicFrameSchema):
    """Last frame."""

    kind: str = "done"
    elapsed_ms: int
    cached_layers: int


# --------------------------------------------------------------------------- #
# Startup: which adapter, and what it asks of the engine
# --------------------------------------------------------------------------- #


class OracleStore:
    """Process-wide holder for the oracle adapter and its load status."""

    _adapter: OracleAdapter | None = None
    _layers: tuple[int, ...] = ()
    _status: str = "not_loaded"  # one of: not_loaded, loaded, off, error
    _error: str | None = None

    @classmethod
    def set_loaded(cls, adapter: OracleAdapter, layers: tuple[int, ...]) -> None:
        cls._adapter, cls._layers, cls._status, cls._error = adapter, layers, "loaded", None

    @classmethod
    def set_off(cls) -> None:
        cls._adapter, cls._layers, cls._status, cls._error = None, (), "off", None

    @classmethod
    def set_error(cls, error: str) -> None:
        cls._adapter, cls._layers, cls._status, cls._error = None, (), "error", error

    @classmethod
    def get(cls) -> OracleAdapter | None:
        return cls._adapter

    @classmethod
    def layers(cls) -> tuple[int, ...]:
        return cls._layers

    @classmethod
    def status(cls) -> str:
        return cls._status

    @classmethod
    def error(cls) -> str | None:
        return cls._error


def resolve_oracle_ref(setting: str | None, hf_model_id: str) -> tuple[str, str] | None:
    """``ORACLE_LENS`` as ``(repo, subdir)``, or ``None`` when there is no oracle for this model."""
    value = (setting or "auto").strip()
    if value.lower() in ("", "off", "false", "none"):
        return None
    if value.lower() == "auto":
        value = ORACLE_REGISTRY.get(hf_model_id, "")
        if not value:
            return None
    repo, sep, subdir = value.partition(":")
    if not sep or not repo or not subdir:
        raise ValueError(f"ORACLE_LENS must be 'auto', 'off' or '<hf repo>:<subdir>'; got {setting!r}.")
    return repo, subdir


def parse_oracle_layers(setting: str | None, trained: tuple[int, ...]) -> tuple[int, ...]:
    """``ORACLE_LAYERS`` ("all", "fast", or a comma list), limited to the adapter's trained band."""
    value = (setting or "all").strip().lower()
    if value == "all":
        wanted: tuple[int, ...] = trained
    elif value == "fast":
        wanted = tuple(layer for layer in FAST_LAYERS if layer in trained) if trained == ALL_LAYERS else trained[::2]
    else:
        wanted = tuple(int(x) for x in value.split(",") if x.strip())
    off = sorted(set(wanted) - set(trained))
    if off:
        raise ValueError(f"ORACLE_LAYERS {off} are outside the adapter's trained layers {list(trained)}.")
    return wanted


def prepare_oracle(args: Any, hf_model_id: str, engine_backend: str) -> dict[str, Any]:
    """Load the oracle adapter before the model, and return what the engine must be built with.

    Best-effort, like the Jacobian lens: a failure is logged and recorded, and the server starts
    without the oracle.
    """
    try:
        ref = resolve_oracle_ref(getattr(args, "oracle_lens", None), hf_model_id)
        if ref is None or engine_backend == "vllm-generate":
            OracleStore.set_off()
            return {}
        adapter = OracleAdapter.load(*ref)
        layers = parse_oracle_layers(getattr(args, "oracle_layers", None), adapter.contract.layers)
        kwargs: dict[str, Any] = {}
        if engine_backend.startswith("vllm"):
            adapter.vllm_dir(hf_model_id)
            kwargs = {"enable_prompt_embeds": True, "max_lora_rank": adapter.rank}
        OracleStore.set_loaded(adapter, layers)
        logger.info("Oracle lens %s ready for layers %s (engine kwargs %s)", adapter.name, list(layers), kwargs)
        return kwargs
    except Exception as exc:  # noqa: BLE001
        logger.exception("Oracle lens failed to load; serving without it")
        OracleStore.set_error(str(exc))
        return {}


# --------------------------------------------------------------------------- #
# Read cache
# --------------------------------------------------------------------------- #


class _ReadCache:
    """LRU of finished reads, keyed by what a greedy read depends on."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._reads: OrderedDict[tuple, OracleRead] = OrderedDict()

    @staticmethod
    def key(adapter: str, prefix: list[int], layer: int, max_bullets: int, max_tokens: int) -> tuple:
        digest = hashlib.sha1(json.dumps(prefix).encode()).hexdigest()
        return (adapter, digest, len(prefix), layer, max_bullets, max_tokens)

    def get(self, key: tuple) -> OracleRead | None:
        read = self._reads.get(key)
        if read is not None:
            self._reads.move_to_end(key)
        return read

    def put(self, key: tuple, read: OracleRead) -> None:
        self._reads[key] = read
        self._reads.move_to_end(key)
        while len(self._reads) > self.size:
            self._reads.popitem(last=False)


READ_CACHE = _ReadCache(_CACHE_SIZE)


def _frame(position: int, read: OracleRead, cached: bool) -> OracleReadMessage:
    return OracleReadMessage(
        position=position, layer=read.layer, bullets=read.bullets, text=read.text, finish=read.finish, cached=cached
    )


OracleItem = OracleRead | OraclePartial


async def _each_position(
    streams: dict[int, AsyncGenerator[OracleItem]], together: bool
) -> AsyncGenerator[tuple[int, OracleItem]]:
    """Each position's reads, tagged with the position.

    ``together`` runs every stream at once, so vLLM batches all their sequences; otherwise the
    streams run one after another (the eager and MLX reads block the event loop anyway).
    """
    if not together:
        for position, stream in streams.items():
            async with aclosing(stream) as it:
                async for item in it:
                    yield position, item
        return
    done = object()
    queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()

    async def pump(position: int, stream: AsyncGenerator[OracleItem]) -> None:
        try:
            async with aclosing(stream) as it:
                async for item in it:
                    queue.put_nowait((position, item))
            queue.put_nowait((position, done))
        except Exception as exc:  # noqa: BLE001 - raised again by the loop below
            queue.put_nowait((position, exc))

    tasks = [asyncio.ensure_future(pump(p, s)) for p, s in streams.items()]
    try:
        left = len(tasks)
        while left:
            position, item = await queue.get()
            if item is done:
                left -= 1
            elif isinstance(item, BaseException):
                raise item
            else:
                yield position, item
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


@router.post("/lens/oracle")
async def lens_oracle(request: LensOracleRequest, http_request: Request):
    config = Config.get_instance()
    model = Model.get_instance()
    adapter = OracleStore.get()
    if adapter is None:
        return JSONResponse(
            content={
                "error": "The oracle lens is not available for this model",
                "status": OracleStore.status(),
                "detail": OracleStore.error(),
            },
            status_code=404,
        )
    if not isinstance(model, EagerModel | VLLMModel | MLXModel):
        return JSONResponse(content={"error": "The oracle lens needs the eager, vLLM or MLX backend."}, status_code=404)
    try:
        assert_residual_available(model, "The oracle lens", point="resid_post")
    except BackendUnsupported as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    token_ids = [int(t) for t in request.token_ids]
    if not token_ids:
        return JSONResponse(content={"error": "token_ids is empty"}, status_code=400)
    positions = list(dict.fromkeys([int(request.position), *(int(p) for p in request.positions)]))
    if len(positions) > MAX_POSITIONS:
        return JSONResponse(
            content={"error": f"{len(positions)} positions; the maximum is {MAX_POSITIONS}"}, status_code=400
        )
    off = [p for p in positions if not 0 <= p < len(token_ids)]
    if off:
        return JSONResponse(
            content={"error": f"positions {off} are outside the {len(token_ids)} token ids"},
            status_code=400,
        )
    if len(token_ids) > config.lens_token_limit:
        return JSONResponse(
            content={
                "error": (
                    f"This conversation is too long ({len(token_ids)} tokens). "
                    f"The maximum is {config.lens_token_limit} tokens."
                )
            },
            status_code=400,
        )
    if not 1 <= request.max_bullets <= MAX_BULLETS:
        return JSONResponse(content={"error": f"max_bullets must be 1..{MAX_BULLETS}"}, status_code=400)
    if not 1 <= request.max_tokens <= MAX_TOKENS:
        return JSONResponse(content={"error": f"max_tokens must be 1..{MAX_TOKENS}"}, status_code=400)
    try:
        layers = adapter.contract.check_layers(request.layers or OracleStore.layers())
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    layers = tuple(dict.fromkeys(layers))

    keys = {
        (p, layer): READ_CACHE.key(adapter.name, token_ids[: p + 1], layer, request.max_bullets, request.max_tokens)
        for p in positions
        for layer in layers
    }
    hits = {pl: read for pl, key in keys.items() if (read := READ_CACHE.get(key)) is not None}
    # Per position, the layers still to read.
    missing: dict[int, list[int]] = {}
    for p, layer in keys:
        if (p, layer) not in hits:
            missing.setdefault(p, []).append(layer)
    capture_ids = token_ids[: max(missing) + 1] if missing else []
    capture_layers = sorted({layer for p_layers in missing.values() for layer in p_layers})

    acquired = None
    budget_claim = None
    if missing:
        try:
            acquired = await _acquire_request_lock(fail_if_busy=request.fail_if_busy)
        except TimeoutError:
            return JSONResponse(content={"error": "Request timed out waiting for lock"}, status_code=503)
        if acquired is None:
            return JSONResponse(content={"error": "Server is busy with another request", "busy": True}, status_code=429)
        cost = lens_cost(
            staged_positions=len(missing),
            layer_counts=[len(capture_layers)],
            d_model=int(getattr(model, "d_model", 0)),
            capture_positions=len(capture_ids),
            n_capture_points=len(capture_layers),
        )
        try:
            budget_claim = await budget.acquire(cost, fail_if_busy=request.fail_if_busy)
        except RequestTooLarge as exc:
            acquired.release()
            return JSONResponse(content={"error": str(exc)}, status_code=400)
        except RequestBusy:
            acquired.release()
            return JSONResponse(
                content={"error": "Server has no free memory for another request", "busy": True}, status_code=429
            )
        except TimeoutError:
            acquired.release()
            return JSONResponse(content={"error": "Request timed out waiting for available memory"}, status_code=503)

    async def _ndjson() -> AsyncIterator[str]:
        start = time.monotonic()
        try:
            tokens = [model.tokenizer.decode([token_ids[p]]) if model.tokenizer is not None else "" for p in positions]
            meta = OracleMetaMessage(
                model=request.model,
                adapter=adapter.name,
                position=positions[0],
                token=tokens[0],
                positions=positions,
                tokens=tokens,
                layers=list(layers),
                max_bullets=request.max_bullets,
            )
            yield json.dumps(meta.model_dump(mode="json")) + "\n"
            for (p, _layer), read in hits.items():
                yield json.dumps(_frame(p, read, True).model_dump(mode="json")) + "\n"
            if missing:
                rows = list(missing)
                captured = await model.capture(
                    capture_ids, [("resid_post", layer) for layer in capture_layers], rows=rows
                )
                acts = {address.layer: picked.float() for address, picked in captured.items()}
                logger.info(
                    "[ORACLE] %d positions x %d layers; capture of %d tokens took %d ms",
                    len(rows),
                    len(capture_layers),
                    len(capture_ids),
                    int((time.monotonic() - start) * 1000),
                )
                streams = {
                    p: stream_oracle(
                        model,
                        adapter,
                        {layer: acts[layer][i] for layer in missing[p]},
                        max_bullets=request.max_bullets,
                        max_tokens=request.max_tokens,
                        partial=request.partial and request.stream,
                    )
                    for i, p in enumerate(rows)
                }
                async with aclosing(_each_position(streams, together=isinstance(model, VLLMModel))) as reads:
                    async for p, read in reads:
                        if isinstance(read, OraclePartial):
                            frame = OraclePartialMessage(position=p, layer=read.layer, text=read.text)
                            yield json.dumps(frame.model_dump(mode="json")) + "\n"
                            continue
                        READ_CACHE.put(keys[(p, read.layer)], read)
                        if await http_request.is_disconnected():
                            logger.info("[ORACLE] Client disconnected; aborting the read.")
                            return
                        yield json.dumps(_frame(p, read, False).model_dump(mode="json")) + "\n"
            done = OracleDoneMessage(elapsed_ms=int((time.monotonic() - start) * 1000), cached_layers=len(hits))
            yield json.dumps(done.model_dump(mode="json")) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error computing an oracle read")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield json.dumps(LensErrorMessage(error=str(exc)).model_dump(mode="json")) + "\n"
        finally:
            if budget_claim is not None:
                await budget.release(budget_claim)
            if acquired is not None:
                acquired.release()

    if request.stream:
        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    meta: dict | None = None
    reads: list[dict] = []
    done: dict | None = None
    error: dict | None = None
    async for line in _ndjson():
        message = json.loads(line)
        kind = message.get("kind")
        if kind == "meta":
            meta = message
        elif kind == "read":
            reads.append(message)
        elif kind == "done":
            done = message
        elif kind == "error":
            error = message
    if error is not None:
        return JSONResponse(content=error, status_code=500)
    return JSONResponse(content={"meta": meta, "reads": reads, "done": done})
