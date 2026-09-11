"""A streaming steer completion holds a request slot for the stream's lifetime.

``with_request_lock`` admits the handler and releases its slot when the handler returns, which for
a streaming request is before a single token is generated: Starlette iterates a
``StreamingResponse`` body only after the handler has returned. The SSE generators therefore take a
slot of their own (``stream_lock``) and hold it until the stream is exhausted or closed. Nothing
else pins that ordering, and it breaks silently in both directions: an admission slot held across
the stream deadlocks the generator's own acquire on the single mutex a non-vLLM pod runs with, and
a slot the generator releases only at garbage collection lets an abandoned stream hold up the next
request for as long as that takes.

Driven with a stub backend: no model, no GPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from functools import wraps
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from neuronpedia_inference import shared
from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.steer import completion as completion_module
from neuronpedia_inference.endpoints.steer import completion_chat as completion_chat_module
from neuronpedia_inference.endpoints.steer.completion import _completion_frame, completion
from neuronpedia_inference.inference_utils import steering
from neuronpedia_inference.inference_utils.steering import SteeringSettings
from neuronpedia_inference.schemas import (
    NPSteerChatMessage,
    NPSteerMethod,
    NPSteerType,
    NPSteerVector,
    SteerCompletionRequest,
)
from neuronpedia_inference.shared import ConcurrencyLimiter, Model, VramBudget

FRAME_TIMEOUT_S = 5.0
STEER_TYPES = [NPSteerType.STEERED]
VECTORS = [NPSteerVector(steering_vector=[0.0] * 8, strength=1.0, hook="blocks.0.hook_resid_post")]
FRAMES = [_completion_frame(STEER_TYPES, {NPSteerType.STEERED: text}) for text in ("hi", "hi there")]


def async_test(func):  # type: ignore[no-untyped-def]
    """Run an async test body. pytest-asyncio is not a dependency of this project."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class StubEagerModel:
    """What the steer endpoints ask of a hooked eager backend before generating, and nothing more."""

    hooks_available = True  # a hooked pod: steering needs no declared write sites
    tokenizer = SimpleNamespace(bos_token="<bos>")
    tok = SimpleNamespace(tokenizer_prepends_bos=True)

    def to_tokens(self, prompt: str, **kwargs: Any) -> torch.Tensor:
        return torch.tensor([[1, 2, 3]])


def _engine_frames(**kwargs: Any) -> Iterator[str]:
    """In place of ``_engine_run_batched_generate``: the eager backend's sync SSE generator."""
    yield from FRAMES


async def _engine_chat_frames(**kwargs: Any) -> AsyncGenerator[str, None]:
    """In place of ``_engine_chat_generate``: the chat endpoint's async SSE generator."""
    for frame in FRAMES:
        yield frame


@pytest.fixture
def limiter(monkeypatch: pytest.MonkeyPatch) -> ConcurrencyLimiter:
    """A fresh single-mutex limiter (the non-vLLM default) in place of the process-wide one.

    The decorator reads ``limiter`` from ``shared`` and ``stream_lock`` from its own import in
    ``steering``, so both names are swapped. The budget is replaced by a disabled one: nothing to
    ration here.
    """
    fresh = ConcurrencyLimiter()
    monkeypatch.setattr(shared, "limiter", fresh)
    monkeypatch.setattr(steering, "limiter", fresh)
    monkeypatch.setattr(shared, "budget", VramBudget())
    return fresh


@pytest.fixture
def stub_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A stub eager backend whose generation replays ``FRAMES``, for both steer endpoints."""
    for module in (completion_module, completion_chat_module):
        monkeypatch.setattr(module, "EagerModel", StubEagerModel)
    monkeypatch.setattr(completion_module, "_engine_run_batched_generate", _engine_frames)
    monkeypatch.setattr(completion_chat_module, "_engine_chat_generate", _engine_chat_frames)

    had_model = hasattr(Model, "_instance")
    previous_model = getattr(Model, "_instance", None)
    previous_config = Config._instance
    Model.set_instance(StubEagerModel())  # type: ignore[arg-type]
    # The real Config would build the SAE directory, which this has no use for.
    with patch.object(Config, "_generate_sae_config", return_value=[]):
        Config._instance = Config(token_limit=100)
    try:
        yield
    finally:
        Config._instance = previous_config
        if had_model:
            Model.set_instance(previous_model)  # type: ignore[arg-type]
        else:
            del Model._instance


def _settings() -> SteeringSettings:
    return SteeringSettings(features=VECTORS, strength_multiplier=1.0)


def _request(*, stream: bool) -> SteerCompletionRequest:
    return SteerCompletionRequest(
        prompt="<bos>hello",
        model="stub",
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=STEER_TYPES,
        vectors=VECTORS,
        n_completion_tokens=2,
        temperature=1.0,
        strength_multiplier=1.0,
        freq_penalty=0.0,
        seed=7,
        stream=stream,
    )


def _live_http_request() -> Request:
    """A client that stays connected: ``receive`` blocks, so ``is_disconnected()`` answers False."""

    async def receive() -> Any:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return Request({"type": "http", "method": "POST", "path": "/steer/completion", "headers": []}, receive)


async def _next_frame(frames: AsyncIterator[str]) -> str:
    """The next frame, or a failure rather than a hang if generation is waiting on a slot it cannot get."""
    try:
        return await asyncio.wait_for(anext(frames), timeout=FRAME_TIMEOUT_S)
    except TimeoutError:
        pytest.fail(f"no frame within {FRAME_TIMEOUT_S}s: generation is deadlocked on a request slot")


def _completion_stream() -> AsyncGenerator[str, None]:
    return completion_module.run_batched_generate(
        prompt="<bos>hello",
        settings=_settings(),
        steer_types=STEER_TYPES,
        seed=7,
        use_stream_lock=True,
        max_new_tokens=2,
        temperature=1.0,
    )


def _completion_chat_stream() -> AsyncGenerator[str, None]:
    return completion_chat_module.run_batched_generate(
        promptTokenized=torch.tensor([1, 2, 3]),
        inputPrompt=[NPSteerChatMessage(role="user", content="hello")],
        settings=_settings(),
        steer_types=STEER_TYPES,
        seed=7,
        use_stream_lock=True,
        max_new_tokens=2,
        temperature=1.0,
    )


STREAMS = {"completion": _completion_stream, "completion-chat": _completion_chat_stream}


@pytest.mark.usefixtures("stub_backend")
@async_test
async def test_a_stream_takes_its_own_slot_once_the_handler_has_released_its(limiter: ConcurrencyLimiter):
    response = await completion(_request(stream=True), _live_http_request())
    assert isinstance(response, StreamingResponse)
    assert not limiter.is_busy(exclusive=False), "the handler's slot must be released before the stream starts"

    frames = cast(AsyncIterator[str], response.body_iterator)
    first = await _next_frame(frames)
    assert limiter.is_busy(exclusive=False), "the stream must hold a slot while it generates"

    rest = [frame async for frame in frames]
    assert [first, *rest] == FRAMES
    assert not limiter.is_busy(exclusive=False), "the slot must be released once the stream is exhausted"


@pytest.mark.usefixtures("stub_backend")
@async_test
async def test_a_non_streaming_request_generates_under_the_handlers_slot(limiter: ConcurrencyLimiter):
    """The handler still holds its admission slot here, so a second slot for the generation would deadlock."""
    response = await asyncio.wait_for(completion(_request(stream=False), _live_http_request()), FRAME_TIMEOUT_S)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert not limiter.is_busy(exclusive=False)


@pytest.mark.parametrize("stream", list(STREAMS.values()), ids=list(STREAMS))
@pytest.mark.usefixtures("stub_backend")
@async_test
async def test_an_exhausted_stream_releases_its_slot(limiter: ConcurrencyLimiter, stream):
    frames = stream()
    await _next_frame(frames)
    assert limiter.is_busy(exclusive=False)

    async for _ in frames:
        pass
    assert not limiter.is_busy(exclusive=False)


@pytest.mark.parametrize("stream", list(STREAMS.values()), ids=list(STREAMS))
@pytest.mark.usefixtures("stub_backend")
@async_test
async def test_a_stream_closed_early_releases_its_slot(limiter: ConcurrencyLimiter, stream):
    """A client that leaves mid-stream must not keep the slot until garbage collection gets to it."""
    frames = stream()
    await _next_frame(frames)
    assert limiter.is_busy(exclusive=False)

    await frames.aclose()
    assert not limiter.is_busy(exclusive=False)
