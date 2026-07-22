"""Regression tests for issue #38 — request-lock lifecycle of the steer endpoints.

``/steer/completion`` and ``/steer/completion-chat`` acquire the global
``request_lock`` in the handler and release it in the streaming generator's
``finally`` (Starlette iterates a ``StreamingResponse`` body after the handler
returns). These tests drive the real handlers with a minimal in-memory model
stub and pin the lock lifecycle:

* a streaming response must yield its first SSE chunk promptly — any
  re-acquire of ``request_lock`` on the generation path deadlocks here;
* the lock is held while streaming and released when the stream is exhausted
  or closed early (client disconnect);
* a non-streaming response releases the lock before returning.
"""

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.responses import JSONResponse, StreamingResponse
from neuronpedia_inference_client.models.np_steer_chat_message import (
    NPSteerChatMessage,
)
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_vector import NPSteerVector
from neuronpedia_inference_client.models.steer_completion_chat_post_request import (
    SteerCompletionChatPostRequest,
)
from neuronpedia_inference_client.models.steer_completion_request import (
    SteerCompletionRequest,
)
from transformer_lens import HookedTransformer

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.steer.completion import completion
from neuronpedia_inference.endpoints.steer.completion_chat import completion_chat
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, request_lock

CHUNK_TIMEOUT_S = 5.0


class StubTokenizer:
    bos_token = "<bos>"
    bos_token_id = 1
    eos_token_id = 2
    name_or_path = "stub-tokenizer"
    chat_template = "{{ messages }}"

    def encode(self, text: str) -> list[int]:
        return [3, 4]

    def decode(self, tokens: Any) -> str:
        return "hi"

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        return [3, 4, 5]


def make_stub_model() -> MagicMock:
    """A HookedTransformer-shaped stub that streams two generation chunks."""
    model = MagicMock(spec=HookedTransformer)
    model.cfg = SimpleNamespace(device="cpu", tokenizer_prepends_bos=True)
    model.tokenizer = StubTokenizer()
    model.to_tokens.return_value = torch.tensor([[3, 4, 5]])
    model.to_string.return_value = "hi"
    model.generate_stream.side_effect = lambda **_: iter(
        [
            (torch.tensor([[3, 4]]), torch.zeros(1, 2, 8)),
            (torch.tensor([[5]]), torch.zeros(1, 1, 8)),
        ]
    )
    return model


@pytest.fixture()
def steer_singletons():
    saved = (
        Config._instance,
        SAEManager._instance,
        getattr(Model, "_instance", None),
    )
    Config._instance = cast(
        Config,
        SimpleNamespace(
            token_limit=100,
            custom_hf_model_id=None,
            steer_special_token_ids=None,
        ),
    )
    SAEManager._instance = cast(SAEManager, SimpleNamespace())
    Model.set_instance(make_stub_model())
    yield
    Config._instance, SAEManager._instance, Model._instance = saved
    if request_lock.locked():
        request_lock.release()


def _steer_vector() -> NPSteerVector:
    return NPSteerVector(
        steering_vector=[0.0, 0.0],
        strength=1.0,
        hook="blocks.0.hook_resid_post",
    )


def _completion_request(stream: bool) -> SteerCompletionRequest:
    return SteerCompletionRequest(
        prompt="<bos>hello world",
        model="stub-model",
        vectors=[_steer_vector()],
        types=[NPSteerType.STEERED],
        n_completion_tokens=2,
        temperature=1.0,
        freq_penalty=0.0,
        seed=7,
        strength_multiplier=1.0,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        stream=stream,
    )


def _chat_request(stream: bool) -> SteerCompletionChatPostRequest:
    return SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(role="user", content="hello")],
        model="stub-model",
        vectors=[_steer_vector()],
        types=[NPSteerType.STEERED],
        n_completion_tokens=2,
        temperature=1.0,
        freq_penalty=0.0,
        seed=7,
        strength_multiplier=1.0,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        steer_special_tokens=False,
        stream=stream,
    )


async def _next_chunk(iterator: AsyncGenerator[str, None]) -> str:
    """Get the next chunk, failing (not hanging) if generation deadlocks."""
    try:
        return await asyncio.wait_for(anext(iterator), timeout=CHUNK_TIMEOUT_S)
    except TimeoutError:
        pytest.fail(
            f"no SSE chunk within {CHUNK_TIMEOUT_S}s — streaming generation "
            "deadlocked on request_lock (issue #38 regression)"
        )


async def _collect_stream(response: StreamingResponse) -> list[str]:
    iterator = cast("AsyncGenerator[str, None]", response.body_iterator)
    chunks: list[str] = []
    while True:
        try:
            chunk = await _next_chunk(iterator)
        except StopAsyncIteration:
            break
        assert request_lock.locked(), "lock must be held while streaming"
        chunks.append(chunk)
    return chunks


@pytest.mark.usefixtures("steer_singletons")
def test_completion_streaming_yields_and_releases_lock():
    async def scenario():
        response = await completion(_completion_request(stream=True))
        assert isinstance(response, StreamingResponse)
        assert request_lock.locked(), "lock must be held when the handler returns"

        chunks = await _collect_stream(response)

        assert len(chunks) == 2
        assert all(chunk.startswith("data: ") for chunk in chunks)
        assert not request_lock.locked(), "lock must be released after the stream ends"

    asyncio.run(scenario())


@pytest.mark.usefixtures("steer_singletons")
def test_completion_streaming_releases_lock_on_early_close():
    async def scenario():
        response = await completion(_completion_request(stream=True))
        assert isinstance(response, StreamingResponse)
        iterator = cast("AsyncGenerator[str, None]", response.body_iterator)

        await _next_chunk(iterator)
        assert request_lock.locked(), "lock must be held mid-stream"

        # Simulate a client disconnect: Starlette closes the body generator.
        await iterator.aclose()
        assert not request_lock.locked(), "lock must be released on early close"

    asyncio.run(scenario())


@pytest.mark.usefixtures("steer_singletons")
def test_completion_non_streaming_releases_lock():
    async def scenario():
        response = await completion(_completion_request(stream=False))
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        assert not request_lock.locked(), "lock must be released before returning"

    asyncio.run(scenario())


@pytest.mark.usefixtures("steer_singletons")
def test_completion_chat_streaming_yields_and_releases_lock():
    async def scenario():
        response = await completion_chat(_chat_request(stream=True))
        assert isinstance(response, StreamingResponse)
        assert request_lock.locked(), "lock must be held when the handler returns"

        chunks = await _collect_stream(response)

        assert len(chunks) == 2
        assert all(chunk.startswith("data: ") for chunk in chunks)
        assert not request_lock.locked(), "lock must be released after the stream ends"

    asyncio.run(scenario())
