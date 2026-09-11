"""Does a streamed frame actually leave the server when it is written?

The lens endpoint computes read-outs one position at a time and yields each as a line of NDJSON,
which is only worth doing if the line reaches the browser then. Starlette's GZipMiddleware never
flushes the compressor, so zlib held roughly a dozen frames at a time and the stream arrived in
clumps -- computed incrementally, delivered in bursts.

This builds on Starlette internals (``GZipResponder``, ``IdentityResponder``), so these tests are
also what catches a Starlette upgrade moving them.

No GPU and no model: this is ASGI plumbing.
"""

from __future__ import annotations

import asyncio
import gzip
from functools import wraps
from typing import Any

import pytest
from starlette.middleware.gzip import GZipMiddleware

from neuronpedia_inference.server import StreamingGZipMiddleware

FRAME = b'{"kind":"token","position":%d,"top_tokens":["the","a","of","Paris"]}\n'


def async_test(func):
    """Run an async test body. pytest-asyncio is not a dependency of this project."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def streaming_app(frames: list[bytes], content_type: bytes = b"application/x-ndjson") -> Any:
    """An app that yields each frame as its own body message, as StreamingResponse does."""

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ARG001
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", content_type)]})
        for frame in frames:
            await send({"type": "http.response.body", "body": frame, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app


async def collect(middleware: Any, accept_gzip: bool = True) -> tuple[dict[bytes, bytes], list[bytes]]:
    """Drive one request; return the response headers and every non-empty body chunk sent."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/lens",
        "raw_path": b"/lens",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"accept-encoding", b"gzip")] if accept_gzip else [],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 80),
    }

    headers: dict[bytes, bytes] = {}
    chunks: list[bytes] = []

    async def send(message: Any) -> None:
        if message["type"] == "http.response.start":
            headers.update(dict(message["headers"]))
        elif message["type"] == "http.response.body" and message.get("body"):
            chunks.append(message["body"])

    async def receive() -> Any:
        return {"type": "http.disconnect"}

    await middleware(scope, receive, send)
    return headers, chunks


def wrap(cls: Any, frames: list[bytes], **kwargs: Any) -> Any:
    return cls(streaming_app(frames), minimum_size=1000, compresslevel=6, **kwargs)


@async_test
async def test_every_frame_reaches_the_client_when_it_is_written():
    frames = [FRAME % i for i in range(50)]
    _, chunks = await collect(wrap(StreamingGZipMiddleware, frames))

    # One non-empty chunk per frame, plus the compressor's own tail.
    assert len(chunks) >= len(frames), f"only {len(chunks)} chunks for {len(frames)} frames"


@async_test
async def test_starlettes_own_middleware_still_swallows_most_of_them():
    """Why this subclass exists. Delete it when Starlette starts flushing."""
    frames = [FRAME % i for i in range(50)]
    _, chunks = await collect(wrap(GZipMiddleware, frames))

    assert len(chunks) < len(frames) / 2, "Starlette appears to flush now; the subclass can go"


@async_test
async def test_the_stream_still_decompresses_to_exactly_what_was_written():
    """A flush that corrupted the stream would be far worse than a slow one."""
    frames = [FRAME % i for i in range(50)]
    headers, chunks = await collect(wrap(StreamingGZipMiddleware, frames))

    assert headers[b"content-encoding"] == b"gzip"
    assert gzip.decompress(b"".join(chunks)) == b"".join(frames)


@async_test
async def test_compression_is_still_worth_doing():
    """Flushing costs block framing, not the LZ77 history: the ratio must survive it."""
    frames = [FRAME % i for i in range(200)]
    _, chunks = await collect(wrap(StreamingGZipMiddleware, frames))

    raw = len(b"".join(frames))
    assert len(b"".join(chunks)) < raw / 3


@async_test
async def test_a_client_that_did_not_ask_for_gzip_gets_none():
    frames = [FRAME % i for i in range(10)]
    headers, chunks = await collect(wrap(StreamingGZipMiddleware, frames), accept_gzip=False)

    assert b"content-encoding" not in headers
    assert b"".join(chunks) == b"".join(frames)


@pytest.mark.parametrize("content_type", [b"text/event-stream", b"text/event-stream; charset=utf-8"])
@async_test
async def test_server_sent_events_are_left_alone(content_type: bytes):
    """Steer streams SSE, which Starlette excludes on purpose. The subclass must not change that."""
    frames = [b"data: %d\n\n" % i for i in range(10)]
    middleware = StreamingGZipMiddleware(streaming_app(frames, content_type), minimum_size=1000, compresslevel=6)
    headers, chunks = await collect(middleware)

    assert b"content-encoding" not in headers
    assert b"".join(chunks) == b"".join(frames)
