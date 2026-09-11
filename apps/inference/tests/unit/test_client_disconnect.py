"""Can an endpoint still tell that the client hung up?

``Request.is_disconnected()`` is the only thing standing between an abandoned lens stream and
a full generation computed for nobody, holding a request slot and a VRAM reservation the whole
time. It is also silently breakable: it reads one message inside an already-cancelled scope,
and Starlette's ``BaseHTTPMiddleware`` wraps ``receive`` in an anyio task group that cannot
deliver a message under those conditions. One such layer is enough to make it answer False
forever, and ``@app.middleware("http")`` installs one.

So these tests pin both halves: that the probe works through the middleware this server
actually installs, and that it does not through the kind it must not.

No GPU and no model: this is ASGI plumbing.
"""

from __future__ import annotations

import asyncio
import json
from functools import wraps
from typing import Any

import pytest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from neuronpedia_inference.server import CheckModelMiddleware, app


def async_test(func):
    """Run an async test body. pytest-asyncio is not a dependency of this project."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


async def probe_endpoint(request: Request) -> JSONResponse:
    """Read the body the way a route does, then ask whether the client is still there."""
    body = await request.body()
    return JSONResponse({"body": body.decode(), "disconnected": await request.is_disconnected()})


async def probe_app(scope: Any, receive: Any, send: Any) -> None:
    response = await probe_endpoint(Request(scope, receive))
    await response(scope, receive, send)


def make_scope() -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/probe",
        "raw_path": b"/probe",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 80),
    }


async def run(asgi_app: Any, body: bytes, *, client_gone: bool) -> dict[str, Any]:
    """Drive one request, then answer further ``receive`` calls the way uvicorn would.

    A live connection with nothing left to send blocks; a dead one returns ``http.disconnect``
    immediately, without a checkpoint. That difference is the entire mechanism under test.
    """
    pending = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> Any:
        if pending:
            return pending.pop(0)
        if client_gone:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    chunks: list[bytes] = []

    async def send(message: Any) -> None:
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await asgi_app(make_scope(), receive, send)
    return json.loads(b"".join(chunks))


@async_test
async def test_the_probe_sees_a_client_that_left():
    result = await run(CheckModelMiddleware(probe_app), b'{"prompt": "hi"}', client_gone=True)
    assert result["disconnected"] is True


@async_test
async def test_the_probe_does_not_cry_wolf_on_a_live_client():
    result = await run(CheckModelMiddleware(probe_app), b'{"prompt": "hi"}', client_gone=False)
    assert result["disconnected"] is False


@async_test
async def test_the_body_still_reaches_the_endpoint_intact():
    """The middleware reads the body to inspect it, so it has to replay what it took."""
    result = await run(CheckModelMiddleware(probe_app), b'{"prompt": "hi"}', client_gone=False)
    assert result["body"] == '{"prompt": "hi"}'


@async_test
async def test_one_base_http_middleware_layer_is_enough_to_break_the_probe():
    """Why the rule exists. Delete this only when Starlette makes it untrue."""

    async def passthrough(request, call_next):  # type: ignore[no-untyped-def]
        return await call_next(request)

    wrapped = BaseHTTPMiddleware(probe_app, dispatch=passthrough)
    result = await run(wrapped, b'{"prompt": "hi"}', client_gone=True)
    assert result["disconnected"] is False


def test_the_server_installs_no_base_http_middleware():
    installed = [entry.cls for entry in app.user_middleware]
    offenders = [cls for cls in installed if isinstance(cls, type) and issubclass(cls, BaseHTTPMiddleware)]
    assert offenders == [], (
        f"{offenders} breaks Request.is_disconnected() for every endpoint below it. "
        "Write the middleware as a pure ASGI class instead of @app.middleware('http')."
    )


@pytest.mark.parametrize("body", [b"", b"not json", b'"a string"'])
@async_test
async def test_an_unparseable_body_is_passed_through_untouched(body: bytes):
    """The model check is advisory, so nothing about it may reject or corrupt a request."""
    result = await run(CheckModelMiddleware(probe_app), body, client_gone=False)
    assert result["body"] == body.decode()
