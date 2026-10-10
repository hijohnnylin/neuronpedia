"""A /health during startup must not replace the config that startup builds.

Startup builds the real Config in a worker thread, which takes seconds. A request in that
window that calls ``Config.get_instance()`` builds a default (gpt2) config, and if the default
is stored last it replaces the real one: the pod then serves with ``device=None`` and no layer
count, and every lens request returns 500 while /health still passes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from starlette.responses import JSONResponse

from neuronpedia_inference import server
from neuronpedia_inference.config import Config


@pytest.fixture
def no_config():
    previous = Config._instance
    Config._instance = None
    yield
    Config._instance = previous


@pytest.mark.usefixtures("no_config")
def test_a_default_does_not_replace_a_config_stored_while_it_was_built(monkeypatch: pytest.MonkeyPatch):
    real = object.__new__(Config)

    def startup_stores_its_config_meanwhile(self: Config, *_: Any) -> None:
        Config._instance = real

    monkeypatch.setattr(Config, "__post_init__", startup_stores_its_config_meanwhile)

    assert Config.get_instance() is real


@pytest.mark.usefixtures("no_config")
def test_health_during_startup_builds_no_config_and_probes_nothing(monkeypatch: pytest.MonkeyPatch):
    probed: list[Any] = []
    monkeypatch.setattr(server, "probe_cuda_or_die", probed.append)
    monkeypatch.setattr(server, "initialized", False)

    async def starting(scope: Any, receive: Any, send: Any) -> None:
        await JSONResponse(status_code=503, content={"status": "starting"})(scope, receive, send)

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "/health", "headers": []}
    asyncio.run(server.CudaHealthMiddleware(starting)(scope, receive, send))

    assert sent[0]["status"] == 503
    assert Config._instance is None
    assert probed == []
