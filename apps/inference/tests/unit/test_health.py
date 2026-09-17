"""The ``/health`` probe, against stand-in models.

What is under test is the verdict, not the model: which backend states come back ``ok``,
``starting`` or ``unhealthy``, and that callers arriving mid-probe share one forward pass.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from interp_engine import EagerModel, VLLMModel

from neuronpedia_inference import health
from neuronpedia_inference.config import Config


@pytest.fixture(autouse=True)
def _fresh_probe():
    health._inflight = None
    yield
    health._inflight = None


def _tokenizer() -> SimpleNamespace:
    return SimpleNamespace(bos_token_id=1, encode=lambda *_args, **_kwargs: [1])


def _eager(logits: torch.Tensor) -> MagicMock:
    model = MagicMock(spec=EagerModel)
    model.tokenizer = _tokenizer()
    model.device = "cpu"
    model.hf_model = MagicMock(return_value=SimpleNamespace(logits=logits))
    return model


def _vllm(engine: object, generate_text: AsyncMock) -> MagicMock:
    model = MagicMock(spec=VLLMModel)
    model.tokenizer = _tokenizer()
    model.engine = engine
    model.generate_text = generate_text
    return model


def _probe(model: MagicMock) -> health.HealthResponse:
    return asyncio.run(health.probe(model, Config(device="cpu")))


def test_eager_forward_pass_is_ok():
    response = _probe(_eager(torch.zeros(1, 1, 4)))
    assert response.status == "ok"
    assert response.backend == "eager"
    assert response.probe_ms is not None
    assert response.error is None


def test_eager_non_finite_logits_are_unhealthy():
    response = _probe(_eager(torch.full((1, 1, 4), float("nan"))))
    assert response.status == "unhealthy"
    assert response.error is not None
    assert "non-finite" in response.error


def test_eager_forward_exception_is_unhealthy():
    model = _eager(torch.zeros(1, 1, 4))
    model.hf_model.side_effect = RuntimeError("CUDA error: an illegal memory access was encountered")
    response = _probe(model)
    assert response.status == "unhealthy"
    assert response.error is not None
    assert "illegal memory access" in response.error


def test_vllm_without_an_engine_is_starting():
    response = _probe(_vllm(None, AsyncMock()))
    assert response.status == "starting"
    assert response.backend == "vllm"


def test_vllm_dead_engine_is_unhealthy():
    generate = AsyncMock(return_value="x")
    response = _probe(_vllm(SimpleNamespace(errored=True), generate))
    assert response.status == "unhealthy"
    assert response.error is not None
    assert "dead" in response.error
    generate.assert_not_called()


def test_vllm_one_token_generation_is_ok():
    generate = AsyncMock(return_value="x")
    response = _probe(_vllm(SimpleNamespace(errored=False), generate))
    assert response.status == "ok"
    generate.assert_awaited_once_with([1], max_tokens=1, temperature=0.0)


def test_vllm_hung_engine_times_out(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(health, "PROBE_TIMEOUT_SECONDS", 0.05)

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(10)

    response = _probe(_vllm(SimpleNamespace(errored=False), AsyncMock(side_effect=hang)))
    assert response.status == "unhealthy"
    assert response.error is not None
    assert "did not finish" in response.error


def test_concurrent_callers_share_one_forward_pass():
    calls = 0

    async def generate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "x"

    model = _vllm(SimpleNamespace(errored=False), AsyncMock(side_effect=generate))
    config = Config(device="cuda")

    async def two_at_once():
        return await asyncio.gather(health.probe(model, config), health.probe(model, config))

    first, second = asyncio.run(two_at_once())
    assert calls == 1
    assert first.status == second.status == "ok"
