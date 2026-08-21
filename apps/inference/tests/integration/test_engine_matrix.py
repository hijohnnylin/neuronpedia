"""Cross-engine matrix: run the core endpoints on both backends and assert they agree.

Each test runs once per engine via the ``matrix_client`` parametrized fixture:

- ``eager`` (:class:`EagerModel`) always runs (CPU or CUDA).
- ``vllm`` runs only where the vLLM backend can actually start -- it is marked
  ``cuda``/``vllm`` and self-skips off a CUDA+vLLM box (e.g. CPU CI or a GPU box without a
  CUDA toolkit for FlashInfer's JIT).

The vLLM server here is module-scoped and stays alive until pytest leaves this module, so
anything that needs to start its own engine (e.g. the EagerModel-vs-vLLM parity check in
``test_engine_parity.py``) must live in a different module.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.harness import (
    EAGER,
    GPT2,
    VLLM,
    X_SECRET_KEY,
    try_initialized_server,
)

PROMPT = "Hello, world!"


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(EAGER, id="eager"),
        pytest.param(VLLM, id="vllm", marks=[pytest.mark.cuda, pytest.mark.vllm]),
    ],
)
def matrix_engine(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def matrix_client(matrix_engine: str) -> Iterator[TestClient]:
    with try_initialized_server(GPT2, engine=matrix_engine) as client:
        yield client


def test_tokenize_across_engines(matrix_client: TestClient):
    resp = matrix_client.post(
        "/v1/tokenize",
        json={
            "model": GPT2.model_id,
            "text": "The quick brown fox",
            "prepend_bos": True,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # gpt2 BOS is <|endoftext|> (id 50256), prepended, and tokens decode per-token.
    assert data["tokens"][0] == 50256
    assert data["tokenStrings"][0] == "<|endoftext|>"
    assert len(data["tokens"]) == len(data["tokenStrings"]) == 5


def test_activation_single_across_engines(matrix_client: TestClient):
    resp = matrix_client.post(
        "/v1/activation/single",
        json={
            "model": GPT2.model_id,
            "prompt": PROMPT,
            "source": "7-res-jb",
            "index": "0",
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["tokens"] == ["Hello", ",", " world", "!"]
    assert len(data["activation"]["values"]) == 4
    assert data["activation"]["maxValue"] >= 0.0


def test_activation_all_across_engines(matrix_client: TestClient):
    resp = matrix_client.post(
        "/v1/activation/all",
        json={
            "prompt": PROMPT,
            "model": GPT2.model_id,
            "source_set": "res-jb",
            "selected_sources": ["7-res-jb"],
            "sort_by_token_indexes": [],
            "num_results": 3,
            "ignore_bos": True,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["activations"]) == 3
    assert len(data["tokens"]) >= 4


def test_lens_prompt_across_engines(matrix_client: TestClient):
    resp = matrix_client.post(
        "/v1/lens/prompt",
        json={
            "model": GPT2.model_id,
            "type": ["LOGIT_LENS"],
            "prompt": PROMPT,
            "top_n": 3,
            "num_completion_tokens": 0,
            "stream": False,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["meta"] is not None
    assert len(data["tokens"]) > 0


def _steer_request(*, strength: float, types: list[str]) -> dict:
    return {
        "prompt": PROMPT,
        "model": GPT2.model_id,
        "steer_method": "SIMPLE_ADDITIVE",
        "normalize_steering": False,
        "types": types,
        "features": [{"model": GPT2.model_id, "source": "7-res-jb", "index": 5, "strength": strength}],
        "n_completion_tokens": 8,
        "temperature": 0,
        "strength_multiplier": strength,
        "freq_penalty": 0.0,
        "seed": 42,
        "stream": False,
    }


def test_steer_completion_deterministic_across_engines(matrix_client: TestClient):
    req = _steer_request(strength=0.0, types=["DEFAULT"])
    a = matrix_client.post("/v1/steer/completion", json=req, headers={"X-SECRET-KEY": X_SECRET_KEY})
    b = matrix_client.post("/v1/steer/completion", json=req, headers={"X-SECRET-KEY": X_SECRET_KEY})
    assert a.status_code == b.status_code == 200, a.text
    out_a = {o["type"]: o["output"] for o in a.json()["outputs"]}["DEFAULT"]
    out_b = {o["type"]: o["output"] for o in b.json()["outputs"]}["DEFAULT"]
    assert out_a == out_b
    assert isinstance(out_a, str) and len(out_a) > 0


def test_steering_actually_steers_across_engines(matrix_client: TestClient):
    """The steered completion must differ from the unsteered one, on **both** backends.

    The determinism test above cannot show this and never could: it asks for `DEFAULT` only, at
    strength 0.0, so it compares two identical unsteered completions. Everything it asserts -- equal,
    a string, non-empty -- holds just as well if steering is absent entirely, which is exactly the
    failure a vLLM pod can have. Steering is applied there by worker forward hooks, and a hook that
    does not fire writes nothing and reports nothing; the request comes back fluent, deterministic and
    un-intervened. `test_engine_backend.py:195` makes this assertion already, but only on eager, whose
    hooks run in-process and cannot fail this way -- so the backend that needed covering was the one
    not covered.

    Both types come from ONE response, so they share a server, a seed and a tokenization: a difference
    is the steering and nothing else.
    """
    resp = matrix_client.post(
        "/v1/steer/completion",
        json=_steer_request(strength=10.0, types=["STEERED", "DEFAULT"]),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    outputs = {o["type"]: o["output"] for o in resp.json()["outputs"]}
    assert set(outputs) == {"STEERED", "DEFAULT"}
    assert isinstance(outputs["DEFAULT"], str) and len(outputs["DEFAULT"]) > 0
    assert outputs["STEERED"] != outputs["DEFAULT"], (
        "the steered continuation is identical to the unsteered one, so steering had no effect. "
        "On the vLLM backend the usual cause is an engine whose Python forward hooks never run "
        "(CUDA graphs left on), which is silent by nature -- see VLLMModel.hooks_available."
    )
