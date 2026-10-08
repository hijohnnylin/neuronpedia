"""The server's responses for a fixed set of requests, against a recorded copy.

This is the gate for moving logic from the server into interp-engine. A move must not change what
a client sees, so each request here is sent to a real server (gpt2, eager, CPU, fp32) and the
response is compared to ``tests/test_data/recorded/gpt2_eager_cpu.json``. Strings, token ids and
shapes must match exactly. Floats must agree to within ``REL`` / ``ABS``, because a move can change
the order of a reduction.

Record again only when a response is meant to change:

    NP_RECORD=1 uv run pytest tests/integration/test_recorded_outputs.py -v
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.harness import EAGER, GPT2, X_SECRET_KEY, initialized_server

RECORDED = Path(__file__).resolve().parents[1] / "test_data" / "recorded" / "gpt2_eager_cpu.json"
RECORD = os.environ.get("NP_RECORD") == "1"
REL = 1e-3
# Lens probabilities are rounded to 4 places, so one step in the last place must pass.
ABS = 2e-4

# Keys whose values depend on the box or on earlier traffic, not on the code under test.
VOLATILE = {"sae_cache", "vram_budget_bytes", "vram_budget_available_bytes", "max_concurrent_requests"}

MODEL = GPT2.model_id
PROMPT = "The capital of France is"
D_MODEL = 768
# Fixed and dense, so a projection or a steer uses every coordinate.
VECTOR = [round(math.sin(0.37 * i) * 0.05, 6) for i in range(D_MODEL)]


def _steer(**overrides: Any) -> dict[str, Any]:
    body = {
        "prompt": PROMPT,
        "model": MODEL,
        "steer_method": "SIMPLE_ADDITIVE",
        "normalize_steering": False,
        "types": ["STEERED", "DEFAULT"],
        "n_completion_tokens": 8,
        "temperature": 0,
        "strength_multiplier": 1.0,
        "freq_penalty": 0.0,
        "seed": 42,
        "stream": False,
    }
    return body | overrides


def _lens(**overrides: Any) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "type": ["LOGIT_LENS"],
        "prompt": PROMPT,
        "top_n": 5,
        "num_completion_tokens": 3,
        "temperature": 0.0,
        "stream": False,
    }
    return body | overrides


CASES: dict[str, tuple[str, str, dict[str, Any] | None]] = {
    "capabilities": ("GET", "/v1/capabilities", None),
    "tokenize": ("POST", "/v1/tokenize", {"model": MODEL, "text": PROMPT, "prepend_bos": True}),
    "activation_single_feature": (
        "POST",
        "/v1/activation/single",
        {"model": MODEL, "prompt": PROMPT, "source": "7-res-jb", "index": "9758"},
    ),
    "activation_single_vector": (
        "POST",
        "/v1/activation/single",
        {"model": MODEL, "prompt": PROMPT, "vector": VECTOR, "hook": "blocks.7.hook_resid_post"},
    ),
    "activation_all": (
        "POST",
        "/v1/activation/all",
        {
            "prompt": PROMPT,
            "model": MODEL,
            "source_set": "res-jb",
            "selected_sources": ["7-res-jb"],
            "sort_by_token_indexes": [],
            "num_results": 5,
            "ignore_bos": True,
        },
    ),
    "lens_logit": ("POST", "/v1/lens/prompt", _lens()),
    "lens_logit_steered": (
        "POST",
        "/v1/lens/prompt",
        _lens(steer_tokens=[{"token": " Paris", "type": "LOGIT_LENS"}], steer_layers=[6], steer_strength=4.0),
    ),
    "lens_jacobian": ("POST", "/v1/lens/prompt", _lens(type=["LOGIT_LENS", "JACOBIAN_LENS"])),
    "lens_jacobian_steered": (
        "POST",
        "/v1/lens/prompt",
        _lens(
            type=["JACOBIAN_LENS"],
            steer_tokens=[{"token": " Paris", "type": "JACOBIAN_LENS"}],
            steer_layers=[6],
            steer_strength=4.0,
        ),
    ),
    "steer_feature": (
        "POST",
        "/v1/steer/completion",
        _steer(features=[{"model": MODEL, "source": "7-res-jb", "index": 5, "strength": 40.0}]),
    ),
    "steer_vector": (
        "POST",
        "/v1/steer/completion",
        _steer(vectors=[{"steering_vector": VECTOR, "strength": 200.0, "hook": "blocks.7.hook_resid_post"}]),
    ),
    "steer_vector_orthogonal": (
        "POST",
        "/v1/steer/completion",
        _steer(
            steer_method="ORTHOGONAL_DECOMP",
            vectors=[{"steering_vector": VECTOR, "strength": 20.0, "hook": "blocks.7.hook_resid_post"}],
        ),
    ),
    "steer_vector_normalized": (
        "POST",
        "/v1/steer/completion",
        _steer(
            normalize_steering=True,
            vectors=[{"steering_vector": VECTOR, "strength": 200.0, "hook": "blocks.7.hook_resid_post"}],
        ),
    ),
}


def _digest(case: tuple[str, str, dict[str, Any] | None]) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _diff(actual: Any, expected: Any, path: str = "$") -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected an object, got {type(actual).__name__}"]
        problems = (
            [f"{path}: key sets differ: {sorted(set(actual) ^ set(expected))}"] if set(actual) != set(expected) else []
        )
        for key in sorted(set(actual) & set(expected)):
            problems += _diff(actual[key], expected[key], f"{path}.{key}")
        return problems
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: expected a list of {len(expected)}, got {actual!r:.120}"]
        return [p for i, (a, e) in enumerate(zip(actual, expected, strict=True)) for p in _diff(a, e, f"{path}[{i}]")]
    if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if math.isclose(actual, expected, rel_tol=REL, abs_tol=ABS):
            return []
        return [f"{path}: {actual} != {expected}"]
    return [] if actual == expected else [f"{path}: {actual!r:.120} != {expected!r:.120}"]


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    spec = replace(GPT2, dtype="float32", include_sae=["7-res-jb"])
    with initialized_server(spec, engine=EAGER, device="cpu") as c:
        yield c


@pytest.fixture(scope="module")
def recorded() -> Iterator[dict[str, Any]]:
    data: dict[str, Any] = {} if RECORD or not RECORDED.exists() else json.loads(RECORDED.read_text())
    yield data
    if RECORD:
        RECORDED.parent.mkdir(parents=True, exist_ok=True)
        RECORDED.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("name", list(CASES))
def test_response_matches_recording(name: str, client: TestClient, recorded: dict[str, Any]):
    method, path, body = CASES[name]
    headers = {"X-SECRET-KEY": X_SECRET_KEY}
    resp = client.get(path, headers=headers) if method == "GET" else client.post(path, json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    actual = _strip(resp.json())

    if RECORD:
        recorded[name] = {"digest": _digest(CASES[name]), "response": actual}
        return
    assert name in recorded, f"no recording for {name}; run with NP_RECORD=1 on a commit before the move"
    assert recorded[name]["digest"] == _digest(CASES[name]), f"{name}: the request changed since it was recorded"
    problems = _diff(actual, recorded[name]["response"])
    assert not problems, f"{name}:\n" + "\n".join(problems[:40])
