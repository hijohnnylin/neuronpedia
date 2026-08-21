"""HTTP smoke coverage for endpoints that previously had no test.

Runs against the shared gpt2-small + res-jb ``client`` fixture. Assertions are structural
(status + response shape) rather than golden values, since these endpoints mostly had zero
coverage and the goal is to lock in their request/response contracts.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import (
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    SAE_SOURCE_SET,
    TEST_PROMPT,
    X_SECRET_KEY,
)

HEADERS = {"X-SECRET-KEY": X_SECRET_KEY}
SOURCE = SAE_SELECTED_SOURCES[0]


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_capabilities(client: TestClient):
    resp = client.get("/v1/capabilities", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # gpt2-small on the EagerModel backend (the session fixture forces CPU/eager).
    assert data["backend"] == "eager"
    assert "resid_post" in data["capture_points"]
    assert data["endpoints"]["tokenize"] is True
    assert data["endpoints"]["neurons"] is False


def test_activation_all_batch(client: TestClient):
    resp = client.post(
        "/v1/activation/all-batch",
        json={
            "prompts": [TEST_PROMPT, TEST_PROMPT],
            "model": MODEL_ID,
            "source_set": SAE_SOURCE_SET,
            "selected_sources": SAE_SELECTED_SOURCES,
            "sort_by_token_indexes": [],
            "ignore_bos": True,
            "num_results": 5,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == 2
    # Identical prompts => identical top activations (batch determinism).
    assert results[0] == results[1]


def test_activation_source(client: TestClient):
    resp = client.post(
        "/v1/activation/source",
        json={"prompts": [TEST_PROMPT], "model": MODEL_ID, "source": SOURCE},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, dict)
    # Response carries per-prompt activation content for the requested source.
    assert data


def test_activation_topk_by_token_batch(client: TestClient):
    resp = client.post(
        "/v1/activation/topk-by-token-batch",
        json={
            "prompts": [TEST_PROMPT, "Goodbye, world!"],
            "model": MODEL_ID,
            "source": SOURCE,
            "top_k": 3,
            "ignore_bos": True,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data


def test_lens_prompt_logit_lens(client: TestClient):
    resp = client.post(
        "/v1/lens/prompt",
        json={
            "model": MODEL_ID,
            "type": ["LOGIT_LENS"],
            "prompt": TEST_PROMPT,
            "top_n": 5,
            "num_completion_tokens": 0,
            "stream": False,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["meta"] is not None
    # One read-out per prompt token position.
    assert len(data["tokens"]) > 0


def test_similarity_matrix_pred_non_temporal_sae(client: TestClient):
    """res-jb is a non-temporal SAE, so this endpoint must reject it with a clean 400.

    Exercises the route + the temporal-arch guard (rather than needing a temporal SAE).
    """
    resp = client.post(
        "/v1/util/similarity-matrix-pred",
        json={"modelId": MODEL_ID, "sourceId": SOURCE, "index": 0, "text": TEST_PROMPT},
        headers=HEADERS,
    )
    assert resp.status_code == 400, resp.text
    assert "temporal" in resp.json()["error"].lower()
