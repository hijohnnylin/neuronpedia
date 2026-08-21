"""A full server bring-up with ``SAE_SETS=[]``, end to end.

The unit tests around this cover the pieces in isolation; this one exists because the
failure mode being guarded against is a startup-ordering one -- a derived limit, a memory
measurement or a banner reaching for SAE state that is no longer there -- and that only
shows up when the real ``initialize()`` runs from top to bottom.
"""

import asyncio
import gc
import json
import os

import pytest
import torch
from fastapi.testclient import TestClient

import neuronpedia_inference.server as server
from neuronpedia_inference.args import parse_env_and_args
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.server import app, initialize
from neuronpedia_inference.shared import Model

X_SECRET_KEY = "cat"
TOKEN_LIMIT = 500


@pytest.fixture(scope="module")
def no_sae_client():
    previous = {k: os.environ.get(k) for k in ("MODEL_ID", "SAE_SETS", "MODEL_DTYPE", "DEVICE", "TOKEN_LIMIT")}
    os.environ.update(
        {
            "MODEL_ID": "openai-community/gpt2",
            "SAE_SETS": json.dumps([]),
            "MODEL_DTYPE": "float32",
            "SAE_DTYPE": "float32",
            "TOKEN_LIMIT": str(TOKEN_LIMIT),
            "DEVICE": "cpu",
            "INCLUDE_SAE": json.dumps([]),
            "EXCLUDE_SAE": json.dumps([]),
            "SECRET": X_SECRET_KEY,
        }
    )
    server.args = parse_env_and_args()
    Config._instance = None
    SAEManager._instance = None
    Model._instance = None  # type: ignore[assignment]
    asyncio.run(initialize())
    yield TestClient(app)

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    Config._instance = None
    SAEManager._instance = None
    Model._instance = None  # type: ignore[assignment]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_starts_with_no_saes_loaded(no_sae_client: TestClient):
    assert no_sae_client.get("/health").json() == {"status": "healthy"}
    assert SAEManager.get_instance().get_valid_sae_sets() == []
    assert SAEManager.get_instance().loaded_saes == {}


def test_activation_token_limit_is_not_shrunk_by_absent_saes(no_sae_client: TestClient):
    # The cap is derived from the widest configured SAE. With none, it must stay at the
    # configured token limit rather than collapsing to the memory floor.
    capabilities = no_sae_client.get("/v1/capabilities", headers={"X-SECRET-KEY": X_SECRET_KEY}).json()
    assert capabilities["activation_token_limit"] == TOKEN_LIMIT
    assert capabilities["sae_sets"] == []
    assert capabilities["num_layers"] == 12
    assert capabilities["endpoints"]["activation_raw"] is True


def test_activation_raw_works_without_saes(no_sae_client: TestClient):
    response = no_sae_client.post(
        "/v1/activation/raw",
        json={"model": "openai-community/gpt2", "prompts": ["Hello, world!"], "layers": [0, 11]},
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 200

    activations = response.json()["results"][0]["activations"]
    assert [layer["layer"] for layer in activations] == [0, 11]
    assert len(activations[0]["values"][0]) == 768


def test_sae_backed_endpoints_reject_their_source_set(no_sae_client: TestClient):
    response = no_sae_client.post(
        "/v1/activation/all",
        json={
            "model": "openai-community/gpt2",
            "prompt": "Hello, world!",
            "source_set": "res-jb",
            "selected_sources": ["7-res-jb"],
            "sort_by_token_indexes": [],
            "ignore_bos": True,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 400
    assert "source set" in response.json()["error"].lower()
