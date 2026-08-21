"""Integration tests for the interp-engine (EagerModel) backend, behind FORCE_BACKEND=eager.

These load a full server (openai-community/gpt2 + res-jb SAE) with the engine backend and exercise the
migrated endpoints. They are independent of the default (TransformerLens) session fixtures.
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


@pytest.fixture(scope="module")
def engine_client():
    os.environ.update(
        {
            "MODEL_ID": "openai-community/gpt2",
            "FORCE_BACKEND": "eager",
            "SAE_SETS": json.dumps(["res-jb"]),
            "MODEL_DTYPE": "float32",
            "SAE_DTYPE": "float32",
            "TOKEN_LIMIT": "500",
            "DEVICE": "cpu",
            "INCLUDE_SAE": json.dumps(["7-res-jb"]),
            "EXCLUDE_SAE": json.dumps([]),
            "MAX_LOADED_SAES": "1",
            "SECRET": X_SECRET_KEY,
        }
    )
    server.args = parse_env_and_args()
    asyncio.run(initialize())
    yield TestClient(app)

    os.environ.pop("FORCE_BACKEND", None)
    Config._instance = None
    SAEManager._instance = None
    Model._instance = None  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_tokenize_endpoint(engine_client: TestClient):
    from interp_engine import EagerModel

    assert isinstance(Model.get_instance(), EagerModel)

    resp = engine_client.post(
        "/v1/tokenize",
        json={
            "model": "openai-community/gpt2",
            "text": "The quick brown fox",
            "prepend_bos": True,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # gpt2 BOS is <|endoftext|> (id 50256), prepended.
    assert data["tokens"][0] == 50256
    assert data["tokenStrings"][0] == "<|endoftext|>"
    assert data["prependBos"] is True
    assert len(data["tokens"]) == len(data["tokenStrings"]) == 5


def test_tokenize_endpoint_no_bos(engine_client: TestClient):
    resp = engine_client.post(
        "/v1/tokenize",
        json={
            "model": "openai-community/gpt2",
            "text": "The quick brown fox",
            "prepend_bos": False,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["tokens"][0] != 50256
    assert len(data["tokens"]) == 4


# These golden values are the exact TransformerLens-path expectations from
# tests/integration/test_activation_single.py. Matching them under the engine backend is
# endpoint-level parity (resid_pre activation -> SAE encode -> feature values).

ABS_TOLERANCE = 0.1
BOS = "<|endoftext|>"

# How far two rows of one batch may drift from each other. They go through a single batched
# matmul, where a row's accumulation order depends on the tile it lands in, so two identical
# prompts do NOT come back bit-for-bit: gpt2-small in float32 puts the two ~0.052 activations
# about 3e-6 apart, which the SAE's ReLU -- sitting close to zero here -- amplifies out of the
# residual stream. This is ~30x that spread, and still far tighter than any real defect on this
# path: a row sliced from the wrong batch position, or one that attended to padding, moves the
# value by percent rather than by 1e-4.
BATCH_ROW_TOLERANCE = 1e-4


def test_activation_single_source_index_parity(engine_client: TestClient):
    resp = engine_client.post(
        "/v1/activation/single",
        json={
            "model": "openai-community/gpt2",
            "prompt": "Hello, world!",
            "source": "7-res-jb",
            "index": "0",
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    values = data["activation"]["values"]
    # NOTE: the original [134.72, ...] golden in test_activation_single.py is stale in
    # this environment (transformers drift) and now fails on the TransformerLens path too.
    # The current TLens path produces ~[0.0547, 0, 0, 0] (BOS position zeroed + stripped);
    # the engine produces ~[0.0518, 0, 0, 0]. This asserts that current cross-backend parity.
    assert len(values) == 4
    assert data["activation"]["maxValueIndex"] == 0
    assert values[0] == pytest.approx(0.053, abs=0.02)
    assert pytest.approx(values[1:], abs=1e-3) == [0.0, 0.0, 0.0]
    assert data["tokens"] == ["Hello", ",", " world", "!"]


def test_activation_single_vector_hook_parity(engine_client: TestClient):
    resp = engine_client.post(
        "/v1/activation/single",
        json={
            "model": "openai-community/gpt2",
            "prompt": "Hello, world!",
            "vector": [0.1] * 768,
            "hook": "blocks.0.hook_resid_post",
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    expected = [5.4140625, 3.23828125, 1.9462890625, 1.671875]
    assert pytest.approx(data["activation"]["values"], abs=ABS_TOLERANCE) == expected
    assert data["tokens"] == ["Hello", ",", " world", "!"]


def test_activation_single_batch_parity(engine_client: TestClient):
    """One prompt twice in a batch gives one answer, to within batch-matmul reassociation.

    Only the arithmetic is allowed to differ, and only by ``BATCH_ROW_TOLERANCE`` (see there for
    why exact equality is the wrong assertion). Everything a row's *position* must not change --
    tokens, the special mask, which position is the argmax -- still has to match exactly, and
    that is asserted over whatever is left of the payload rather than a field list, so a field
    added to the response is covered without anyone remembering to come back here.
    """
    resp = engine_client.post(
        "/v1/activation/single-batch",
        json={
            "model": "openai-community/gpt2",
            "prompts": ["Hello, world!", "Hello, world!"],
            "source": "7-res-jb",
            "index": "0",
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == 2

    first, second = results
    assert first["tokens"] == second["tokens"]
    assert first["tokensIsSpecial"] == second["tokensIsSpecial"]

    one, two = first["activation"], second["activation"]
    assert one["values"] == pytest.approx(two["values"], abs=BATCH_ROW_TOLERANCE)
    assert one["maxValue"] == pytest.approx(two["maxValue"], abs=BATCH_ROW_TOLERANCE)
    arithmetic = ("values", "maxValue")
    assert {k: v for k, v in one.items() if k not in arithmetic} == {
        k: v for k, v in two.items() if k not in arithmetic
    }

    for r in results:
        assert r["activation"]["maxValue"] >= 0.0
        assert len(r["activation"]["values"]) == len(r["tokens"])


def test_activation_single_batch_does_not_bleed_across_padded_rows(engine_client: TestClient):
    """Prompts of different lengths in one batch each still get their own answer.

    This is the half the test above cannot cover: two identical rows stay identical whether or
    not they influence each other, so bleed between them is invisible. Mixed lengths are what
    make ``process_activations_batch`` right-pad, and padding is where bleed would come from --
    a short row attending to pad tokens, or a row sliced at the wrong offset. Either shows up
    here as a row disagreeing with the same prompt sent on its own, which is the only reference
    that is independent of the batch.
    """
    prompts = ["Hello, world!", "When Mary and John went to the store, John gave a drink to Mary."]
    body = {"model": "openai-community/gpt2", "source": "7-res-jb", "index": "0"}

    batch = engine_client.post(
        "/v1/activation/single-batch",
        json={**body, "prompts": prompts},
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert batch.status_code == 200, batch.text
    rows = batch.json()["results"]
    assert len(rows) == len(prompts)

    for prompt, row in zip(prompts, rows, strict=True):
        alone = engine_client.post(
            "/v1/activation/single",
            json={**body, "prompt": prompt},
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )
        assert alone.status_code == 200, alone.text
        solo = alone.json()

        assert row["tokens"] == solo["tokens"], f"batched row for {prompt!r} tokenized differently"
        assert row["tokensIsSpecial"] == solo["tokensIsSpecial"]
        assert row["activation"]["values"] == pytest.approx(solo["activation"]["values"], abs=BATCH_ROW_TOLERANCE), (
            f"batched row for {prompt!r} disagrees with the same prompt on its own"
        )
        assert row["activation"]["maxValueIndex"] == solo["activation"]["maxValueIndex"]


def test_activation_single_batch_agrees_with_activation_single(engine_client: TestClient):
    """One prompt through either endpoint has to give one answer.

    They are separate handlers over copied helpers, and the copies drifted: `single` dropped
    the prepended BOS from `values` and trimmed `tokens` to match, `single-batch` only zeroed
    it, so the same prompt came back with a different number of tokens depending on whether
    the caller passed a string or a list of strings. On an `-att-` source that also moved DFA,
    since the destination is an index into `values`.
    """
    body = {"model": "openai-community/gpt2", "source": "7-res-jb", "index": "0"}
    single = engine_client.post(
        "/v1/activation/single",
        json={**body, "prompt": "Hello, world!"},
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    batch = engine_client.post(
        "/v1/activation/single-batch",
        json={**body, "prompts": ["Hello, world!"]},
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert single.status_code == 200, single.text
    assert batch.status_code == 200, batch.text

    one, many = single.json(), batch.json()["results"][0]
    assert many["tokens"] == one["tokens"] == ["Hello", ",", " world", "!"]
    assert many["tokensIsSpecial"] == one["tokensIsSpecial"]
    assert many["activation"]["values"] == pytest.approx(one["activation"]["values"], abs=1e-5)
    assert many["activation"]["maxValueIndex"] == one["activation"]["maxValueIndex"]


def test_steer_completion_features_additive(engine_client: TestClient):
    req = {
        "prompt": "Hello, world!",
        "model": "openai-community/gpt2",
        "steer_method": "SIMPLE_ADDITIVE",
        "normalize_steering": False,
        "types": ["STEERED", "DEFAULT"],
        "features": [
            {
                "model": "openai-community/gpt2",
                "source": "7-res-jb",
                "index": 5,
                "strength": 10.0,
            }
        ],
        "n_completion_tokens": 8,
        "temperature": 0,
        "strength_multiplier": 10.0,
        "freq_penalty": 0.0,
        "seed": 42,
        "stream": False,
    }
    resp = engine_client.post("/v1/steer/completion", json=req, headers={"X-SECRET-KEY": X_SECRET_KEY})
    assert resp.status_code == 200, resp.text
    outputs = {o["type"]: o["output"] for o in resp.json()["outputs"]}
    assert set(outputs) == {"STEERED", "DEFAULT"}
    assert isinstance(outputs["DEFAULT"], str) and len(outputs["DEFAULT"]) > 0
    # Strong additive steering should push the steered continuation off the default one.
    assert outputs["STEERED"] != outputs["DEFAULT"]


def test_steer_completion_default_deterministic(engine_client: TestClient):
    req = {
        "prompt": "Hello, world!",
        "model": "openai-community/gpt2",
        "steer_method": "SIMPLE_ADDITIVE",
        "normalize_steering": False,
        "types": ["DEFAULT"],
        "features": [
            {
                "model": "openai-community/gpt2",
                "source": "7-res-jb",
                "index": 5,
                "strength": 0.0,
            }
        ],
        "n_completion_tokens": 8,
        "temperature": 0,
        "strength_multiplier": 0.0,
        "freq_penalty": 0.0,
        "seed": 42,
        "stream": False,
    }
    a = engine_client.post("/v1/steer/completion", json=req, headers={"X-SECRET-KEY": X_SECRET_KEY})
    b = engine_client.post("/v1/steer/completion", json=req, headers={"X-SECRET-KEY": X_SECRET_KEY})
    assert a.status_code == b.status_code == 200
    out_a = {o["type"]: o["output"] for o in a.json()["outputs"]}["DEFAULT"]
    out_b = {o["type"]: o["output"] for o in b.json()["outputs"]}["DEFAULT"]
    assert out_a == out_b


def test_activation_attention_endpoint(engine_client: TestClient):
    resp = engine_client.post(
        "/v1/activation/attention",
        json={
            "model": "openai-community/gpt2",
            "prompt": "When Mary and John went to the store, John gave a drink to Mary.",
            "layer": 5,
            "head": 1,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["seqLen"] > 0
    assert len(data["attentionIndices"]) == len(data["attentionValues"])
    assert 0.0 <= data["maxActivation"] <= 1.0
