"""Tests for ``/activation/raw`` -- final-token residual stream vectors.

The parity test is the load-bearing one: everything else checks shapes and rejections, but
only a comparison against a direct ``run_with_cache`` establishes that the numbers coming
back are the residual stream and not, say, an off-by-one layer or the wrong token position.
"""

import pytest
import torch
from fastapi.testclient import TestClient
from interp_engine import EagerModel, run_with_cache

from neuronpedia_inference.shared import Model
from tests.conftest import MODEL_ID, TEST_PROMPT, X_SECRET_KEY

ENDPOINT = "/v1/activation/raw"

GPT2_N_LAYERS = 12
GPT2_D_MODEL = 768


def post(client: TestClient, **body: object):
    return client.post(ENDPOINT, json=body, headers={"X-SECRET-KEY": X_SECRET_KEY})


def test_defaults_to_every_layer(client: TestClient):
    response = post(client, model=MODEL_ID, prompts=[TEST_PROMPT])
    assert response.status_code == 200

    data = response.json()
    assert data["hookPoint"] == "residual_stream"
    assert data["type"] == "final_output_token"
    assert len(data["results"]) == 1

    result = data["results"][0]
    assert [layer["layer"] for layer in result["activations"]] == list(range(GPT2_N_LAYERS))
    final_index = len(result["tokenIds"]) - 1
    for layer in result["activations"]:
        assert layer["tokenIndices"] == [final_index]
        assert len(layer["values"]) == 1
        assert len(layer["values"][0]) == GPT2_D_MODEL


def test_layers_parameter_selects_and_orders_layers(client: TestClient):
    # Deliberately unsorted and duplicated: the response is deduplicated and ascending.
    response = post(client, model=MODEL_ID, prompts=[TEST_PROMPT], layers=[5, 0, 5])
    assert response.status_code == 200

    activations = response.json()["results"][0]["activations"]
    assert [layer["layer"] for layer in activations] == [0, 5]


def test_batched_prompts_use_their_own_final_token(client: TestClient):
    short, long = "Hello", "Hello, world! This is a longer prompt."
    response = post(client, model=MODEL_ID, prompts=[short, long], layers=[0])
    assert response.status_code == 200

    results = response.json()["results"]
    assert len(results) == 2
    for result in results:
        # Padding must not leak in: each prompt reports its own length and reads the last
        # real token, not the last column of the padded batch.
        assert result["activations"][0]["tokenIndices"] == [len(result["tokenIds"]) - 1]
    assert len(results[0]["tokenIds"]) < len(results[1]["tokenIds"])


def test_matches_a_direct_resid_post_capture(client: TestClient):
    """The returned vector is ``resid_post[layer]`` at the final token, to fp16 precision."""
    model = Model.get_instance()
    if not isinstance(model, EagerModel):
        pytest.skip("parity check reads the eager cache directly")

    layer = 5
    response = post(client, model=MODEL_ID, prompts=[TEST_PROMPT], layers=[layer])
    assert response.status_code == 200
    served = torch.tensor(response.json()["results"][0]["activations"][0]["values"][0])

    # The endpoint prepends BOS to the string, so reproduce that here rather than relying
    # on the tokenizer's defaults.
    bos = model.tokenizer.bos_token or ""
    prompt = TEST_PROMPT if TEST_PROMPT.startswith(bos) else bos + TEST_PROMPT
    tokens = model.to_tokens(prompt, prepend_bos=False, truncate=False)
    cache = run_with_cache(model, tokens, [("resid_post", layer)])
    expected = cache.get("resid_post", layer)[0, -1, :].float().cpu()

    assert served.shape == expected.shape
    torch.testing.assert_close(served, expected, atol=1e-2, rtol=1e-2)


def test_rejects_out_of_range_layers(client: TestClient):
    response = post(client, model=MODEL_ID, prompts=[TEST_PROMPT], layers=[0, GPT2_N_LAYERS])
    assert response.status_code == 400
    assert str(GPT2_N_LAYERS) in response.json()["error"]


def test_rejects_an_empty_batch(client: TestClient):
    assert post(client, model=MODEL_ID, prompts=[]).status_code == 400


def test_rejects_unsupported_hook_point(client: TestClient):
    response = post(client, model=MODEL_ID, prompts=[TEST_PROMPT], hook_point="mlp_out")
    assert response.status_code == 400


def test_serves_a_request_naming_an_unknown_model(client: TestClient):
    """The ``model`` field is advisory: a pod holds one model and answers for it regardless."""
    response = post(client, model="some/model-this-pod-never-heard-of", prompts=[TEST_PROMPT], layers=[0])
    assert response.status_code == 200


def test_model_is_optional(client: TestClient):
    assert post(client, prompts=[TEST_PROMPT], layers=[0]).status_code == 200
