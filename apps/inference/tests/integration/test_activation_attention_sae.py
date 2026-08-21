"""Attention-output (`hook_z`) SAE sources, end to end through /activation/single.

Two bugs made every `-att-`/`-att_` source 500 after the move off TransformerLens, and
both are pinned here:

1. SAELens hands attention-output SAEs an input reshape that folds the last two dims
   (`... n_heads d_head -> ... (n_heads d_head)`), because TransformerLens' `hook_z` is
   `[batch, pos, n_heads, d_head]`. The engine captures `z` as the INPUT to the attention
   output projection, which is already the concatenated `[batch, pos, n_heads*d_head]`, so
   the reshape folded the POSITION axis into `d_in` instead and `sae.encode` raised "The
   size of tensor a (262144) must match the size of tensor b (2048)".
2. DFA then failed on the vLLM backend only, where the captured value/attn_probs come back
   on the CPU while the SAE weights are on the GPU ("Expected all tensors to be on the same
   device"). Hence the engine parametrization: the eager backend cannot see that one.

This file NEVER runs automatically: every row downloads SAE weights, the gemma-2-2b rows
want ~5 GB of model weights plus a token for the gated repo, and the vLLM rows need a GPU.
Run it by hand after touching activation capture or SAE loading:

  cd apps/inference
  NP_RUN_MANUAL_TESTS=1 uv run pytest tests/integration/test_activation_attention_sae.py -v

Rows self-skip when the box can't serve them (no GPU -> vLLM, no HF token -> gemma-2-2b),
so check for PASSED rather than a zero exit.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationSingleResponse,
)
from tests.harness import (
    EAGER,
    VLLM,
    X_SECRET_KEY,
    ModelSpec,
    initialized_server,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("NP_RUN_MANUAL_TESTS"),
    reason="manual test (downloads SAE/model weights); set NP_RUN_MANUAL_TESTS=1 to run",
)

ENDPOINT = "/v1/activation/single"

# Long enough that a position axis folded into `d_in` cannot coincidentally match it.
PROMPT = "When Mary and John went to the store, John gave a drink to Mary."

GPT2_ATT_SOURCE = "7-att-kk"
GPT2_ATT = ModelSpec(
    key="gpt2-att-kk",
    model_id="openai-community/gpt2",
    dtype="float32",
    sae_sets=["att-kk"],
    include_sae=[GPT2_ATT_SOURCE],
)

GEMMA_ATT_SOURCE = "6-gemmascope-att-16k"
GEMMA_ATT = ModelSpec(
    key="gemma-2-2b-gemmascope-att",
    model_id="google/gemma-2-2b",
    dtype="bfloat16",
    is_gated=True,
    sae_sets=["gemmascope-att-16k"],
    include_sae=[GEMMA_ATT_SOURCE],
)


@pytest.mark.parametrize("engine", [EAGER, VLLM])
@pytest.mark.parametrize(
    ("spec", "source"),
    [(GPT2_ATT, GPT2_ATT_SOURCE), (GEMMA_ATT, GEMMA_ATT_SOURCE)],
    ids=["openai-community/gpt2", "google/gemma-2-2b"],
)
def test_activation_single_on_an_attention_sae(spec: ModelSpec, source: str, engine: str):
    """One activation per returned token, plus DFA, from an `-att-` source."""
    with initialized_server(spec, engine=engine) as client:
        response = client.post(
            ENDPOINT,
            json={
                "prompt": PROMPT,
                "model": spec.model_id,
                "source": source,
                "index": "0",
            },
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        assert response.status_code == 200, response.text
        data = ActivationSingleResponse(**response.json())
        activation = data.activation

        # The shape bug's signature: the feature activations are indexed by token
        # position, so a folded position axis cannot line up here even if `encode`
        # were to survive it.
        assert len(activation.values) == len(data.tokens)
        assert all(math.isfinite(v) for v in activation.values)
        assert activation.max_value == pytest.approx(max(activation.values))
        assert activation.values[activation.max_value_index] == pytest.approx(activation.max_value)

        # `-att-` sources carry DFA (attention probs x value), one value per SOURCE
        # position, in the SAME coordinates as `values`/`tokens`. It used to come back one
        # longer, because the attention pattern still counts the BOS these drop, and a
        # consumer reading `dfaValues[i]` for token `i` was then reading its neighbour.
        assert activation.dfa_values is not None
        assert len(activation.dfa_values) == len(data.tokens)
        assert all(math.isfinite(v) for v in activation.dfa_values)
        assert activation.dfa_max_value == pytest.approx(max(activation.dfa_values))
        # The destination indexes `values` too, so it names the token that actually fired.
        assert activation.dfa_target_index == activation.max_value_index


def _single(client, spec: ModelSpec, source: str, index: int) -> dict:
    response = client.post(
        ENDPOINT,
        json={
            "prompt": PROMPT,
            "model": spec.model_id,
            "source": source,
            "index": str(index),
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 200, response.text
    return response.json()["activation"]


def _top_feature_index(client, spec: ModelSpec, source: str) -> int:
    """The source's highest-activating feature on PROMPT, so the parity check has signal."""
    response = client.post(
        "/v1/activation/all",
        json={
            "prompt": PROMPT,
            "model": spec.model_id,
            "source_set": source.split("-", 1)[1],
            "selected_sources": [source],
            "sort_by_token_indexes": [],
            "ignore_bos": True,
            "num_results": 1,
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 200, response.text
    top = response.json()["activations"][0]
    assert top["maxValue"] > 0, "no feature fires on this prompt; parity would be vacuous"
    return int(top["index"])


def test_attention_sae_reads_the_same_on_both_backends():
    """Both backends must AGREE, not merely respond.

    The endpoint test above would pass on plausible-looking wrong numbers, which is the
    real risk in `z` space: the eager and vLLM captures come from different code paths
    (`o_proj` input vs a worker forward-hook) and the DFA operands from different devices.
    """
    with initialized_server(GPT2_ATT, engine=EAGER) as client:
        index = _top_feature_index(client, GPT2_ATT, GPT2_ATT_SOURCE)
        eager = _single(client, GPT2_ATT, GPT2_ATT_SOURCE, index)

    with initialized_server(GPT2_ATT, engine=VLLM) as client:
        vllm = _single(client, GPT2_ATT, GPT2_ATT_SOURCE, index)

    assert vllm["values"] == pytest.approx(eager["values"], abs=1e-3)
    assert vllm["dfaValues"] == pytest.approx(eager["dfaValues"], abs=1e-3)
    assert vllm["maxValueIndex"] == eager["maxValueIndex"]


def test_attention_sae_encodes_concatenated_z():
    """The loader contract behind the endpoint test: `d_in` is the concatenated width.

    Asserted on the SAE itself so a future SAELens/loader change is diagnosed here
    rather than as a 500 from whichever activation endpoint runs first.
    """
    with initialized_server(GPT2_ATT):
        sae = SAEManager.get_instance().get_sae(GPT2_ATT_SOURCE)
        assert sae.hook_z_reshaping_mode is False

        n_positions = 5
        z = torch.zeros(1, n_positions, sae.cfg.d_in, dtype=sae.dtype, device=sae.device)
        assert tuple(sae.encode(z).shape) == (1, n_positions, sae.cfg.d_sae)
