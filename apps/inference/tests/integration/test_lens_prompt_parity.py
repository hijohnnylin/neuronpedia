"""Parity test: the logit-lens read-out must agree across inference engines.

Loads gpt2-small under both TransformerLens (``HookedTransformer``) and
nnsight/nnterp (``StandardizedTransformer``) on CPU and checks that the
per-layer LOGIT_LENS logits agree. This is the correctness gate for serving the
lens on either engine: the Jacobian lens is fitted in the raw-HF residual basis,
so the engines must reproduce the same logit-lens baseline for ``J_bar`` to be
meaningful when applied at serve time.

Slow (downloads + runs two models on CPU). Run explicitly with, e.g.:

    cd apps/inference && uv run pytest tests/integration/test_lens_prompt_parity.py -v
"""

from __future__ import annotations

import pytest
import torch

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.lens.prompt import (
    LensType,
    _compute_logits_for_types,
)

HF_MODEL_ID = "openai-community/gpt2"
PROMPT = "The capital of France is"
N_LAYERS = 12
LAYERS = [0, 4, 8, 11]


class _StubConfig:
    """Minimal stand-in for Config used by the nnsight read-out path."""

    model_dtype = "float32"
    device = "cpu"
    num_layers = N_LAYERS
    token_limit = 500
    lens_token_limit = 1024
    model_id = "gpt2-small"
    custom_hf_model_id = None
    override_model_id = None


@pytest.fixture(scope="module")
def stub_config():
    previous = Config._instance
    Config._instance = _StubConfig()  # type: ignore[assignment]
    yield
    Config._instance = previous


@pytest.fixture(scope="module")
def tlens_model():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained_no_processing(
        "gpt2-small", device="cpu", dtype=torch.float32
    )


@pytest.fixture(scope="module")
def nnsight_model():
    from nnterp import StandardizedTransformer

    return StandardizedTransformer(HF_MODEL_ID, dtype=torch.float32)


def test_logit_lens_parity_across_engines(
    stub_config: None,  # noqa: ARG001
    tlens_model,
    nnsight_model,
):
    token_ids = list(
        tlens_model.tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    )
    assert len(token_ids) > 1

    layers_by_type = {LensType.LOGIT_LENS: LAYERS}
    tl_logits = _compute_logits_for_types(
        tlens_model, token_ids, layers_by_type, None, None
    )[LensType.LOGIT_LENS]
    nn_logits = _compute_logits_for_types(
        nnsight_model, token_ids, layers_by_type, None, None
    )[LensType.LOGIT_LENS]

    for layer in LAYERS:
        a = tl_logits[layer]
        b = nn_logits[layer]
        assert a.shape == b.shape

        # Top-1 next-token agreement across positions.
        agreement = (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()

        # Logit-distribution correlation across the whole vocab (last position).
        a_last = a[-1]
        b_last = b[-1]
        corr = torch.corrcoef(torch.stack([a_last, b_last]))[0, 1].item()

        if layer == N_LAYERS - 1:
            # Final layer is the model's true output on both engines.
            assert agreement == pytest.approx(1.0, abs=1e-6)
            assert corr > 0.999
        else:
            assert (
                agreement >= 0.75
            ), f"layer {layer}: top-1 agreement {agreement:.3f} too low"
            assert corr > 0.99, f"layer {layer}: logit correlation {corr:.4f} too low"
