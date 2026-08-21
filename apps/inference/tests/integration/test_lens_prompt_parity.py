"""Engine lens read-out self-consistency.

The cross-backend parity tests (engine vs TransformerLens vs nnsight/nnterp) that gated the
migration were retired when those two backends were removed; the interp-engine is now the
only lens backend. What remains here is an engine-only correctness gate: the residual captured by
the streaming read-out path, decoded through the model's real final-norm + lm_head, must reproduce
the model's true next-token prediction (this also validates the arch mapping — final_norm /
lm_head). The golden engine-vs-TLens parity lives in the standalone engine suite
(the engine's tests/test_parity_gpt2.py).

Slow (downloads + runs gpt2 on CPU). Run explicitly with, e.g.:

    cd apps/inference && uv run pytest tests/integration/test_lens_prompt_parity.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from neuronpedia_inference.config import Config

HF_MODEL_ID = "openai-community/gpt2"
PROMPT = "The capital of France is"
N_LAYERS = 12
LAYERS = [0, 4, 8, 11]


class _StubConfig:
    """Minimal stand-in for Config used by the read-out path."""

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
def engine_model():
    from interp_engine import EagerModel

    return EagerModel(HF_MODEL_ID, dtype="float32", device="cpu", attn_implementation="eager")


def test_engine_layer_logits_shapes(stub_config: None, engine_model: Any):  # noqa: ARG001
    """The per-layer read-out returns [seq, vocab] logits for each requested layer."""
    from neuronpedia_inference.endpoints.lens.prompt import (
        LensType,
        _compute_logits_for_types,
    )

    token_ids = list(engine_model.tokenizer(PROMPT, add_special_tokens=False)["input_ids"])
    layers_by_type = {LensType.LOGIT_LENS: LAYERS}
    out = _compute_logits_for_types(engine_model, token_ids, layers_by_type, None, None)[LensType.LOGIT_LENS]
    for layer in LAYERS:
        assert out[layer].shape == (len(token_ids), engine_model.vocab_size)


def test_iter_residuals_engine_prefill_decodes_to_true_output(
    stub_config: None,  # noqa: ARG001
    engine_model: Any,
):
    """The last-layer residual captured by the engine streaming path must decode
    (via the real norm + lm_head) to the model's true next-token prediction."""
    from neuronpedia_inference.endpoints.lens.prompt import (
        _decode_residuals,
        _iter_residuals,
    )

    token_ids = list(engine_model.tokenizer(PROMPT, add_special_tokens=False)["input_ids"])

    async def _collect():
        # `_iter_residuals` is an async generator of position BATCHES (unifying the eager
        # + vLLM paths), and `_decode_residuals` is a coroutine, so drive them on an event
        # loop and flatten.
        steps = [
            step
            async for batch in _iter_residuals(
                engine_model,
                token_ids,
                [N_LAYERS - 1],
                num_completion_tokens=3,
                temperature=0.0,
                eos_token_ids=None,
            )
            for step in batch
        ]
        # Decoding the last prompt position's last-layer residual should predict the
        # greedy next token, which for a greedy generated run equals the next step's id.
        last_resid = steps[len(token_ids) - 1][2][N_LAYERS - 1]
        logits = (await _decode_residuals(engine_model, last_resid.unsqueeze(0)))[0]
        return steps, logits

    steps, logits = asyncio.run(_collect())

    # Prompt positions (is_generated=False) + 3 generated.
    assert sum(1 for _, gen, _ in steps if not gen) == len(token_ids)
    assert sum(1 for _, gen, _ in steps if gen) == 3

    first_generated_id = steps[len(token_ids)][0]
    assert int(logits.argmax().item()) == first_generated_id
