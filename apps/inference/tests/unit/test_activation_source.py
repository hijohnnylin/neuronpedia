"""/activation/source: which token positions it skips, for SAEs and MLP neuron layers.

CPU-only. The model, SAE manager and capture are mocks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from neuronpedia_inference.endpoints.activation.source import ActivationProcessor
from neuronpedia_inference.sae_manager import SAE_TYPE
from neuronpedia_inference.schemas import ActivationSourceRequest

MODULE = "neuronpedia_inference.endpoints.activation.source"
HOOK = "blocks.0.mlp.hook_post"
BOS_ID = 50
N_FEATURES = 4


def _run(token_ids: list[int], bos_token_id: int | None, source_type: str) -> dict[str, list[list[float]]]:
    """Run one prompt through the endpoint's processor and return its active features."""
    model = MagicMock()
    model.to_tokens.return_value = torch.tensor([token_ids])
    model.to_str_tokens.return_value = [f"t{t}" for t in token_ids]
    model.tokenizer.bos_token_id = bos_token_id
    model.tokenizer.pad_token_id = 0

    # Every feature is non-zero at every position, so only the skip rule removes entries.
    acts = torch.arange(1, len(token_ids) * N_FEATURES + 1, dtype=torch.float32).reshape(1, len(token_ids), -1)
    sae = MagicMock()
    sae.encode.side_effect = lambda x: x
    sae_manager = MagicMock()
    sae_manager.get_sae_hook.return_value = HOOK
    sae_manager.get_sae_type.return_value = source_type
    sae_manager.get_sae.return_value = sae

    config = MagicMock()
    config.activation_token_limit = 1024
    config.device = "cpu"
    config.neuron_act_range_path = None

    source = "0" if source_type == SAE_TYPE.NEURONS else "0-res-test"
    request = ActivationSourceRequest(prompts=["x"], model="m", source=source, top_k=2)
    with (
        patch(f"{MODULE}.Model.get_instance", return_value=model),
        patch(f"{MODULE}.SAEManager.get_instance", return_value=sae_manager),
        patch(f"{MODULE}.Config.get_instance", return_value=config),
        patch(f"{MODULE}.capture_padded_cache_async", AsyncMock(return_value={HOOK: acts})),
    ):
        results = asyncio.run(ActivationProcessor().process_activations_batch(request, request.prompts))
    return results[0].active_features or {}


def _positions(active_features: dict[str, list[list[float]]]) -> set[int]:
    return {int(token) for entries in active_features.values() for token, _ in entries}


@pytest.mark.parametrize("source_type", [SAE_TYPE.NEURONS, SAE_TYPE.SAELENS], ids=["neurons", "sae"])
def test_a_leading_bos_is_skipped(source_type: str):
    assert _positions(_run([BOS_ID, 7, 8], BOS_ID, source_type)) == {1, 2}


@pytest.mark.parametrize("source_type", [SAE_TYPE.NEURONS, SAE_TYPE.SAELENS], ids=["neurons", "sae"])
def test_with_no_bos_token_the_first_token_is_kept(source_type: str):
    """Qwen3 has no BOS, so position 0 is the first word of the prompt."""
    assert _positions(_run([7, 8, 9], None, source_type)) == {0, 1, 2}


def test_neuron_layers_keep_top_k_per_token():
    active_features = _run([7, 8, 9], None, SAE_TYPE.NEURONS)

    # Values grow with the neuron index, so the top 2 at each token are the last two neurons.
    assert set(active_features) == {str(N_FEATURES - 2), str(N_FEATURES - 1)}
    assert all(len(entries) == 3 for entries in active_features.values())
