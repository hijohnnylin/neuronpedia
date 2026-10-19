"""MLP neuron values from /activation/single: BOS handling, with and without a BOS token."""

import torch

from neuronpedia_inference.endpoints.activation.single import bos_positions, process_neuron_activations

HOOK = "blocks.0.mlp.hook_post"


def _cache(values: list[float]) -> dict[str, torch.Tensor]:
    """A [1, seq, d_mlp=2] cache whose neuron 1 holds ``values``."""
    acts = torch.zeros(1, len(values), 2)
    acts[0, :, 1] = torch.tensor(values)
    return {HOOK: acts}


def test_bos_positions_without_a_bos_token():
    """Qwen has no BOS token; comparing against None must not raise."""
    assert bos_positions(torch.tensor([5, 6, 7]), None) == []


def test_bos_positions_finds_bos():
    """Positions of the BOS id."""
    assert bos_positions(torch.tensor([1, 6, 1]), 1) == [0, 2]


def test_neuron_values_drop_a_leading_bos():
    """A large activation on BOS must not become the max, as for SAE features."""
    result = process_neuron_activations(_cache([90.0, 0.5, 2.0, -1.0]), HOOK, 1, "cpu", [0])
    assert result.values == [0.5, 2.0, -1.0]
    assert result.max_value == 2.0
    assert result.max_value_index == 1


def test_neuron_values_without_bos_keep_every_position():
    """No BOS: nothing is dropped."""
    result = process_neuron_activations(_cache([0.5, 2.0, -1.0]), HOOK, 1, "cpu")
    assert result.values == [0.5, 2.0, -1.0]
