"""search-all with rank_by_percentile: rank neurons by how rare their activation is for them.

CPU-only, with small synthetic tables in the dashboard format.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from neuronpedia_inference.endpoints.activation.all import rank_by_percentile_error
from neuronpedia_inference.inference_utils.neuron_act_range import (
    MIN_SCALE_FRACTION_OF_LAYER_MEDIAN,
    NeuronActRangeError,
    load_neuron_rarity_table,
    neuron_rarity,
)
from tests.unit.test_activation_all_topk import (
    N_FEATURES,
    N_TOKENS,
    SOURCES,
    _activations,
    _reference_global_sort,
    _request,
    _stream,
)

MODULE = "neuronpedia_inference.endpoints.activation.all"
LEVELS = [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 0.9999, 0.99999]
N_CORPUS_TOKENS = 10**8


def _write_table(
    tmp_path: Path, quantiles: torch.Tensor, table_max: torch.Tensor, n_tokens: int = N_CORPUS_TOKENS
) -> str:
    """Write a table and clear the loader cache, so each test reads its own file."""
    path = tmp_path / "act_range.safetensors"
    save_file(
        {
            "max": table_max.contiguous(),
            "min": (-table_max).contiguous(),
            "quantile_levels": torch.tensor(LEVELS, dtype=torch.float64),
            "quantiles": quantiles.contiguous(),
            "n_tokens": torch.tensor([n_tokens], dtype=torch.int64),
        },
        str(path),
    )
    load_neuron_rarity_table.cache_clear()
    return str(path)


def _random_table(n_layers: int, n_features: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Ascending quantiles per neuron, with neuron scales spread over two orders of magnitude."""
    generator = torch.Generator().manual_seed(seed)
    steps = torch.rand(n_layers, n_features, len(LEVELS), generator=generator)
    scale = 10 ** (torch.rand(n_layers, n_features, 1, generator=generator) * 2 - 1)
    quantiles = (torch.cumsum(steps, dim=-1) - steps.sum(-1, keepdim=True) / 2) * scale
    table_max = quantiles[..., -1] + scale.squeeze(-1)
    return quantiles.float(), table_max.float()


def _reference_rarity(quantiles: torch.Tensor, table_max: torch.Tensor, layer: int, n: int, a: float) -> float:
    """Scalar oracle: interpolate -log10(1 - p) between the median-and-up knots and the max."""
    row_max = table_max[layer]
    positive = row_max[row_max > 0]
    floor = (float(positive.median()) if positive.numel() else 1.0) * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN
    knots, rarity = [], []
    for p, v in zip(LEVELS, quantiles[layer, n].tolist(), strict=True):
        if p >= 0.5:
            knots.append(v)
            rarity.append(-math.log10(1 - p))
    knots.append(float(table_max[layer, n]))
    rarity.append(math.log10(N_CORPUS_TOKENS))
    running = -math.inf
    for i, k in enumerate(knots):
        running = max(running, k, floor)
        knots[i] = running
    if a <= knots[0]:
        return 0.0
    if a > knots[-1]:
        return rarity[-1] + math.log10(a / knots[-1])
    for i in range(1, len(knots)):
        if a <= knots[i]:
            span = knots[i] - knots[i - 1]
            t = (a - knots[i - 1]) / span if span > 0 else 1.0
            return rarity[i - 1] + t * (rarity[i] - rarity[i - 1])
    raise AssertionError("unreachable")


def test_rarity_matches_the_scalar_oracle(tmp_path: Path):
    quantiles, table_max = _random_table(2, 40, seed=3)
    path = _write_table(tmp_path, quantiles, table_max)
    generator = torch.Generator().manual_seed(4)
    for layer in range(2):
        # From below the median to well past the max, so every branch is hit.
        a = (torch.rand(40, generator=generator) * 1.6 - 0.2) * (table_max[layer] * 1.3)
        got = neuron_rarity(path, layer, a)
        want = [_reference_rarity(quantiles, table_max, layer, n, float(a[n])) for n in range(40)]
        assert got.tolist() == pytest.approx(want, abs=1e-4)


def test_rarity_at_the_knots(tmp_path: Path):
    """The 99th percentile scores 2, the 99.999th 5, the corpus max log10(n_tokens)."""
    quantiles = torch.tensor([[[float(i) for i in range(len(LEVELS))]]])
    path = _write_table(tmp_path, quantiles, torch.tensor([[20.0]]))
    p99, p99999 = LEVELS.index(0.99), LEVELS.index(0.99999)
    values = [float(p99), float(p99999), 20.0, 40.0, 2.0]

    got = [neuron_rarity(path, 0, torch.tensor([v])).item() for v in values]

    assert got == pytest.approx([2.0, 5.0, 8.0, 8.0 + math.log10(2.0), 0.0], abs=1e-5)


@pytest.mark.parametrize("sort_by_token_indexes", [None, [1, 3]], ids=["by-max-value", "by-sort-tokens"])
def test_ranked_streaming_matches_the_global_sort(tmp_path: Path, sort_by_token_indexes: list[int] | None):
    """Per-source reduction stays exact when the key is the rarity of each row's peak."""
    quantiles, table_max = _random_table(len(SOURCES), N_FEATURES, seed=1)
    path = _write_table(tmp_path, quantiles, table_max)
    blocks = _activations()
    request = _request(6, sort_by_token_indexes=sort_by_token_indexes)
    request.rank_by_percentile = True

    streamed = _stream(blocks, request, act_range_path=path)

    expected = _reference_global_sort(blocks, request, rank_key=lambda layer, peak: neuron_rarity(path, layer, peak))
    assert len(streamed) == len(expected)
    for rank, (got, want) in enumerate(zip(streamed, expected, strict=True)):
        assert got == pytest.approx(want), f"row {rank} differs"


def test_a_rare_activation_outranks_a_large_common_one(tmp_path: Path):
    """The point of the option: a sparse neuron at its 99.999th percentile beats a dense one
    at 60% of its max, though the dense one's raw value is four times larger."""
    quantiles = torch.zeros(len(SOURCES), N_FEATURES, len(LEVELS))
    table_max = torch.ones(len(SOURCES), N_FEATURES)
    dense = torch.linspace(-1.0, 9.5, len(LEVELS))  # 6.0, 60% of its max, is its 95th percentile
    sparse = torch.tensor([0.0] * 10 + [0.1, 0.3, 1.0])
    quantiles[2, 0], table_max[2, 0] = dense, 10.0
    quantiles[0, 4], table_max[0, 4] = sparse, 2.0
    path = _write_table(tmp_path, quantiles, table_max)
    blocks = [torch.zeros(N_FEATURES, N_TOKENS) for _ in SOURCES]
    blocks[2][0, 3] = 6.0
    blocks[0][4, 3] = 1.5
    request = _request(2)

    raw = _stream(blocks, request)
    request.rank_by_percentile = True
    ranked = _stream(blocks, request, act_range_path=path)

    assert [(r[0], r[1]) for r in raw] == [(2, 0), (0, 4)]
    assert [(r[0], r[1]) for r in ranked] == [(0, 4), (2, 0)]
    assert ranked[0][5] > 5.0 > ranked[1][5]
    # The returned activations stay raw.
    assert ranked[0][2] == pytest.approx(1.5)
    assert ranked[1][2] == pytest.approx(6.0)


def test_a_near_zero_neuron_needs_the_layer_floor_to_score(tmp_path: Path):
    """A dead neuron's knots are raised to 5% of the layer's median max, so a small
    activation is not rare for it."""
    quantiles = torch.zeros(1, 5, len(LEVELS))
    quantiles[0, 1:] = torch.linspace(0.0, 3.0, len(LEVELS))
    table_max = torch.tensor([[1e-6, 4.0, 4.0, 4.0, 4.0]])
    path = _write_table(tmp_path, quantiles, table_max)
    floor = 4.0 * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN

    below = neuron_rarity(path, 0, torch.tensor([floor * 0.9, 0.0, 0.0, 0.0, 0.0]))
    above = neuron_rarity(path, 0, torch.tensor([floor * 2, 0.0, 0.0, 0.0, 0.0]))

    assert below[0].item() == 0.0
    assert above[0].item() == pytest.approx(math.log10(N_CORPUS_TOKENS) + math.log10(2.0), abs=1e-5)


@pytest.mark.parametrize(
    ("layer", "d_mlp", "message"),
    [(3, N_FEATURES, "layer 3 is out of range"), (0, N_FEATURES + 1, "neurons per layer")],
    ids=["layer-out-of-range", "width-mismatch"],
)
def test_a_table_that_does_not_fit_the_model_is_refused(tmp_path: Path, layer: int, d_mlp: int, message: str):
    quantiles, table_max = _random_table(len(SOURCES), N_FEATURES, seed=2)
    path = _write_table(tmp_path, quantiles, table_max)
    with pytest.raises(NeuronActRangeError, match=message):
        neuron_rarity(path, layer, torch.zeros(d_mlp))


def test_no_table_is_refused():
    with pytest.raises(NeuronActRangeError, match="NEURON_ACT_RANGE_PATH"):
        neuron_rarity(None, 0, torch.zeros(N_FEATURES))


def test_a_max_only_table_is_refused(tmp_path: Path):
    """A table from before quantiles were added names what it lacks."""
    path = tmp_path / "act_range.safetensors"
    save_file({"max": torch.ones(1, N_FEATURES), "min": -torch.ones(1, N_FEATURES)}, str(path))
    load_neuron_rarity_table.cache_clear()
    with pytest.raises(NeuronActRangeError, match="quantile_levels"):
        neuron_rarity(str(path), 0, torch.zeros(N_FEATURES))


@pytest.mark.parametrize("contents", [None, b"not a safetensors file"], ids=["missing", "corrupt"])
def test_an_unreadable_table_is_refused(tmp_path: Path, contents: bytes | None):
    """A table lost from the pod's disk gives a 400 with the reason, not a 500."""
    path = tmp_path / "act_range.safetensors"
    if contents is not None:
        path.write_bytes(contents)
    load_neuron_rarity_table.cache_clear()
    with pytest.raises(NeuronActRangeError, match="Cannot read"):
        neuron_rarity(str(path), 0, torch.zeros(N_FEATURES))


@pytest.mark.parametrize(
    ("rank", "source_set", "path", "status"),
    [
        (False, "res-test", None, None),
        (True, "res-test", "/tables/x.safetensors", 400),
        (True, "neurons", None, 400),
        (True, "neurons", "/tables/x.safetensors", None),
    ],
    ids=["off", "sae-source-set", "no-table", "served"],
)
def test_requests_that_cannot_be_served_get_a_400(rank: bool, source_set: str, path: str | None, status):
    request = MagicMock()
    request.rank_by_percentile = rank
    request.source_set = source_set
    config = MagicMock()
    config.neuron_act_range_path = path

    with patch(f"{MODULE}.Config.get_instance", return_value=config):
        error = rank_by_percentile_error(request)

    assert (error.status_code if error is not None else None) == status
