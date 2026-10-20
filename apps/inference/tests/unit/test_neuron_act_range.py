"""search-all with rank_by_percentile: rank neurons by how rare their activation is for them.

CPU-only, with small synthetic tables in the dashboard format.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from neuronpedia_inference.endpoints.activation.all import ActivationProcessor, rank_by_percentile_error
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
TOP_RARITY = math.log10(N_CORPUS_TOKENS)


def _write_table(
    tmp_path: Path,
    quantiles: torch.Tensor,
    table_max: torch.Tensor,
    table_min: torch.Tensor | None = None,
    n_tokens: int = N_CORPUS_TOKENS,
) -> str:
    """Write a table and clear the loader cache, so each test reads its own file."""
    path = tmp_path / "act_range.safetensors"
    save_file(
        {
            "max": table_max.contiguous(),
            "min": (-table_max if table_min is None else table_min).contiguous(),
            "quantile_levels": torch.tensor(LEVELS, dtype=torch.float64),
            "quantiles": quantiles.contiguous(),
            "n_tokens": torch.tensor([n_tokens], dtype=torch.int64),
        },
        str(path),
    )
    load_neuron_rarity_table.cache_clear()
    return str(path)


def _random_table(n_layers: int, n_features: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ascending quantiles per neuron around zero, with scales spread over two orders of magnitude."""
    generator = torch.Generator().manual_seed(seed)
    steps = torch.rand(n_layers, n_features, len(LEVELS), generator=generator)
    scale = 10 ** (torch.rand(n_layers, n_features, 1, generator=generator) * 2 - 1)
    quantiles = (torch.cumsum(steps, dim=-1) - steps.sum(-1, keepdim=True) / 2) * scale
    table_max = quantiles[..., -1] + scale.squeeze(-1)
    table_min = quantiles[..., 0] - scale.squeeze(-1)
    return quantiles.float(), table_max.float(), table_min.float()


def _reference_tail(points: list[tuple[float, float]], extreme: float, layer_extremes: torch.Tensor, d: float) -> float:
    """Scalar oracle for one tail: interpolate rarity between (distance, rarity) knots and the extreme."""
    positive = layer_extremes[layer_extremes > 0]
    floor = (float(positive.median()) if positive.numel() else 1.0) * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN
    knots = [v for v, _ in points] + [extreme]
    rarity = [r for _, r in points] + [TOP_RARITY]
    running = -math.inf
    for i, k in enumerate(knots):
        running = max(running, k, floor)
        knots[i] = running
    if d <= knots[0]:
        return 0.0
    if d > knots[-1]:
        return rarity[-1] + math.log10(d / knots[-1])
    for i in range(1, len(knots)):
        if d <= knots[i]:
            span = knots[i] - knots[i - 1]
            t = (d - knots[i - 1]) / span if span > 0 else 1.0
            return rarity[i - 1] + t * (rarity[i] - rarity[i - 1])
    raise AssertionError("unreachable")


def _reference_rarity(
    quantiles: torch.Tensor,
    table_max: torch.Tensor,
    table_min: torch.Tensor,
    layer: int,
    n: int,
    highest: float,
    lowest: float,
) -> tuple[float, bool]:
    """-log10(1 - p) up from the median, -log10(p) down from it; the larger one wins."""
    values = quantiles[layer, n].tolist()
    high_points = [(v, -math.log10(1 - p)) for p, v in zip(LEVELS, values, strict=True) if p >= 0.5]
    low_points = [(-v, -math.log10(p)) for p, v in reversed(list(zip(LEVELS, values, strict=True))) if p <= 0.5]
    high = _reference_tail(high_points, float(table_max[layer, n]), table_max[layer], highest)
    low = _reference_tail(low_points, -float(table_min[layer, n]), -table_min[layer], -lowest)
    return max(high, low), low > high


def _symmetric_table(tmp_path: Path) -> str:
    """One neuron: quantile k is k - 6 (median 0, 1st and 99th percentiles -3 and 3), max 20, min -20."""
    quantiles = torch.tensor([[[float(i - 6) for i in range(len(LEVELS))]]])
    return _write_table(tmp_path, quantiles, torch.tensor([[20.0]]), torch.tensor([[-20.0]]))


def test_rarity_matches_the_scalar_oracle(tmp_path: Path):
    quantiles, table_max, table_min = _random_table(2, 40, seed=3)
    path = _write_table(tmp_path, quantiles, table_max, table_min)
    generator = torch.Generator().manual_seed(4)
    for layer in range(2):
        # From inside the median to well past each extreme, so every branch of both tails is hit.
        highest = (torch.rand(40, generator=generator) * 1.6 - 0.2) * (table_max[layer] * 1.3)
        lowest = (torch.rand(40, generator=generator) * 1.6 - 0.2) * (table_min[layer] * 1.3)
        got, got_low = neuron_rarity(path, layer, highest, lowest)
        want = [
            _reference_rarity(quantiles, table_max, table_min, layer, n, float(highest[n]), float(lowest[n]))
            for n in range(40)
        ]
        assert got.tolist() == pytest.approx([w[0] for w in want], abs=1e-4)
        assert got_low.tolist() == [w[1] for w in want]
        assert any(w[1] for w in want) and not all(w[1] for w in want)


@pytest.mark.parametrize(
    ("value", "rarity", "low"),
    [
        (3.0, 2.0, False),
        (6.0, 5.0, False),
        (20.0, TOP_RARITY, False),
        (40.0, TOP_RARITY + math.log10(2.0), False),
        (-3.0, 2.0, True),
        (-6.0, 5.0, True),
        (-20.0, TOP_RARITY, True),
        (-40.0, TOP_RARITY + math.log10(2.0), True),
        (0.0, 0.0, False),
    ],
)
def test_rarity_at_the_knots_of_each_tail(tmp_path: Path, value: float, rarity: float, low: bool):
    """The 99th and 1st percentiles score 2, the 99.999th and 0.001st 5, the corpus extremes log10(n_tokens)."""
    path = _symmetric_table(tmp_path)

    got, got_low = neuron_rarity(path, 0, torch.tensor([value]), torch.tensor([value]))

    assert got.item() == pytest.approx(rarity, abs=1e-5)
    assert bool(got_low.item()) is low


def test_the_rarer_extreme_wins(tmp_path: Path):
    """The 99th percentile high and the 0.001st low: the low one ranks the neuron."""
    path = _symmetric_table(tmp_path)

    got, got_low = neuron_rarity(path, 0, torch.tensor([3.0]), torch.tensor([-6.0]))

    assert got.item() == pytest.approx(5.0, abs=1e-5)
    assert bool(got_low.item())


@pytest.mark.parametrize("sort_by_token_indexes", [None, [1, 3]], ids=["by-max-value", "by-sort-tokens"])
def test_ranked_streaming_matches_the_global_sort(tmp_path: Path, sort_by_token_indexes: list[int] | None):
    """Per-source reduction stays exact when the key is the rarity of each row's rarer extreme."""
    quantiles, table_max, table_min = _random_table(len(SOURCES), N_FEATURES, seed=1)
    path = _write_table(tmp_path, quantiles, table_max, table_min)
    blocks = _activations()
    request = _request(6, sort_by_token_indexes=sort_by_token_indexes)
    request.rank_by_percentile = True

    streamed = _stream(blocks, request, act_range_path=path)

    expected = _reference_global_sort(
        blocks, request, rank_key=lambda layer, highest, lowest: neuron_rarity(path, layer, highest, lowest)
    )
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


def test_a_rare_negative_activation_ranks_and_is_flagged(tmp_path: Path):
    """A neuron at its 0.001st percentile on a sort token outranks one at its 99th, and its row says so."""
    quantiles = torch.zeros(len(SOURCES), N_FEATURES, len(LEVELS))
    symmetric = torch.tensor([float(i - 6) for i in range(len(LEVELS))])
    quantiles[1, 2] = symmetric
    quantiles[0, 7] = symmetric
    table_max = torch.full((len(SOURCES), N_FEATURES), 20.0)
    path = _write_table(tmp_path, quantiles, table_max)
    blocks = [torch.zeros(N_FEATURES, N_TOKENS) for _ in SOURCES]
    blocks[1][2, 1], blocks[1][2, 3] = 0.5, -6.0
    blocks[0][7, 3] = 3.0
    request = _request(2, sort_by_token_indexes=[1, 3])
    request.rank_by_percentile = True

    ranked = _stream(blocks, request, act_range_path=path)

    assert [(r[0], r[1]) for r in ranked] == [(1, 2), (0, 7)]
    assert ranked[0][5] == pytest.approx(5.0, abs=1e-5)
    assert ranked[1][5] == pytest.approx(2.0, abs=1e-5)
    assert [r[6] for r in ranked] == [1.0, 0.0]


def test_the_response_says_which_side_ranked_each_neuron():
    """rankedByLowest is set from the row with rankByPercentile, and left out without it."""
    request = _request(1)
    request.source_set = "neurons"
    rows = [[0.0, 0.0, 0.5, 1.0, 0.0, 5.0, 1.0, 0.0, 0.5, -6.0]]
    tokens = torch.zeros(3, dtype=torch.long)
    sae_manager = MagicMock()
    sae_manager.is_dfa_enabled.return_value = False

    with patch(f"{MODULE}.SAEManager.get_instance", return_value=sae_manager):
        request.rank_by_percentile = True
        ranked = asyncio.run(ActivationProcessor()._format_result_and_calculate_dfa(rows, MagicMock(), tokens, request))
        request.rank_by_percentile = False
        raw = asyncio.run(ActivationProcessor()._format_result_and_calculate_dfa(rows, MagicMock(), tokens, request))

    assert ranked[0].ranked_by_lowest is True
    assert ranked[0].values == [0.0, 0.5, -6.0]
    assert raw[0].ranked_by_lowest is None
    assert "rankedByLowest" not in raw[0].model_dump(exclude_none=True)


def test_a_near_zero_neuron_needs_the_layer_floor_to_score(tmp_path: Path):
    """A dead neuron's knots are raised to 5% of the layer's median extreme on each side, so a
    small activation of either sign is not rare for it."""
    quantiles = torch.zeros(1, 5, len(LEVELS))
    quantiles[0, 1:] = torch.linspace(-3.0, 3.0, len(LEVELS))
    table_max = torch.tensor([[1e-6, 4.0, 4.0, 4.0, 4.0]])
    table_min = torch.tensor([[-1e-6, -8.0, -8.0, -8.0, -8.0]])
    path = _write_table(tmp_path, quantiles, table_max, table_min)
    high_floor = 4.0 * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN
    low_floor = 8.0 * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN
    zeros = torch.zeros(4)

    def rarity(highest: float, lowest: float) -> float:
        return neuron_rarity(
            path, 0, torch.cat((torch.tensor([highest]), zeros)), torch.cat((torch.tensor([lowest]), zeros))
        )[0][0].item()

    assert rarity(high_floor * 0.9, -low_floor * 0.9) == 0.0
    assert rarity(high_floor * 2, 0.0) == pytest.approx(TOP_RARITY + math.log10(2.0), abs=1e-5)
    assert rarity(0.0, -low_floor * 2) == pytest.approx(TOP_RARITY + math.log10(2.0), abs=1e-5)


@pytest.mark.parametrize(
    ("layer", "d_mlp", "message"),
    [(3, N_FEATURES, "layer 3 is out of range"), (0, N_FEATURES + 1, "neurons per layer")],
    ids=["layer-out-of-range", "width-mismatch"],
)
def test_a_table_that_does_not_fit_the_model_is_refused(tmp_path: Path, layer: int, d_mlp: int, message: str):
    quantiles, table_max, table_min = _random_table(len(SOURCES), N_FEATURES, seed=2)
    path = _write_table(tmp_path, quantiles, table_max, table_min)
    with pytest.raises(NeuronActRangeError, match=message):
        neuron_rarity(path, layer, torch.zeros(d_mlp), torch.zeros(d_mlp))


def test_no_table_is_refused():
    with pytest.raises(NeuronActRangeError, match="NEURON_ACT_RANGE_PATH"):
        neuron_rarity(None, 0, torch.zeros(N_FEATURES), torch.zeros(N_FEATURES))


def test_a_max_only_table_is_refused(tmp_path: Path):
    """A table from before quantiles were added names what it lacks."""
    path = tmp_path / "act_range.safetensors"
    save_file({"max": torch.ones(1, N_FEATURES), "min": -torch.ones(1, N_FEATURES)}, str(path))
    load_neuron_rarity_table.cache_clear()
    with pytest.raises(NeuronActRangeError, match="quantile_levels"):
        neuron_rarity(str(path), 0, torch.zeros(N_FEATURES), torch.zeros(N_FEATURES))


@pytest.mark.parametrize("contents", [None, b"not a safetensors file"], ids=["missing", "corrupt"])
def test_an_unreadable_table_is_refused(tmp_path: Path, contents: bytes | None):
    """A table lost from the pod's disk gives a 400 with the reason, not a 500."""
    path = tmp_path / "act_range.safetensors"
    if contents is not None:
        path.write_bytes(contents)
    load_neuron_rarity_table.cache_clear()
    with pytest.raises(NeuronActRangeError, match="Cannot read"):
        neuron_rarity(str(path), 0, torch.zeros(N_FEATURES), torch.zeros(N_FEATURES))


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
