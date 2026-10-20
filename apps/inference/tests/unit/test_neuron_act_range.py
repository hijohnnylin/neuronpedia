"""search-all with normalize_by_max: rank neurons by fraction of their corpus max.

CPU-only: a small table on disk and synthetic activations, reusing the streaming top-K harness.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from neuronpedia_inference.endpoints.activation.all import normalize_by_max_error
from neuronpedia_inference.inference_utils.neuron_act_range import (
    MIN_SCALE_FRACTION_OF_LAYER_MEDIAN,
    NeuronActRangeError,
    load_neuron_max_scales,
    neuron_max_scale,
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


def _write_table(tmp_path: Path, table_max: torch.Tensor) -> str:
    """Write a range table and clear the loader cache, so each test reads its own file."""
    path = tmp_path / "act_range.safetensors"
    save_file({"max": table_max.contiguous(), "min": (-table_max).contiguous()}, str(path))
    load_neuron_max_scales.cache_clear()
    return str(path)


@pytest.mark.parametrize("sort_by_token_indexes", [None, [1, 3]], ids=["by-max-value", "by-sum-values"])
def test_normalized_streaming_matches_the_global_sort(tmp_path: Path, sort_by_token_indexes: list[int] | None):
    """Per-source reduction stays exact when every key is divided by its neuron's scale."""
    generator = torch.Generator().manual_seed(1)
    table_max = torch.rand(len(SOURCES), N_FEATURES, generator=generator) * 10 + 0.5
    path = _write_table(tmp_path, table_max)
    blocks = _activations()
    request = _request(6, sort_by_token_indexes=sort_by_token_indexes)
    request.normalize_by_max = True

    streamed = _stream(blocks, request, act_range_path=path)

    expected = _reference_global_sort(blocks, request, scales=load_neuron_max_scales(path))
    assert len(streamed) == len(expected)
    for rank, (got, want) in enumerate(zip(streamed, expected, strict=True)):
        assert got == pytest.approx(want), f"row {rank} differs"


def test_a_small_scale_neuron_outranks_a_large_scale_one(tmp_path: Path):
    """The point of the option: 100% of a small max beats 10% of a large one."""
    table_max = torch.ones(len(SOURCES), N_FEATURES)
    table_max[2, 0] = 100.0
    path = _write_table(tmp_path, table_max)
    blocks = [torch.zeros(N_FEATURES, N_TOKENS) for _ in SOURCES]
    blocks[2][0, 3] = 10.0  # large raw value, 10% of its max
    blocks[0][4, 3] = 1.0  # small raw value, 100% of its max
    request = _request(2)

    raw = _stream(blocks, request)
    request.normalize_by_max = True
    normalized = _stream(blocks, request, act_range_path=path)

    assert [(r[0], r[1]) for r in raw] == [(2, 0), (0, 4)]
    assert [(r[0], r[1]) for r in normalized] == [(0, 4), (2, 0)]
    # The returned activations stay raw.
    assert normalized[1][2] == pytest.approx(10.0)
    assert normalized[1][5] == pytest.approx(0.1)


def test_a_near_zero_max_is_raised_to_the_layer_floor(tmp_path: Path):
    table_max = torch.full((1, 5), 4.0)
    table_max[0, 0] = 0.0
    table_max[0, 1] = 1e-6
    path = _write_table(tmp_path, table_max)

    scale = neuron_max_scale(path, 0, 5)

    floor = 4.0 * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN
    assert scale.tolist() == pytest.approx([floor, floor, 4.0, 4.0, 4.0])


@pytest.mark.parametrize(
    ("layer", "d_mlp", "message"),
    [(3, N_FEATURES, "layer 3 is out of range"), (0, N_FEATURES + 1, "neurons per layer")],
    ids=["layer-out-of-range", "width-mismatch"],
)
def test_a_table_that_does_not_fit_the_model_is_refused(tmp_path: Path, layer: int, d_mlp: int, message: str):
    path = _write_table(tmp_path, torch.ones(len(SOURCES), N_FEATURES))
    with pytest.raises(NeuronActRangeError, match=message):
        neuron_max_scale(path, layer, d_mlp)


def test_no_table_is_refused():
    with pytest.raises(NeuronActRangeError, match="NEURON_ACT_RANGE_PATH"):
        neuron_max_scale(None, 0, N_FEATURES)


@pytest.mark.parametrize("contents", [None, b"not a safetensors file"], ids=["missing", "corrupt"])
def test_an_unreadable_table_is_refused(tmp_path: Path, contents: bytes | None):
    """A table lost from the pod's disk gives a 400 with the reason, not a 500."""
    path = tmp_path / "act_range.safetensors"
    if contents is not None:
        path.write_bytes(contents)
    load_neuron_max_scales.cache_clear()
    with pytest.raises(NeuronActRangeError, match="Cannot read"):
        neuron_max_scale(str(path), 0, N_FEATURES)


@pytest.mark.parametrize(
    ("normalize", "source_set", "path", "status"),
    [
        (False, "res-test", None, None),
        (True, "res-test", "/tables/x.safetensors", 400),
        (True, "neurons", None, 400),
        (True, "neurons", "/tables/x.safetensors", None),
    ],
    ids=["off", "sae-source-set", "no-table", "served"],
)
def test_requests_that_cannot_be_served_get_a_400(normalize: bool, source_set: str, path: str | None, status):
    request = MagicMock()
    request.normalize_by_max = normalize
    request.source_set = source_set
    config = MagicMock()
    config.neuron_act_range_path = path

    with patch(f"{MODULE}.Config.get_instance", return_value=config):
        error = normalize_by_max_error(request)

    assert (error.status_code if error is not None else None) == status
