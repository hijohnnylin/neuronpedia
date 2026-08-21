"""The streaming top-K in /activation/all must agree with the global sort it replaced.

`_stream_top_activations` reduces one source at a time so peak memory does not grow with the
number of selected sources -- which matters because an empty `selected_sources` expands to
the whole set, and that is the default for the site-wide search box. The rewrite is only
safe if it returns the same rows, in the same order, as materializing every source and
sorting the concatenation. `_reference_global_sort` below is that original algorithm, kept
here as the oracle.

The one deliberate difference: the original sorted with torch's default (unstable) sort, so
its order among EQUAL keys was undefined. Both sides here sort stably, which pins ties to
(source order, feature index) -- the natural row order of the old concatenation.

CPU-only synthetic activations: no model, no SAE weights, cheap enough for CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from neuronpedia_inference.endpoints.activation.all import (
    MAX_NUM_RESULTS,
    ActivationProcessor,
)

MODULE = "neuronpedia_inference.endpoints.activation.all"

SOURCE_SET = "res-test"
SOURCES = ["0-res-test", "1-res-test", "2-res-test"]
N_TOKENS = 7
N_FEATURES = 11


def _activations(*, sparse: bool = False) -> list[torch.Tensor]:
    """One [n_features, n_tokens] block per source.

    `sparse` clamps most values to exactly zero, which is what a real short prompt looks
    like and the case where tie-breaking actually decides the tail of the top-K.
    """
    generator = torch.Generator().manual_seed(0)
    blocks = []
    for _ in SOURCES:
        block = torch.randn(N_FEATURES, N_TOKENS, generator=generator)
        if sparse:
            block = torch.clamp(block - 1.2, min=0.0)
        blocks.append(block)
    return blocks


def _request(
    num_results: int,
    *,
    sort_by_token_indexes: list[int] | None = None,
    ignore_bos: bool = False,
) -> MagicMock:
    request = MagicMock()
    request.selected_sources = list(SOURCES)
    request.source_set = SOURCE_SET
    request.num_results = num_results
    request.sort_by_token_indexes = sort_by_token_indexes or []
    request.ignore_bos = ignore_bos
    return request


def _reference_global_sort(blocks: list[torch.Tensor], request: MagicMock) -> list[list[float]]:
    """The original: every source's rows concatenated, sorted once, then truncated."""
    rows = []
    for layer_num, block in enumerate(blocks):
        activations = block.clone()
        if request.ignore_bos:
            activations[:, 0] = 0
        max_values, max_indices = torch.max(activations, dim=1)
        if request.sort_by_token_indexes:
            sum_values = activations[:, request.sort_by_token_indexes].sum(dim=1)
        else:
            sum_values = torch.zeros_like(max_values)
        n_features = activations.shape[0]
        rows.append(
            torch.cat(
                (
                    torch.full((n_features, 1), float(layer_num)),
                    torch.arange(n_features).unsqueeze(1).to(torch.float32),
                    max_values.unsqueeze(1).to(torch.float32),
                    max_indices.unsqueeze(1).to(torch.float32),
                    sum_values.unsqueeze(1).to(torch.float32),
                    activations.to(torch.float32),
                ),
                dim=1,
            )
        )
    all_rows = torch.cat(rows, dim=0)
    key_col = 4 if request.sort_by_token_indexes else 2
    _, order = torch.sort(all_rows[:, key_col], descending=True, stable=True)
    return all_rows[order][: request.num_results].tolist()


def _stream(blocks: list[torch.Tensor], request: MagicMock) -> list[list[float]]:
    """Run `_stream_top_activations` with the SAE/model/config singletons stubbed out."""
    sae_manager = MagicMock()
    sae_manager.sae_set_to_saes = {SOURCE_SET: list(SOURCES)}
    sae_manager.get_sae_hook.side_effect = lambda source: f"hook-{source}"
    sae_manager.get_sae_type.side_effect = lambda _source: "saelens-1"

    model = MagicMock()
    model.default_prepend_bos = True

    config = MagicMock()
    config.device = "cpu"

    cache = {f"hook-{source}": block for source, block in zip(SOURCES, blocks, strict=True)}

    with (
        patch(f"{MODULE}.SAEManager.get_instance", return_value=sae_manager),
        patch(f"{MODULE}.Model.get_instance", return_value=model),
        patch(f"{MODULE}.Config.get_instance", return_value=config),
        patch.object(
            ActivationProcessor,
            "_get_activations_by_index",
            # The cache holds the post-encode [n_features, n_tokens] block directly, so the
            # stub just hands it back. Cloned because the real code mutates it for ignore_bos.
            side_effect=lambda _type, _source, cache, hook: cache[hook].clone(),
        ),
    ):
        return ActivationProcessor()._stream_top_activations(request, cache, ["tok"] * N_TOKENS)


@pytest.mark.parametrize(
    ("num_results", "sort_by_token_indexes", "ignore_bos", "sparse"),
    [
        (5, None, False, False),
        (5, None, True, False),
        (5, [1, 3], False, False),
        (5, [1, 3], True, False),
        (1, None, False, False),
        # More results than there are rows: exercises the keep = min(...) clamp.
        (N_FEATURES * len(SOURCES) + 10, None, False, False),
        # Mostly-zero activations, so most of the top-K is decided by tie-breaking.
        (12, None, True, True),
        (12, [0, 2], True, True),
    ],
    ids=[
        "by-max-value",
        "by-max-value-ignore-bos",
        "by-sum-values",
        "by-sum-values-ignore-bos",
        "single-result",
        "more-results-than-rows",
        "sparse-ties-by-max-value",
        "sparse-ties-by-sum-values",
    ],
)
def test_streaming_top_k_matches_the_global_sort(
    num_results: int,
    sort_by_token_indexes: list[int] | None,
    ignore_bos: bool,
    sparse: bool,
):
    blocks = _activations(sparse=sparse)
    request = _request(
        num_results,
        sort_by_token_indexes=sort_by_token_indexes,
        ignore_bos=ignore_bos,
    )

    streamed = _stream(blocks, request)

    expected = _reference_global_sort(blocks, request)
    assert len(streamed) == len(expected)
    # Row at a time: pytest.approx does not descend into nested lists.
    for rank, (got, want) in enumerate(zip(streamed, expected, strict=True)):
        assert got == pytest.approx(want), f"row {rank} differs"


def test_a_selected_source_beyond_the_set_listing_is_reduced():
    """resid-post-aa is sparse (3,7,...,27). A selected source must be driven off the
    request alone, never off the configured set listing, which can be stale or filtered:
    deriving anything per-layer from the set is what produced the llama3.1-8b-it search-all
    failure (index 27 out of bounds for dimension 0 with size 20)."""
    set_sources = [
        "3-resid-post-aa",
        "7-resid-post-aa",
        "11-resid-post-aa",
        "15-resid-post-aa",
        "19-resid-post-aa",
    ]
    selected = ["27-resid-post-aa"]
    n_tokens = 19
    block = torch.randn(N_FEATURES, n_tokens)

    request = MagicMock()
    request.selected_sources = list(selected)
    request.source_set = SOURCE_SET
    request.num_results = 5
    request.sort_by_token_indexes = [1, 2, 3]
    request.ignore_bos = True

    sae_manager = MagicMock()
    sae_manager.sae_set_to_saes = {SOURCE_SET: list(set_sources)}
    sae_manager.get_sae_hook.side_effect = lambda source: f"hook-{source}"
    sae_manager.get_sae_type.side_effect = lambda _source: "saelens-1"

    model = MagicMock()
    model.default_prepend_bos = True
    config = MagicMock()
    config.device = "cpu"
    cache = {f"hook-{selected[0]}": block}

    with (
        patch(f"{MODULE}.SAEManager.get_instance", return_value=sae_manager),
        patch(f"{MODULE}.Model.get_instance", return_value=model),
        patch(f"{MODULE}.Config.get_instance", return_value=config),
        patch.object(
            ActivationProcessor,
            "_get_activations_by_index",
            side_effect=lambda _type, _source, cache, hook: cache[hook].clone(),
        ),
    ):
        rows = ActivationProcessor()._stream_top_activations(request, cache, ["tok"] * n_tokens)

    assert len(rows) == request.num_results
    # Column 0 of each row is the layer number the feature came from.
    assert {row[0] for row in rows} == {27}


def test_num_results_is_clamped():
    """An unbounded num_results would size both the result buffer and the JSON payload."""
    blocks = _activations()
    request = _request(MAX_NUM_RESULTS + 500)

    streamed = _stream(blocks, request)

    assert len(streamed) == N_FEATURES * len(SOURCES)
    assert len(streamed) <= MAX_NUM_RESULTS
