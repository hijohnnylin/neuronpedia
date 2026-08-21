"""Decoder cosine similarity must not copy the decoder, and must still rank the same.

`get_top_k_by_decoder_cosine_similarity` used to drop NaN rows with
`W_dec[~isnan(W_dec).any(dim=1)]`. Boolean-mask indexing copies, so serving one request
allocated a full-size bool temporary plus an entire second copy of W_dec -- roughly 11 GiB
for a 1M-feature SAE at d_model 2304, to produce one scalar per feature. It now contracts
straight to [n_features]. These tests pin the ranking (against the masked formulation) and
the NaN exclusion.

CPU-only synthetic decoders: no model, no SAE weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from neuronpedia_inference.endpoints.util.sae_topk_by_decoder_cossim import (
    get_top_k_by_decoder_cosine_similarity,
)
from neuronpedia_inference.sae_manager import SAE_TYPE

MODULE = "neuronpedia_inference.endpoints.util.sae_topk_by_decoder_cossim"

SOURCE = "5-res-test"
MODEL = "gpt2-small"
N_FEATURES = 32
D_MODEL = 8


def _with_decoder(W_dec: torch.Tensor):
    sae = MagicMock()
    sae.W_dec = W_dec
    sae_manager = MagicMock()
    sae_manager.sae_data = {SOURCE: {"type": SAE_TYPE.SAELENS, "sae": sae}}
    # get_sae rather than a sae_data read: with SAE paging the weights may be on the host
    # until the cache stages them in, so the endpoint has to go through the manager.
    sae_manager.get_sae_type.side_effect = lambda s: SAE_TYPE.SAELENS if s == SOURCE else None
    sae_manager.get_sae.side_effect = lambda s: sae if s == SOURCE else None
    return patch(f"{MODULE}.SAEManager.get_instance", return_value=sae_manager)


def _masked_reference(W_dec: torch.Tensor, query: torch.Tensor, num_results: int) -> tuple[list[int], list[float]]:
    """The original: mask NaN rows out (copying W_dec), then cosine_similarity + topk."""
    valid_mask = ~torch.isnan(W_dec).any(dim=1)
    valid_W_dec = W_dec[valid_mask]
    sims = torch.nn.functional.cosine_similarity(query.unsqueeze(0), valid_W_dec)
    full = torch.full((W_dec.shape[0],), float("-inf"), dtype=sims.dtype)
    full[valid_mask] = sims
    values, indices = torch.topk(full, k=num_results)
    return indices.tolist(), values.tolist()


def _top_k(W_dec: torch.Tensor, query: torch.Tensor, num_results: int):
    with _with_decoder(W_dec):
        return get_top_k_by_decoder_cosine_similarity(SOURCE, MODEL, query, num_results)


def _decoder(*, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(N_FEATURES, D_MODEL, generator=generator)


def test_ranking_matches_the_masked_formulation():
    W_dec = _decoder()
    query = W_dec[7].clone()

    results = _top_k(W_dec, query, 5)

    expected_indices, expected_values = _masked_reference(W_dec, query, 5)
    assert [r.feature.index for r in results if r.feature is not None] == expected_indices
    assert [r.cosine_similarity for r in results] == pytest.approx(expected_values, abs=1e-5)


def test_a_feature_is_its_own_nearest_neighbour():
    W_dec = _decoder()
    query = W_dec[7].clone()

    results = _top_k(W_dec, query, 3)

    assert results[0].feature is not None
    assert results[0].feature.index == 7
    assert results[0].cosine_similarity == pytest.approx(1.0, abs=1e-5)
    assert results[0].feature.source == SOURCE
    assert results[0].feature.model == MODEL


def test_nan_decoder_rows_never_outrank_a_real_one():
    """A row with any NaN is demoted below every valid row, whether the NaN is the whole
    row or a single component."""
    W_dec = _decoder()
    W_dec[3] = float("nan")
    W_dec[11, 0] = float("nan")
    query = W_dec[7].clone()

    top = [r.feature.index for r in _top_k(W_dec, query, 5) if r.feature is not None]
    assert 3 not in top
    assert 11 not in top

    # Asking for every feature still returns every feature -- the demoted ones just sort to
    # the bottom, which is what the masked version did too.
    full_ranking = _top_k(W_dec, query, N_FEATURES)
    tail = [r.feature.index for r in full_ranking[-2:] if r.feature is not None]
    assert tail == [3, 11] or tail == [11, 3]
    # -inf is reported as 0.0: JSON cannot carry an infinity, which is why the caller
    # squashes non-finite similarities.
    assert full_ranking[-1].cosine_similarity == 0.0
    assert full_ranking[-2].cosine_similarity == 0.0


def test_num_results_is_clamped_to_the_feature_count():
    """torch.topk raises when k exceeds the axis; the request used to 500 instead."""
    W_dec = _decoder()

    results = _top_k(W_dec, W_dec[0].clone(), N_FEATURES + 50)

    assert len(results) == N_FEATURES


def test_bfloat16_decoder_is_ranked_in_fp32():
    """Norms and dots accumulate in fp32 even for a bf16 decoder, so near-ties are ordered
    by their real values rather than by bf16 rounding."""
    W_dec = _decoder().to(torch.bfloat16)
    query = W_dec[7].clone()

    results = _top_k(W_dec, query, 4)

    assert results[0].feature is not None
    assert results[0].feature.index == 7
    assert results[0].cosine_similarity == pytest.approx(1.0, abs=1e-2)
    assert all(isinstance(r.cosine_similarity, float) for r in results)
