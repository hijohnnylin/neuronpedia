"""activation_token_limit is derived from the measured budget + widest SAE dims."""

from __future__ import annotations

from neuronpedia_inference.startup_memory import (
    FLOOR_MAX_TOKENS,
    compute_activation_token_limit,
)


def test_unchanged_when_budget_already_fits_token_limit():
    # Tiny SAE, generous budget: leave the pods.yaml token_limit alone.
    assert (
        compute_activation_token_limit(
            budget_bytes=8 * 1024**3,
            token_limit=550,
            d_sae=16_384,
            d_in=2304,
            n_hooks=26,
            sae_dtype="bfloat16",
            model_dtype="bfloat16",
        )
        == 550
    )


def test_shrinks_for_a_wide_sae_on_a_tight_budget():
    # 1M-feature SAE at bf16: ~4 MB/token just for the encode. A 1 GiB budget cannot
    # hold the default 550-token cap.
    capped = compute_activation_token_limit(
        budget_bytes=1 * 1024**3,
        token_limit=550,
        d_sae=1_000_000,
        d_in=2304,
        n_hooks=26,
        sae_dtype="bfloat16",
        model_dtype="bfloat16",
    )
    assert FLOOR_MAX_TOKENS <= capped < 550


def test_never_raises_above_token_limit():
    assert (
        compute_activation_token_limit(
            budget_bytes=100 * 1024**3,
            token_limit=256,
            d_sae=16_384,
            d_in=2304,
            n_hooks=1,
            sae_dtype="bfloat16",
            model_dtype="bfloat16",
        )
        == 256
    )


def test_no_budget_or_dims_leaves_token_limit():
    assert (
        compute_activation_token_limit(
            budget_bytes=0,
            token_limit=550,
            d_sae=65_536,
            d_in=2304,
            n_hooks=26,
            sae_dtype="bfloat16",
            model_dtype="bfloat16",
        )
        == 550
    )
    assert (
        compute_activation_token_limit(
            budget_bytes=4 * 1024**3,
            token_limit=550,
            d_sae=0,
            d_in=0,
            n_hooks=0,
            sae_dtype="bfloat16",
            model_dtype="bfloat16",
        )
        == 550
    )
