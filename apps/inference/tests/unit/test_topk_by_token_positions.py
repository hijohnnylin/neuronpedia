"""`token_position` must index the prompt, not what survived the `ignore_bos` slice.

The topk-by-token endpoints prepend a BOS and then, when `ignore_bos` is set, drop
position 0 from the tokens and both top-K tensors. They used to number the results
by their index in the *sliced* list, so every reported position was one too low --
and `is_special`, computed from the unsliced token ids, would inherit that skew if
it were applied the same way.

`build_token_results` is shared by the batch and non-batch endpoints, so pinning it
here covers both. CPU-only synthetic tensors: no model, no SAE weights.
"""

from __future__ import annotations

import torch

from neuronpedia_inference.endpoints.activation.topk_by_token import (
    build_token_results,
)

STR_TOKENS = ["<bos>", "the", "capital", "<eos>"]
# Two features per token, values irrelevant to positioning.
TOP_K_VALUES = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
TOP_K_INDICES = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]])
# As `special_token_positions` would report them for the full tokenization.
SPECIAL_POSITIONS = {0, 3}


def _build(ignore_bos: bool):
    return build_token_results(
        list(STR_TOKENS),
        TOP_K_VALUES,
        TOP_K_INDICES,
        SPECIAL_POSITIONS,
        ignore_bos,
    )


def test_positions_are_prompt_indices_when_bos_is_kept():
    tokens, results = _build(ignore_bos=False)
    assert tokens == STR_TOKENS
    assert [r.token_position for r in results] == [0, 1, 2, 3]
    assert [r.token for r in results] == STR_TOKENS


def test_positions_stay_prompt_indices_when_bos_is_dropped():
    # The regression: these used to come back as 0, 1, 2 -- indices into the
    # slice rather than into the prompt, so every one pointed at the wrong token.
    tokens, results = _build(ignore_bos=True)
    assert tokens == ["the", "capital", "<eos>"]
    assert [r.token_position for r in results] == [1, 2, 3]
    assert [r.token for r in results] == ["the", "capital", "<eos>"]


def test_special_flags_follow_the_same_positions():
    _, kept = _build(ignore_bos=False)
    assert [r.is_special for r in kept] == [True, False, False, True]

    # A trailing EOS survives `ignore_bos`, which only drops position 0 -- that
    # is exactly why callers need the flag rather than the slice.
    _, dropped = _build(ignore_bos=True)
    assert [r.is_special for r in dropped] == [False, False, True]


def test_top_features_stay_with_their_token():
    _, results = _build(ignore_bos=True)
    assert [f.feature_index for f in results[0].top_features] == [20, 21]
    assert [f.activation_value for f in results[0].top_features] == [
        TOP_K_VALUES[1][0],
        TOP_K_VALUES[1][1],
    ]
