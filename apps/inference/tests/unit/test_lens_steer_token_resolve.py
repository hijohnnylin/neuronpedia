"""Steer and swap token strings resolve by exact match only, with a closest-token hint.

A near match (for example ``"ants"`` -> ``" ants"``) is not used without notice, because the
model then steers on a token the user did not name. A miss raises ``SteerTokenNotFound`` with
the longest vocab entry that is a prefix of the input, which the webapp offers to apply.

Model-free: the reverse index is a plain dict, as ``_decoded_string_to_ids`` builds it.
"""

from __future__ import annotations

import pytest

from neuronpedia_inference.endpoints.lens.prompt import (
    _MAX_SUGGEST_PREFIX_CHARS,
    SteerTokenNotFound,
    _resolve_steer_token_id,
    _suggest_steer_token,
)

INDEX: dict[str, list[int]] = {
    " ants": [10],
    "ant": [11],
    " ant": [12],
    "anti": [13],
    " cat": [20, 21],
    "cat": [22],
    "\n": [30],
}


def test_exact_match_resolves_to_lowest_id() -> None:
    """An exact string resolves, and a collision takes the lowest id."""
    assert _resolve_steer_token_id(INDEX, " cat") == 20
    assert _resolve_steer_token_id(INDEX, "cat") == 22


def test_near_match_is_not_used() -> None:
    """``"ants"`` is not in the index, so it fails even though ``" ants"`` is."""
    with pytest.raises(SteerTokenNotFound) as info:
        _resolve_steer_token_id(INDEX, "ants")
    assert info.value.token == "ants"
    assert info.value.suggestion == " ants"
    assert "' ants'" in str(info.value)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        # A leading space gives a longer match than the bare form.
        ("ants", " ants"),
        # Longest prefix of a multi-token word.
        ("antidisestablishment", "anti"),
        # A leading space the user typed is kept, even when "anti" is longer.
        (" antidisestablishment", " ant"),
        # A trailing space is dropped by the prefix search.
        (" ants ", " ants"),
        # An extra leading space is dropped.
        ("  ants", " ants"),
        # On a tie, the form the user typed wins.
        ("anthem", "ant"),
        (" anthem", " ant"),
    ],
)
def test_suggestion_is_longest_prefix(token: str, expected: str) -> None:
    """The suggestion is the vocab entry that covers the most of the input."""
    assert _suggest_steer_token(INDEX, token) == expected


@pytest.mark.parametrize("token", ["", "   ", "zebra", " zebra"])
def test_no_suggestion(token: str) -> None:
    """No suggestion for whitespace, or when no prefix is in the index."""
    assert _suggest_steer_token(INDEX, token) is None
    with pytest.raises(SteerTokenNotFound) as info:
        _resolve_steer_token_id(INDEX, token)
    assert info.value.suggestion is None
    assert "Closest token" not in str(info.value)


def test_long_input_is_bounded() -> None:
    """A very long input still finds a short prefix, with a bounded search."""
    token = "ant" + "x" * (_MAX_SUGGEST_PREFIX_CHARS * 40)
    assert _suggest_steer_token(INDEX, token) == "ant"
