"""DFA has to be indexed by the same array the response reports activations in.

`/activation/single` drops the leading BOS from `values` and then reports `max_value_index`
into the trimmed array, while the attention pattern DFA reads is still indexed by the forward
pass. Nothing about that mismatch fails: DFA came back for the position BEFORE the one that
fired, in an array one longer than the tokens it is displayed against, so the webapp named the
neighbouring token as the DFA source. It showed up as `gemma-2-2b/10-gemmascope-att-16k/0`
disagreeing with its own shipping dashboard -- 2.9 attributed to ' .' where the dashboard has
3.6 attributed to the ',' one position earlier.

Synthetic: `calculate_dfa` is stubbed, so this is about the coordinate change alone.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from neuronpedia_inference import engine_adapter
from neuronpedia_inference.engine_adapter import DfaResult, calculate_dfa_for_values

SEQ = 6
FULL_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def _stub_calculate_dfa(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> dict[str, int]:
    """Replace `calculate_dfa` with one that records the destination it was asked for."""
    seen: dict[str, int] = {}

    async def fake(model, sae, layer_num, index, max_value_index, tokens):  # type: ignore[no-untyped-def] # noqa: ANN001
        seen["dest"] = max_value_index
        return DfaResult(
            dfa_values=list(values),
            dfa_target_index=max_value_index,
            dfa_max_value=max(values),
        )

    monkeypatch.setattr(engine_adapter, "calculate_dfa", fake)
    return seen


def _call(max_value_index: int, n_values: int) -> DfaResult:
    return asyncio.run(
        calculate_dfa_for_values(
            object(),
            object(),
            0,
            0,
            max_value_index,
            torch.arange(SEQ),
            n_values=n_values,
        )
    )


def test_a_trimmed_values_array_shifts_the_dfa_destination(monkeypatch: pytest.MonkeyPatch):
    """The bug itself: `max_value_index` names a trimmed row, DFA wants the untrimmed one."""
    seen = _stub_calculate_dfa(monkeypatch, FULL_VALUES)

    result = _call(max_value_index=3, n_values=SEQ - 1)

    assert seen["dest"] == 4
    # ...and reported back in the caller's coordinates, so it still indexes `values`.
    assert result["dfa_target_index"] == 3


def test_dfa_values_come_back_one_per_reported_value(monkeypatch: pytest.MonkeyPatch):
    """The half the webapp sees: `dfaValues[i]` has to be the DFA of token `i`."""
    _stub_calculate_dfa(monkeypatch, FULL_VALUES)

    result = _call(max_value_index=3, n_values=SEQ - 1)

    assert result["dfa_values"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_dfa_max_value_ignores_the_trimmed_bos(monkeypatch: pytest.MonkeyPatch):
    """BOS is an attention sink, so it can hold the largest attribution in the full array.

    Carrying that maximum through would name a position the response no longer contains.
    """
    _stub_calculate_dfa(monkeypatch, [99.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    result = _call(max_value_index=3, n_values=SEQ - 1)

    assert result["dfa_max_value"] == 5.0
    assert result["dfa_max_value"] == max(result["dfa_values"])


def test_an_untrimmed_values_array_is_left_exactly_alone(monkeypatch: pytest.MonkeyPatch):
    """`/activation/single-batch` and `/activation/all` keep BOS; they must not shift."""
    seen = _stub_calculate_dfa(monkeypatch, FULL_VALUES)

    result = _call(max_value_index=3, n_values=SEQ)

    assert seen["dest"] == 3
    assert result["dfa_target_index"] == 3
    assert result["dfa_values"] == FULL_VALUES


def test_more_values_than_tokens_is_refused(monkeypatch: pytest.MonkeyPatch):
    """A negative offset would silently pad rather than trim."""
    _stub_calculate_dfa(monkeypatch, FULL_VALUES)

    with pytest.raises(ValueError, match="longer than its tokens"):
        _call(max_value_index=3, n_values=SEQ + 1)
