"""Unit tests for clock scheduler multi-op steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from steerllm.backends.vllm.runtime import (
    _premerge_adds,
    _ClockSlot,
    _build_clock_schedule,
    _execute_clock_schedule,
    _fused_gather_steering,
    _slow_path_loop_impl,
)


@dataclass
class MockLayerSpec:
    """Mock layer spec for testing."""
    operations: list[tuple[str, torch.Tensor, Any]] = field(default_factory=list)


# =============================================================================
# _premerge_adds tests
# =============================================================================


def test_premerge_adds_empty():
    """Empty list returns empty."""
    assert _premerge_adds([]) == []


def test_premerge_adds_single_add():
    """Single add is unchanged."""
    v = torch.randn(4)
    ops = [("add", v, None)]
    result = _premerge_adds(ops)
    assert len(result) == 1
    assert result[0][0] == "add"
    assert torch.equal(result[0][1], v)


def test_premerge_adds_consecutive_adds():
    """Consecutive adds are merged."""
    v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 2.0, 0.0, 0.0])
    v3 = torch.tensor([0.0, 0.0, 3.0, 0.0])
    ops = [("add", v1, None), ("add", v2, None), ("add", v3, None)]

    result = _premerge_adds(ops)

    assert len(result) == 1
    assert result[0][0] == "add"
    expected = torch.tensor([1.0, 2.0, 3.0, 0.0])
    assert torch.allclose(result[0][1], expected)


def test_premerge_adds_non_consecutive():
    """Non-consecutive adds are not merged."""
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])
    cap_vec = torch.tensor([1.0, 0.0])
    ops = [
        ("add", v1, None),
        ("cap", cap_vec, (-1.0, 1.0)),
        ("add", v2, None),
    ]

    result = _premerge_adds(ops)

    assert len(result) == 3
    assert result[0][0] == "add"
    assert result[1][0] == "cap"
    assert result[2][0] == "add"


def test_premerge_adds_mixed_sequence():
    """Complex sequence with multiple merge opportunities."""
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])
    v3 = torch.tensor([0.5, 0.5])
    cap_vec = torch.tensor([1.0, 0.0])
    ablate_vec = torch.tensor([0.0, 1.0])

    ops = [
        ("add", v1, None),
        ("add", v2, None),  # Merge with v1
        ("cap", cap_vec, (-1.0, 1.0)),
        ("ablation", ablate_vec, 0.5),
        ("add", v3, None),  # Standalone
    ]

    result = _premerge_adds(ops)

    assert len(result) == 4
    assert result[0][0] == "add"
    assert torch.allclose(result[0][1], v1 + v2)
    assert result[1][0] == "cap"
    assert result[2][0] == "ablation"
    assert result[3][0] == "add"


# =============================================================================
# _build_clock_schedule tests
# =============================================================================


def test_build_schedule_empty():
    """Empty specs produce empty schedule."""
    schedule = _build_clock_schedule([])
    assert schedule == []


def test_build_schedule_all_none():
    """All None specs produce empty schedule."""
    schedule = _build_clock_schedule([None, None, None])
    assert schedule == []


def test_build_schedule_single_op_per_request():
    """Single op per request creates single slot."""
    v1 = torch.randn(4)
    v2 = torch.randn(4)
    specs = [
        MockLayerSpec(operations=[("add", v1, None)]),
        MockLayerSpec(operations=[("add", v2, None)]),
    ]

    schedule = _build_clock_schedule(specs)

    assert len(schedule) == 1
    assert len(schedule[0].add_vecs) == 2
    assert schedule[0].add_request_mask == [True, True]


def test_build_schedule_mixed_single_ops():
    """Mixed single ops create single slot with all phases."""
    add_vec = torch.randn(4)
    cap_vec = torch.randn(4)
    ablate_vec = torch.randn(4)

    specs = [
        MockLayerSpec(operations=[("add", add_vec, None)]),
        MockLayerSpec(operations=[("cap", cap_vec, (-1.0, 1.0))]),
        MockLayerSpec(operations=[("ablation", ablate_vec, 0.5)]),
    ]

    schedule = _build_clock_schedule(specs)

    assert len(schedule) == 1
    slot = schedule[0]
    assert len(slot.add_vecs) == 1
    assert len(slot.cap_vecs) == 1
    assert len(slot.ablate_vecs) == 1
    assert slot.add_request_mask == [True, False, False]
    assert slot.cap_request_mask == [False, True, False]
    assert slot.ablate_request_mask == [False, False, True]


def test_build_schedule_multi_op():
    """Multiple ops per request create multiple slots."""
    v1 = torch.randn(4)
    v2 = torch.randn(4)
    cap_vec = torch.randn(4)

    specs = [
        MockLayerSpec(operations=[("add", v1, None), ("cap", cap_vec, (-1.0, 1.0))]),
        MockLayerSpec(operations=[("add", v2, None)]),
    ]

    schedule = _build_clock_schedule(specs)

    # Slot 0: both adds, req 0's cap
    # Slot 1: nothing more (req 0 has no more ops after cap scheduled in slot 0)
    # Wait, greedy schedules ALL ready ops per slot
    # So slot 0: add(req0), add(req1)
    # Then slot 1: cap(req0)
    # But actually the greedy schedules all ready ops at once...
    # Let me re-check: req0 has [add, cap], req1 has [add]
    # Slot 0: req0's first op is add, req1's first op is add → both adds scheduled
    # After slot 0: req0 cursor=1 (cap next), req1 cursor=1 (done)
    # Slot 1: req0's cursor points to cap → cap scheduled

    assert len(schedule) == 2
    assert len(schedule[0].add_vecs) == 2  # Both adds
    assert len(schedule[1].cap_vecs) == 1  # Just the cap


def test_build_schedule_variable_length():
    """Requests with different numbers of ops."""
    v1 = torch.randn(4)
    v2 = torch.randn(4)
    v3 = torch.randn(4)
    cap_vec = torch.randn(4)
    ablate_vec = torch.randn(4)

    # A: [add, cap, add] (after merge: [add, cap, add] since not consecutive)
    # Actually wait - [add, cap, add] has non-consecutive adds so no merge
    # B: [ablate, add]
    # C: [add]
    specs = [
        MockLayerSpec(operations=[("add", v1, None), ("cap", cap_vec, (-1.0, 1.0)), ("add", v2, None)]),
        MockLayerSpec(operations=[("ablation", ablate_vec, 0.5), ("add", v3, None)]),
        MockLayerSpec(operations=[("add", v1, None)]),
    ]

    schedule = _build_clock_schedule(specs)

    # Slot 0: A's add, B's ablate, C's add
    # Slot 1: A's cap, B's add
    # Slot 2: A's add
    assert len(schedule) == 3


# =============================================================================
# Correctness tests: clock scheduler vs slow path
# =============================================================================


@pytest.fixture
def hidden_setup():
    """Create test hidden states."""
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 8
    hidden_size = 16
    total_tokens = batch_size * seq_len
    hidden = torch.randn(total_tokens, hidden_size, dtype=torch.float32)
    seq_lens = [seq_len] * batch_size
    return hidden, seq_lens, batch_size, hidden_size


def test_clock_scheduler_matches_slow_path_single_add(hidden_setup):
    """Clock scheduler matches slow path for single add ops."""
    hidden, seq_lens, batch_size, hidden_size = hidden_setup

    vec = torch.randn(hidden_size)
    specs = [MockLayerSpec(operations=[("add", vec, None)]) for _ in range(batch_size)]

    h_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs)
    h_fused = _fused_gather_steering(hidden.clone(), seq_lens, specs)

    assert torch.allclose(h_slow, h_fused, atol=1e-5)


def test_clock_scheduler_matches_slow_path_multi_op(hidden_setup):
    """Clock scheduler matches slow path for multi-op case."""
    hidden, seq_lens, batch_size, hidden_size = hidden_setup

    add_vec = torch.randn(hidden_size)
    cap_vec = torch.randn(hidden_size)
    cap_vec = cap_vec / cap_vec.norm()  # Normalize for cap

    specs = [
        MockLayerSpec(operations=[("add", add_vec, None), ("cap", cap_vec, (-0.5, 0.5))]),
        MockLayerSpec(operations=[("add", add_vec, None)]),
        MockLayerSpec(operations=[("cap", cap_vec, (-0.5, 0.5))]),
        MockLayerSpec(operations=[("add", add_vec, None), ("cap", cap_vec, (-0.5, 0.5))]),
    ]

    h_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs)
    h_fused = _fused_gather_steering(hidden.clone(), seq_lens, specs)

    assert torch.allclose(h_slow, h_fused, atol=1e-5)


def test_clock_scheduler_matches_slow_path_all_ops(hidden_setup):
    """Clock scheduler matches slow path with add, cap, and ablation."""
    hidden, seq_lens, batch_size, hidden_size = hidden_setup

    add_vec = torch.randn(hidden_size)
    cap_vec = torch.randn(hidden_size)
    cap_vec = cap_vec / cap_vec.norm()
    ablate_vec = torch.randn(hidden_size)
    ablate_vec = ablate_vec / ablate_vec.norm()

    specs = [
        MockLayerSpec(operations=[("add", add_vec, None), ("ablation", ablate_vec, 0.5)]),
        MockLayerSpec(operations=[("ablation", ablate_vec, 0.5), ("add", add_vec, None)]),
        MockLayerSpec(operations=[("cap", cap_vec, (-0.5, 0.5)), ("add", add_vec, None)]),
        MockLayerSpec(operations=[("add", add_vec, None), ("cap", cap_vec, (-0.5, 0.5)), ("ablation", ablate_vec, 0.5)]),
    ]

    h_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs)
    h_fused = _fused_gather_steering(hidden.clone(), seq_lens, specs)

    assert torch.allclose(h_slow, h_fused, atol=1e-5)


def test_clock_scheduler_preserves_op_order(hidden_setup):
    """Verify operation order is preserved (add then cap != cap then add)."""
    hidden, seq_lens, batch_size, hidden_size = hidden_setup

    # Use specific vectors where order matters
    add_vec = torch.ones(hidden_size) * 0.5
    cap_vec = torch.zeros(hidden_size)
    cap_vec[0] = 1.0  # Cap along first dimension

    # Order 1: add then cap
    specs1 = [
        MockLayerSpec(operations=[("add", add_vec, None), ("cap", cap_vec, (-0.1, 0.1))]),
    ] * batch_size

    # Order 2: cap then add
    specs2 = [
        MockLayerSpec(operations=[("cap", cap_vec, (-0.1, 0.1)), ("add", add_vec, None)]),
    ] * batch_size

    h1 = _fused_gather_steering(hidden.clone(), seq_lens, specs1)
    h2 = _fused_gather_steering(hidden.clone(), seq_lens, specs2)

    # Results should differ because cap and add don't commute
    assert not torch.allclose(h1, h2, atol=1e-5)

    # But each should match slow path
    h1_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs1)
    h2_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs2)

    assert torch.allclose(h1, h1_slow, atol=1e-5)
    assert torch.allclose(h2, h2_slow, atol=1e-5)


def test_add_merge_correctness(hidden_setup):
    """Verify consecutive adds are correctly merged."""
    hidden, seq_lens, batch_size, hidden_size = hidden_setup

    v1 = torch.randn(hidden_size)
    v2 = torch.randn(hidden_size)
    v3 = torch.randn(hidden_size)

    # Three consecutive adds
    specs_separate = [
        MockLayerSpec(operations=[("add", v1, None), ("add", v2, None), ("add", v3, None)]),
    ] * batch_size

    # Single merged add
    specs_merged = [
        MockLayerSpec(operations=[("add", v1 + v2 + v3, None)]),
    ] * batch_size

    h_separate = _fused_gather_steering(hidden.clone(), seq_lens, specs_separate)
    h_merged = _fused_gather_steering(hidden.clone(), seq_lens, specs_merged)

    assert torch.allclose(h_separate, h_merged, atol=1e-5)


def test_non_uniform_seq_lens(hidden_setup):
    """Test with non-uniform sequence lengths."""
    _, _, _, hidden_size = hidden_setup

    torch.manual_seed(42)
    seq_lens = [4, 8, 6, 10]
    total_tokens = sum(seq_lens)
    hidden = torch.randn(total_tokens, hidden_size, dtype=torch.float32)

    add_vec = torch.randn(hidden_size)
    cap_vec = torch.randn(hidden_size)
    cap_vec = cap_vec / cap_vec.norm()

    specs = [
        MockLayerSpec(operations=[("add", add_vec, None), ("cap", cap_vec, (-0.5, 0.5))]),
        MockLayerSpec(operations=[("add", add_vec, None)]),
        MockLayerSpec(operations=[("cap", cap_vec, (-0.5, 0.5)), ("add", add_vec, None)]),
        MockLayerSpec(operations=[("add", add_vec, None)]),
    ]

    h_slow = _slow_path_loop_impl(hidden.clone(), seq_lens, specs)
    h_fused = _fused_gather_steering(hidden.clone(), seq_lens, specs)

    assert torch.allclose(h_slow, h_fused, atol=1e-5)
