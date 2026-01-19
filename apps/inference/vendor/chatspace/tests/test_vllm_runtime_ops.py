"""Unit tests for steering runtime helper operations."""

from __future__ import annotations

import torch

from chatspace.vllm_steering import runtime as steering_runtime


def test_apply_projection_cap_enforces_bounds():
    unit = torch.tensor([1.0, 0.0], dtype=torch.float32)
    config = steering_runtime._ProjectionCapConfig(  # type: ignore[attr-defined]
        unit_vector=unit,
        min=-1.0,
        max=1.5,
    )
    hidden = torch.tensor([[2.0, 3.0], [-5.0, 1.0]], dtype=torch.float32)

    capped = steering_runtime._apply_projection_cap(hidden, config)  # type: ignore[attr-defined]

    expected = torch.tensor([[1.5, 3.0], [-1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(capped, expected)


def test_apply_projection_cap_noop_when_within_bounds():
    unit = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    config = steering_runtime._ProjectionCapConfig(  # type: ignore[attr-defined]
        unit_vector=unit,
        min=-2.0,
        max=2.0,
    )
    hidden = torch.tensor([[0.0, 1.0, -3.0]], dtype=torch.float32)

    capped = steering_runtime._apply_projection_cap(hidden, config)  # type: ignore[attr-defined]

    assert torch.allclose(capped, hidden)


def test_apply_ablation_scales_component():
    unit = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    config = steering_runtime._AblationConfig(  # type: ignore[attr-defined]
        unit_vector=unit,
        scale=0.5,
    )
    hidden = torch.tensor([[2.0, 3.0, -1.0], [-4.0, 5.0, 0.5]], dtype=torch.float32)

    ablated = steering_runtime._apply_ablation(hidden, config)  # type: ignore[attr-defined]

    expected = torch.tensor([[1.0, 3.0, -1.0], [-2.0, 5.0, 0.5]], dtype=torch.float32)
    assert torch.allclose(ablated, expected)
