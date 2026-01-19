"""Forward hooks for HuggingFace steering."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from steerllm.core.specs import (
    AddSpec,
    AblationSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
)


class ResidualHook(nn.Module):
    """Trainable steering vector module for residual stream injection.

    Parameters
    ----------
    hidden_size :
        Model hidden dimension.
    init_scale :
        Initialization scale. Zero for deterministic start,
        small positive for random initialization.
    """

    def __init__(self, hidden_size: int, init_scale: float = 0.0) -> None:
        super().__init__()
        if init_scale == 0:
            init_tensor = torch.zeros(hidden_size)
        else:
            init_tensor = torch.randn(hidden_size) * init_scale
        self.vector = nn.Parameter(init_tensor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add steering vector to hidden states."""
        return hidden_states + self.vector


def apply_steering_ops(
    hidden_states: torch.Tensor,
    spec: LayerSteeringSpec,
) -> torch.Tensor:
    """Apply a sequence of steering operations to hidden states.

    Parameters
    ----------
    hidden_states :
        Input tensor of shape [..., hidden_size].
    spec :
        Layer steering specification with operations.

    Returns
    -------
    torch.Tensor
        Steered hidden states.
    """
    h = hidden_states

    for op in spec.operations:
        if isinstance(op, AddSpec):
            # Additive steering: h += vector * scale
            vec = op.vector.to(device=h.device, dtype=h.dtype)
            h = h + vec * op.scale

        elif isinstance(op, ProjectionCapSpec):
            # Projection capping: clamp component along direction
            vec = op.vector.to(device=h.device, dtype=h.dtype)
            vec = vec / vec.norm()  # Ensure unit vector

            # Project hidden states onto direction
            proj = (h @ vec).unsqueeze(-1)  # [..., 1]

            # Clamp projection
            clamped = proj.clone()
            if op.min is not None:
                clamped = torch.clamp(clamped, min=op.min)
            if op.max is not None:
                clamped = torch.clamp(clamped, max=op.max)

            # Update hidden states: h' = h + (clamped - proj) * vec
            h = h + (clamped - proj) * vec

        elif isinstance(op, AblationSpec):
            # Ablation: scale component along direction
            vec = op.vector.to(device=h.device, dtype=h.dtype)
            vec = vec / vec.norm()  # Ensure unit vector

            # Project hidden states onto direction
            proj = (h @ vec).unsqueeze(-1)  # [..., 1]

            # Scale the projected component: h' = h + (scale - 1) * proj * vec
            h = h + (op.scale - 1) * proj * vec

    return h


def create_steering_hook(
    spec: LayerSteeringSpec,
) -> Any:
    """Create a forward hook function for steering.

    Parameters
    ----------
    spec :
        Layer steering specification.

    Returns
    -------
    Callable
        Hook function for register_forward_hook().
    """
    def hook_fn(
        module: nn.Module,
        args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        # Handle tuple outputs (hidden_states, *extras)
        if isinstance(output, tuple):
            hidden_states = output[0]
            steered = apply_steering_ops(hidden_states, spec)
            return (steered,) + output[1:]
        else:
            return apply_steering_ops(output, spec)

    return hook_fn


def create_capture_hook(
    captures: dict[str, torch.Tensor],
    capture_key: str = "hidden",
) -> Any:
    """Create a forward hook for capturing activations.

    Parameters
    ----------
    captures :
        Dictionary to store captured activations.
    capture_key :
        Key under which to store the capture.

    Returns
    -------
    Callable
        Hook function for register_forward_hook().
    """
    def hook_fn(
        module: nn.Module,
        args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Clone and detach to avoid holding computation graph
        captures[capture_key] = hidden_states.detach().clone()

    return hook_fn
