"""Steering operation specifications.

Pure dataclasses describing steering operations with no backend dependencies.
These specs are passed to backends to configure per-request steering behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

import torch


@dataclass
class AddSpec:
    """Additive steering: hidden += vector * scale.

    Parameters
    ----------
    vector :
        Steering direction. Should be L2-normalized for interpretable scaling.
        Stored in float32 for numerical stability.
    scale :
        Magnitude applied to the direction. Zero disables steering.

    Notes
    -----
    The helper validates that vector is finite and non-zero. Supply scale=0
    to disable steering without removing the spec.
    """

    vector: torch.Tensor
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not torch.isfinite(self.vector).all():
            raise ValueError("Steering vector contains NaN or Inf values")
        norm = float(self.vector.norm().item())
        if norm == 0.0:
            raise ValueError("Steering vector has zero norm (cannot be normalized)")

    def clone(self) -> AddSpec:
        """Create a deep copy of this spec."""
        return AddSpec(vector=self.vector.detach().clone(), scale=self.scale)

    def materialize(self) -> torch.Tensor:
        """Return the scaled steering vector ready for application."""
        return (self.vector * self.scale).contiguous()

    @classmethod
    def from_unnormalized(
        cls, vector: torch.Tensor, scale: float = 1.0
    ) -> AddSpec:
        """Create AddSpec from an unnormalized vector.

        Normalizes the vector and incorporates its norm into the scale.

        Parameters
        ----------
        vector :
            Unnormalized steering direction.
        scale :
            Additional scale factor applied after normalization.

        Returns
        -------
        AddSpec
            Spec with normalized vector and adjusted scale.
        """
        norm = float(vector.norm().item())
        if norm == 0.0:
            raise ValueError("Cannot normalize zero vector")
        unit = vector / norm
        return cls(vector=unit.float().contiguous(), scale=norm * scale)


@dataclass
class ProjectionCapSpec:
    """Clamp hidden state projection onto a direction.

    Computes h' = h + (clamp(h @ v, min, max) - h @ v) * v

    Parameters
    ----------
    vector :
        Unit direction for projection. Should be L2-normalized.
    min :
        Optional minimum bound for the projection. None leaves unconstrained.
    max :
        Optional maximum bound for the projection. None leaves unconstrained.

    Raises
    ------
    ValueError
        If neither min nor max is set, or if vector is invalid.
    """

    vector: torch.Tensor
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError(
                "ProjectionCapSpec requires at least one of min or max to be set"
            )
        if not torch.isfinite(self.vector).all():
            raise ValueError("Projection cap vector contains NaN or Inf values")
        norm = float(self.vector.norm().item())
        if norm == 0.0:
            raise ValueError(
                "Projection cap vector has zero norm (cannot be normalized)"
            )

    def clone(self) -> ProjectionCapSpec:
        """Create a deep copy of this spec."""
        return ProjectionCapSpec(
            vector=self.vector.detach().clone(),
            min=self.min,
            max=self.max,
        )

    @classmethod
    def from_unnormalized(
        cls,
        vector: torch.Tensor,
        min: float | None = None,
        max: float | None = None,
    ) -> ProjectionCapSpec:
        """Create ProjectionCapSpec from an unnormalized vector.

        Parameters
        ----------
        vector :
            Unnormalized direction for projection.
        min :
            Optional minimum bound.
        max :
            Optional maximum bound.

        Returns
        -------
        ProjectionCapSpec
            Spec with normalized vector.
        """
        norm = float(vector.norm().item())
        if norm == 0.0:
            raise ValueError("Cannot normalize zero vector")
        unit = vector / norm
        return cls(vector=unit.float().contiguous(), min=min, max=max)


@dataclass
class AblationSpec:
    """Multiplicative ablation along a direction.

    Computes h' = h + (scale - 1) * (h @ v) * v

    With scale=0, this removes the component entirely (full ablation).
    With scale=1, this is a no-op. Values between diminish; values above amplify.

    Parameters
    ----------
    vector :
        Unit direction for ablation. Should be L2-normalized.
    scale :
        Multiplicative factor. 0.0 = full ablation, 1.0 = no change.
    """

    vector: torch.Tensor
    scale: float = 0.0

    def __post_init__(self) -> None:
        if not torch.isfinite(self.vector).all():
            raise ValueError("Ablation vector contains NaN or Inf values")
        norm = float(self.vector.norm().item())
        if norm == 0.0:
            raise ValueError("Ablation vector has zero norm (cannot be normalized)")

    def clone(self) -> AblationSpec:
        """Create a deep copy of this spec."""
        return AblationSpec(vector=self.vector.detach().clone(), scale=self.scale)

    @classmethod
    def from_unnormalized(
        cls, vector: torch.Tensor, scale: float = 0.0
    ) -> AblationSpec:
        """Create AblationSpec from an unnormalized vector.

        Parameters
        ----------
        vector :
            Unnormalized direction for ablation.
        scale :
            Multiplicative factor.

        Returns
        -------
        AblationSpec
            Spec with normalized vector.
        """
        norm = float(vector.norm().item())
        if norm == 0.0:
            raise ValueError("Cannot normalize zero vector")
        unit = vector / norm
        return cls(vector=unit.float().contiguous(), scale=scale)


# Union type for any steering operation
SteeringOp = Union[AddSpec, ProjectionCapSpec, AblationSpec]


@dataclass
class LayerSteeringSpec:
    """All steering operations for a single transformer layer.

    Operations are applied in sequence order, allowing arbitrary interleaving
    of adds, caps, and ablations.

    Parameters
    ----------
    operations :
        Ordered list of steering operations to apply.

    Example
    -------
    >>> LayerSteeringSpec(operations=[
    ...     AddSpec(vector=v1, scale=1.0),
    ...     ProjectionCapSpec(vector=v2, min=-1.0, max=1.0),
    ...     AblationSpec(vector=v3, scale=0.5),
    ... ])
    """

    operations: list[SteeringOp] = field(default_factory=list)

    def clone(self) -> LayerSteeringSpec:
        """Create a deep copy of this spec."""
        return LayerSteeringSpec(operations=[op.clone() for op in self.operations])

    def is_empty(self) -> bool:
        """Check if this spec has no active operations.

        Returns True if operations is empty or all AddSpecs have zero scale.
        """
        if not self.operations:
            return True
        for op in self.operations:
            if isinstance(op, AddSpec):
                scale = float(op.scale)
                if math.isfinite(scale) and not math.isclose(
                    scale, 0.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    return False
            else:
                # ProjectionCapSpec and AblationSpec are always active
                return False
        return True


@dataclass
class SteeringSpec:
    """Complete steering configuration for a generation request.

    Parameters
    ----------
    layers :
        Mapping of layer indices to LayerSteeringSpec instances.

    Example
    -------
    >>> SteeringSpec(layers={
    ...     5: LayerSteeringSpec(operations=[AddSpec(v1, scale=1.0)]),
    ...     10: LayerSteeringSpec(operations=[ProjectionCapSpec(v2, max=0.5)]),
    ... })
    """

    layers: dict[int, LayerSteeringSpec] = field(default_factory=dict)

    def clone(self) -> SteeringSpec:
        """Create a deep copy of this spec."""
        return SteeringSpec(
            layers={layer: spec.clone() for layer, spec in self.layers.items()}
        )

    def is_empty(self) -> bool:
        """Check if this spec has no active operations on any layer."""
        return all(spec.is_empty() for spec in self.layers.values())

    @classmethod
    def simple_add(
        cls, layer: int, vector: torch.Tensor, scale: float = 1.0
    ) -> SteeringSpec:
        """Convenience constructor for single-layer additive steering.

        Parameters
        ----------
        layer :
            Layer index to apply steering.
        vector :
            Steering direction (will be normalized).
        scale :
            Magnitude to apply.

        Returns
        -------
        SteeringSpec
            Spec with a single AddSpec at the specified layer.
        """
        add_spec = AddSpec.from_unnormalized(vector, scale)
        return cls(layers={layer: LayerSteeringSpec(operations=[add_spec])})

    @classmethod
    def simple_cap(
        cls,
        layer: int,
        vector: torch.Tensor,
        min: float | None = None,
        max: float | None = None,
    ) -> SteeringSpec:
        """Convenience constructor for single-layer projection capping.

        Parameters
        ----------
        layer :
            Layer index to apply capping.
        vector :
            Projection direction (will be normalized).
        min :
            Optional minimum bound.
        max :
            Optional maximum bound.

        Returns
        -------
        SteeringSpec
            Spec with a single ProjectionCapSpec at the specified layer.
        """
        cap_spec = ProjectionCapSpec.from_unnormalized(vector, min=min, max=max)
        return cls(layers={layer: LayerSteeringSpec(operations=[cap_spec])})

    @classmethod
    def simple_ablation(
        cls, layer: int, vector: torch.Tensor, scale: float = 0.0
    ) -> SteeringSpec:
        """Convenience constructor for single-layer ablation.

        Parameters
        ----------
        layer :
            Layer index to apply ablation.
        vector :
            Direction to ablate (will be normalized).
        scale :
            Ablation factor (0.0 = full removal, 1.0 = no change).

        Returns
        -------
        SteeringSpec
            Spec with a single AblationSpec at the specified layer.
        """
        abl_spec = AblationSpec.from_unnormalized(vector, scale)
        return cls(layers={layer: LayerSteeringSpec(operations=[abl_spec])})
