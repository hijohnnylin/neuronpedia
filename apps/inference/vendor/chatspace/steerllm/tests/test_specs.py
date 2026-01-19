"""Tests for steering specifications."""

import pytest
import torch

from steerllm import (
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
    LayerSteeringSpec,
    SteeringSpec,
)


class TestAddSpec:
    """Tests for AddSpec."""

    def test_basic_creation(self):
        """Test basic AddSpec creation."""
        v = torch.randn(128)
        spec = AddSpec(vector=v, scale=1.0)
        assert spec.scale == 1.0
        assert spec.vector.shape == (128,)

    def test_materialize(self):
        """Test materialization of scaled vector."""
        v = torch.ones(64)
        spec = AddSpec(vector=v, scale=2.0)
        result = spec.materialize()
        assert torch.allclose(result, torch.ones(64) * 2.0)

    def test_clone(self):
        """Test deep cloning."""
        v = torch.randn(32)
        spec = AddSpec(vector=v, scale=1.5)
        cloned = spec.clone()

        assert cloned.scale == spec.scale
        assert torch.allclose(cloned.vector, spec.vector)
        assert cloned.vector is not spec.vector

    def test_from_unnormalized(self):
        """Test creation from unnormalized vector."""
        v = torch.ones(64) * 3.0  # norm = sqrt(64) * 3
        spec = AddSpec.from_unnormalized(v, scale=1.0)

        # Vector should be normalized
        assert torch.isclose(spec.vector.norm(), torch.tensor(1.0), atol=1e-5)
        # Scale should incorporate original norm
        expected_norm = (torch.ones(64) * 3.0).norm().item()
        assert pytest.approx(spec.scale, rel=1e-5) == expected_norm

    def test_validation_nan(self):
        """Test validation rejects NaN values."""
        v = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            AddSpec(vector=v, scale=1.0)

    def test_validation_inf(self):
        """Test validation rejects Inf values."""
        v = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            AddSpec(vector=v, scale=1.0)

    def test_validation_zero_norm(self):
        """Test validation rejects zero vectors."""
        v = torch.zeros(64)
        with pytest.raises(ValueError, match="zero norm"):
            AddSpec(vector=v, scale=1.0)


class TestProjectionCapSpec:
    """Tests for ProjectionCapSpec."""

    def test_basic_creation(self):
        """Test basic creation with min/max."""
        v = torch.randn(128)
        spec = ProjectionCapSpec(vector=v, min=-1.0, max=1.0)
        assert spec.min == -1.0
        assert spec.max == 1.0

    def test_min_only(self):
        """Test creation with only min bound."""
        v = torch.randn(64)
        spec = ProjectionCapSpec(vector=v, min=0.0)
        assert spec.min == 0.0
        assert spec.max is None

    def test_max_only(self):
        """Test creation with only max bound."""
        v = torch.randn(64)
        spec = ProjectionCapSpec(vector=v, max=5.0)
        assert spec.min is None
        assert spec.max == 5.0

    def test_requires_bound(self):
        """Test that at least one bound is required."""
        v = torch.randn(64)
        with pytest.raises(ValueError, match="at least one of min or max"):
            ProjectionCapSpec(vector=v)

    def test_clone(self):
        """Test deep cloning."""
        v = torch.randn(32)
        spec = ProjectionCapSpec(vector=v, min=-0.5, max=0.5)
        cloned = spec.clone()

        assert cloned.min == spec.min
        assert cloned.max == spec.max
        assert torch.allclose(cloned.vector, spec.vector)
        assert cloned.vector is not spec.vector


class TestAblationSpec:
    """Tests for AblationSpec."""

    def test_basic_creation(self):
        """Test basic creation."""
        v = torch.randn(128)
        spec = AblationSpec(vector=v, scale=0.5)
        assert spec.scale == 0.5

    def test_full_ablation(self):
        """Test full ablation (scale=0)."""
        v = torch.randn(64)
        spec = AblationSpec(vector=v, scale=0.0)
        assert spec.scale == 0.0

    def test_clone(self):
        """Test deep cloning."""
        v = torch.randn(32)
        spec = AblationSpec(vector=v, scale=0.3)
        cloned = spec.clone()

        assert cloned.scale == spec.scale
        assert torch.allclose(cloned.vector, spec.vector)
        assert cloned.vector is not spec.vector


class TestLayerSteeringSpec:
    """Tests for LayerSteeringSpec."""

    def test_empty_spec(self):
        """Test empty spec detection."""
        spec = LayerSteeringSpec()
        assert spec.is_empty()

    def test_non_empty_with_add(self):
        """Test non-empty with AddSpec."""
        v = torch.randn(64)
        spec = LayerSteeringSpec(operations=[AddSpec(vector=v, scale=1.0)])
        assert not spec.is_empty()

    def test_empty_with_zero_scale(self):
        """Test empty detection with zero scale AddSpec."""
        v = torch.randn(64)
        spec = LayerSteeringSpec(operations=[AddSpec(vector=v, scale=0.0)])
        assert spec.is_empty()

    def test_non_empty_with_cap(self):
        """Test non-empty with ProjectionCapSpec."""
        v = torch.randn(64)
        spec = LayerSteeringSpec(operations=[ProjectionCapSpec(vector=v, max=1.0)])
        assert not spec.is_empty()

    def test_multi_operation(self):
        """Test multiple operations."""
        v1 = torch.randn(64)
        v2 = torch.randn(64)
        v3 = torch.randn(64)

        spec = LayerSteeringSpec(operations=[
            AddSpec(vector=v1, scale=1.0),
            ProjectionCapSpec(vector=v2, min=-1.0, max=1.0),
            AblationSpec(vector=v3, scale=0.5),
        ])

        assert len(spec.operations) == 3
        assert not spec.is_empty()


class TestSteeringSpec:
    """Tests for SteeringSpec."""

    def test_empty_spec(self):
        """Test empty spec."""
        spec = SteeringSpec()
        assert spec.is_empty()

    def test_simple_add(self):
        """Test simple_add convenience constructor."""
        v = torch.randn(128)
        spec = SteeringSpec.simple_add(layer=5, vector=v, scale=2.0)

        assert 5 in spec.layers
        assert not spec.is_empty()
        assert len(spec.layers[5].operations) == 1
        assert isinstance(spec.layers[5].operations[0], AddSpec)

    def test_simple_cap(self):
        """Test simple_cap convenience constructor."""
        v = torch.randn(128)
        spec = SteeringSpec.simple_cap(layer=10, vector=v, min=-0.5, max=0.5)

        assert 10 in spec.layers
        assert not spec.is_empty()
        assert len(spec.layers[10].operations) == 1
        assert isinstance(spec.layers[10].operations[0], ProjectionCapSpec)

    def test_simple_ablation(self):
        """Test simple_ablation convenience constructor."""
        v = torch.randn(128)
        spec = SteeringSpec.simple_ablation(layer=15, vector=v, scale=0.0)

        assert 15 in spec.layers
        assert not spec.is_empty()
        assert len(spec.layers[15].operations) == 1
        assert isinstance(spec.layers[15].operations[0], AblationSpec)

    def test_multi_layer(self):
        """Test multi-layer spec."""
        v1 = torch.randn(64)
        v2 = torch.randn(64)

        spec = SteeringSpec(layers={
            5: LayerSteeringSpec(operations=[AddSpec(vector=v1, scale=1.0)]),
            10: LayerSteeringSpec(operations=[AblationSpec(vector=v2, scale=0.5)]),
        })

        assert 5 in spec.layers
        assert 10 in spec.layers
        assert not spec.is_empty()

    def test_clone(self):
        """Test deep cloning."""
        v = torch.randn(64)
        spec = SteeringSpec.simple_add(layer=5, vector=v, scale=1.0)
        cloned = spec.clone()

        assert cloned.layers.keys() == spec.layers.keys()
        assert cloned.layers[5] is not spec.layers[5]
