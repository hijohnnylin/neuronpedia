"""Tests for input validation in steering specs.

These tests verify that invalid inputs are properly rejected with
informative error messages.
"""

from __future__ import annotations

import pytest
import torch

from steerllm import (
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
    LayerSteeringSpec,
    SteeringSpec,
)


class TestAddSpecValidation:
    """Validation tests for AddSpec."""

    def test_nan_vector_rejected(self):
        """Test that NaN vectors are rejected."""
        v = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            AddSpec(vector=v, scale=1.0)

    def test_inf_vector_rejected(self):
        """Test that Inf vectors are rejected."""
        v = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            AddSpec(vector=v, scale=1.0)

    def test_negative_inf_vector_rejected(self):
        """Test that -Inf vectors are rejected."""
        v = torch.tensor([1.0, float("-inf"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            AddSpec(vector=v, scale=1.0)

    def test_zero_vector_rejected(self):
        """Test that zero vectors are rejected."""
        v = torch.zeros(64)
        with pytest.raises(ValueError, match="zero norm"):
            AddSpec(vector=v, scale=1.0)

    def test_empty_vector_rejected(self):
        """Test that empty vectors are rejected."""
        v = torch.tensor([])
        with pytest.raises(ValueError, match="zero norm"):
            AddSpec(vector=v, scale=1.0)

    def test_very_small_vector_rejected(self):
        """Test that vectors with very small norms are rejected."""
        v = torch.tensor([1e-45, 1e-45])  # Subnormal, effectively zero
        with pytest.raises(ValueError, match="zero norm"):
            AddSpec(vector=v, scale=1.0)

    def test_valid_vector_accepted(self):
        """Test that valid vectors are accepted."""
        v = torch.randn(64)
        spec = AddSpec(vector=v, scale=1.0)
        assert spec.vector.shape == (64,)

    def test_nan_scale_produces_nan_materialize(self):
        """Test behavior with NaN scale."""
        v = torch.ones(64)
        spec = AddSpec(vector=v, scale=float("nan"))
        result = spec.materialize()
        assert torch.isnan(result).all()

    def test_inf_scale_produces_inf_materialize(self):
        """Test behavior with Inf scale."""
        v = torch.ones(64)
        spec = AddSpec(vector=v, scale=float("inf"))
        result = spec.materialize()
        assert torch.isinf(result).all()


class TestProjectionCapSpecValidation:
    """Validation tests for ProjectionCapSpec."""

    def test_no_bounds_rejected(self):
        """Test that having no bounds is rejected."""
        v = torch.randn(64)
        with pytest.raises(ValueError, match="at least one of min or max"):
            ProjectionCapSpec(vector=v)

    def test_none_bounds_rejected(self):
        """Test that explicit None bounds is rejected."""
        v = torch.randn(64)
        with pytest.raises(ValueError, match="at least one of min or max"):
            ProjectionCapSpec(vector=v, min=None, max=None)

    def test_min_only_accepted(self):
        """Test that min-only bounds are accepted."""
        v = torch.randn(64)
        spec = ProjectionCapSpec(vector=v, min=-1.0)
        assert spec.min == -1.0
        assert spec.max is None

    def test_max_only_accepted(self):
        """Test that max-only bounds are accepted."""
        v = torch.randn(64)
        spec = ProjectionCapSpec(vector=v, max=1.0)
        assert spec.min is None
        assert spec.max == 1.0

    def test_both_bounds_accepted(self):
        """Test that both bounds are accepted."""
        v = torch.randn(64)
        spec = ProjectionCapSpec(vector=v, min=-1.0, max=1.0)
        assert spec.min == -1.0
        assert spec.max == 1.0

    def test_inverted_bounds_allowed(self):
        """Test that inverted bounds (min > max) are allowed.

        This may produce unexpected results but should not crash.
        """
        v = torch.randn(64)
        spec = ProjectionCapSpec(vector=v, min=1.0, max=-1.0)
        assert spec.min == 1.0
        assert spec.max == -1.0

    def test_nan_vector_rejected(self):
        """Test that NaN vectors are rejected."""
        v = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            ProjectionCapSpec(vector=v, max=1.0)

    def test_zero_vector_rejected(self):
        """Test that zero vectors are rejected."""
        v = torch.zeros(64)
        with pytest.raises(ValueError, match="zero norm"):
            ProjectionCapSpec(vector=v, max=1.0)


class TestAblationSpecValidation:
    """Validation tests for AblationSpec."""

    def test_nan_vector_rejected(self):
        """Test that NaN vectors are rejected."""
        v = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            AblationSpec(vector=v, scale=0.5)

    def test_zero_vector_rejected(self):
        """Test that zero vectors are rejected."""
        v = torch.zeros(64)
        with pytest.raises(ValueError, match="zero norm"):
            AblationSpec(vector=v, scale=0.5)

    def test_negative_scale_allowed(self):
        """Test that negative scales are allowed (sign flip)."""
        v = torch.randn(64)
        spec = AblationSpec(vector=v, scale=-1.0)
        assert spec.scale == -1.0

    def test_large_scale_allowed(self):
        """Test that large scales are allowed."""
        v = torch.randn(64)
        spec = AblationSpec(vector=v, scale=1000.0)
        assert spec.scale == 1000.0


class TestLayerSteeringSpecValidation:
    """Validation tests for LayerSteeringSpec."""

    def test_empty_operations_allowed(self):
        """Test that empty operations list is allowed."""
        spec = LayerSteeringSpec(operations=[])
        assert spec.is_empty()

    def test_empty_spec_is_empty(self):
        """Test that default spec is empty."""
        spec = LayerSteeringSpec()
        assert spec.is_empty()

    def test_zero_scale_add_is_empty(self):
        """Test that zero-scale add is detected as empty."""
        v = torch.randn(64)
        spec = LayerSteeringSpec(operations=[AddSpec(vector=v, scale=0.0)])
        assert spec.is_empty()

    def test_nonzero_scale_add_not_empty(self):
        """Test that non-zero scale add is not empty."""
        v = torch.randn(64)
        spec = LayerSteeringSpec(operations=[AddSpec(vector=v, scale=0.001)])
        assert not spec.is_empty()


class TestSteeringSpecValidation:
    """Validation tests for SteeringSpec."""

    def test_empty_layers_is_empty(self):
        """Test that empty layers dict is detected as empty."""
        spec = SteeringSpec(layers={})
        assert spec.is_empty()

    def test_default_spec_is_empty(self):
        """Test that default spec is empty."""
        spec = SteeringSpec()
        assert spec.is_empty()

    def test_negative_layer_index_allowed(self):
        """Test that negative layer indices are allowed at spec level.

        Validation of layer indices happens at runtime against the model.
        """
        v = torch.randn(64)
        spec = SteeringSpec.simple_add(layer=-1, vector=v, scale=1.0)
        assert -1 in spec.layers

    def test_large_layer_index_allowed(self):
        """Test that large layer indices are allowed at spec level."""
        v = torch.randn(64)
        spec = SteeringSpec.simple_add(layer=9999, vector=v, scale=1.0)
        assert 9999 in spec.layers


class TestFromUnnormalizedValidation:
    """Tests for from_unnormalized factory method."""

    def test_normalizes_vector(self):
        """Test that from_unnormalized properly normalizes."""
        v = torch.ones(64) * 3.0
        spec = AddSpec.from_unnormalized(v, scale=1.0)

        # Vector should be unit
        assert torch.isclose(spec.vector.norm(), torch.tensor(1.0), atol=1e-5)

        # Scale should capture original norm
        expected_norm = v.norm().item()
        assert pytest.approx(spec.scale, rel=1e-5) == expected_norm

    def test_preserves_direction(self):
        """Test that from_unnormalized preserves direction."""
        v = torch.randn(64)
        spec = AddSpec.from_unnormalized(v, scale=1.0)

        # Materialized should be proportional to original
        materialized = spec.materialize()
        cosine = torch.nn.functional.cosine_similarity(
            v.unsqueeze(0), materialized.unsqueeze(0)
        )
        assert torch.isclose(cosine, torch.tensor(1.0), atol=1e-5)

    def test_zero_vector_rejected(self):
        """Test that zero vectors are rejected in from_unnormalized."""
        v = torch.zeros(64)
        with pytest.raises(ValueError, match="normalize zero"):
            AddSpec.from_unnormalized(v, scale=1.0)


class TestCloneValidation:
    """Tests for clone operations."""

    def test_add_spec_clone_independence(self):
        """Test that AddSpec clone is truly independent."""
        v = torch.randn(64)
        spec = AddSpec(vector=v, scale=1.0)
        cloned = spec.clone()

        # Modify original
        spec.vector.data.fill_(0.0)

        # Clone should be unaffected
        assert cloned.vector.norm() > 0

    def test_projection_cap_clone_independence(self):
        """Test that ProjectionCapSpec clone is independent."""
        v = torch.randn(64)
        spec = ProjectionCapSpec(vector=v, min=-1.0, max=1.0)
        cloned = spec.clone()

        # Modify original
        spec.vector.data.fill_(0.0)

        # Clone should be unaffected
        assert cloned.vector.norm() > 0

    def test_ablation_spec_clone_independence(self):
        """Test that AblationSpec clone is independent."""
        v = torch.randn(64)
        spec = AblationSpec(vector=v, scale=0.5)
        cloned = spec.clone()

        # Modify original
        spec.vector.data.fill_(0.0)

        # Clone should be unaffected
        assert cloned.vector.norm() > 0

    def test_steering_spec_clone_deep(self):
        """Test that SteeringSpec clone is deep."""
        v = torch.randn(64)
        spec = SteeringSpec.simple_add(layer=5, vector=v, scale=1.0)
        cloned = spec.clone()

        # Modify original
        spec.layers[5].operations[0].vector.data.fill_(0.0)

        # Clone should be unaffected
        assert cloned.layers[5].operations[0].vector.norm() > 0


class TestDtypePreservation:
    """Tests for dtype handling in specs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_add_spec_preserves_dtype(self, dtype):
        """Test that AddSpec preserves vector dtype."""
        v = torch.randn(64, dtype=dtype)
        spec = AddSpec(vector=v, scale=1.0)
        assert spec.vector.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_materialize_preserves_dtype(self, dtype):
        """Test that materialize preserves dtype."""
        v = torch.randn(64, dtype=dtype)
        spec = AddSpec(vector=v, scale=2.0)
        result = spec.materialize()
        assert result.dtype == dtype
