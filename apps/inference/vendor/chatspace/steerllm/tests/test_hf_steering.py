"""Tests for HuggingFace backend steering operations."""

import pytest
import torch

from steerllm import AddSpec, ProjectionCapSpec, AblationSpec, LayerSteeringSpec
from steerllm.backends.huggingface.hooks import (
    apply_steering_ops,
    ResidualHook,
)


class TestResidualHook:
    """Tests for ResidualHook module."""

    def test_zero_init(self):
        """Test zero initialization."""
        hook = ResidualHook(hidden_size=64, init_scale=0.0)
        assert torch.allclose(hook.vector, torch.zeros(64))

    def test_random_init(self):
        """Test random initialization."""
        torch.manual_seed(42)
        hook = ResidualHook(hidden_size=64, init_scale=0.1)
        assert hook.vector.norm() > 0
        assert hook.vector.shape == (64,)

    def test_forward_adds_vector(self):
        """Test forward pass adds vector."""
        hook = ResidualHook(hidden_size=64, init_scale=0.0)
        # Set a specific vector
        hook.vector.data = torch.ones(64) * 0.5

        hidden = torch.zeros(10, 64)  # [seq_len, hidden_size]
        output = hook(hidden)

        expected = torch.ones(10, 64) * 0.5
        assert torch.allclose(output, expected)


class TestSteeringOps:
    """Tests for steering operations."""

    def test_add_spec_steering(self):
        """Test additive steering."""
        hidden = torch.zeros(5, 64)  # [seq_len, hidden_size]
        vector = torch.randn(64)
        vector = vector / vector.norm()  # Normalize

        spec = LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=2.0)])
        output = apply_steering_ops(hidden, spec)

        expected = hidden + vector * 2.0
        assert torch.allclose(output, expected)

    def test_projection_cap_upper_bound(self):
        """Test projection capping with upper bound."""
        # Create hidden states with large positive projection
        direction = torch.zeros(64)
        direction[0] = 1.0  # Unit vector along first dimension

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 10.0  # Large positive projection

        spec = LayerSteeringSpec(operations=[
            ProjectionCapSpec(vector=direction, max=1.0)
        ])
        output = apply_steering_ops(hidden, spec)

        # Projection should be clamped to 1.0
        assert torch.allclose(output[:, 0], torch.ones(5))

    def test_projection_cap_lower_bound(self):
        """Test projection capping with lower bound."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = -10.0  # Large negative projection

        spec = LayerSteeringSpec(operations=[
            ProjectionCapSpec(vector=direction, min=-1.0)
        ])
        output = apply_steering_ops(hidden, spec)

        # Projection should be clamped to -1.0
        assert torch.allclose(output[:, 0], -torch.ones(5))

    def test_projection_cap_both_bounds(self):
        """Test projection capping with both bounds."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(10, 64)
        # Mix of values: some above, some below, some in range
        hidden[:, 0] = torch.tensor([-5, -2, -0.5, 0, 0.5, 1, 2, 5, -1, 1])

        spec = LayerSteeringSpec(operations=[
            ProjectionCapSpec(vector=direction, min=-1.0, max=1.0)
        ])
        output = apply_steering_ops(hidden, spec)

        # All projections should be in [-1, 1]
        expected = torch.tensor([-1, -1, -0.5, 0, 0.5, 1, 1, 1, -1, 1])
        assert torch.allclose(output[:, 0], expected)

    def test_ablation_full_removal(self):
        """Test full ablation (scale=0)."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 5.0  # Component to ablate

        spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=direction, scale=0.0)
        ])
        output = apply_steering_ops(hidden, spec)

        # Component along direction should be zero
        assert torch.allclose(output[:, 0], torch.zeros(5), atol=1e-6)

    def test_ablation_partial(self):
        """Test partial ablation (scale=0.5)."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 10.0

        spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=direction, scale=0.5)
        ])
        output = apply_steering_ops(hidden, spec)

        # Component should be scaled by 0.5
        assert torch.allclose(output[:, 0], torch.ones(5) * 5.0)

    def test_ablation_amplify(self):
        """Test ablation with amplification (scale > 1)."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 2.0

        spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=direction, scale=2.0)
        ])
        output = apply_steering_ops(hidden, spec)

        # Component should be doubled
        assert torch.allclose(output[:, 0], torch.ones(5) * 4.0)

    def test_multi_operation_sequence(self):
        """Test multiple operations applied in sequence."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 0.0

        spec = LayerSteeringSpec(operations=[
            # First, add a large value
            AddSpec(vector=direction, scale=10.0),
            # Then cap it
            ProjectionCapSpec(vector=direction, max=2.0),
            # Then scale it down
            AblationSpec(vector=direction, scale=0.5),
        ])
        output = apply_steering_ops(hidden, spec)

        # After add: 10.0
        # After cap: 2.0
        # After ablation (scale=0.5): 1.0
        assert torch.allclose(output[:, 0], torch.ones(5), atol=1e-5)

    def test_orthogonal_directions_independent(self):
        """Test that operations on orthogonal directions are independent."""
        dir1 = torch.zeros(64)
        dir1[0] = 1.0
        dir2 = torch.zeros(64)
        dir2[1] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 5.0
        hidden[:, 1] = 3.0

        spec = LayerSteeringSpec(operations=[
            # Ablate along dir1
            AblationSpec(vector=dir1, scale=0.0),
            # Cap along dir2
            ProjectionCapSpec(vector=dir2, max=1.0),
        ])
        output = apply_steering_ops(hidden, spec)

        # Dir1 component should be zeroed
        assert torch.allclose(output[:, 0], torch.zeros(5), atol=1e-6)
        # Dir2 component should be capped to 1
        assert torch.allclose(output[:, 1], torch.ones(5))


class TestDeviceDtypeHandling:
    """Tests for device and dtype handling."""

    def test_cpu_float32(self):
        """Test operations work with CPU float32."""
        hidden = torch.randn(5, 64, dtype=torch.float32)
        vector = torch.randn(64, dtype=torch.float32)
        vector = vector / vector.norm()

        spec = LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=1.0)])
        output = apply_steering_ops(hidden, spec)

        assert output.dtype == torch.float32
        assert output.device == hidden.device

    def test_cpu_float16(self):
        """Test operations work with CPU float16."""
        hidden = torch.randn(5, 64, dtype=torch.float16)
        vector = torch.randn(64, dtype=torch.float32)  # Vector may be different dtype
        vector = vector / vector.norm()

        spec = LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=1.0)])
        output = apply_steering_ops(hidden, spec)

        # Output should match hidden dtype
        assert output.dtype == torch.float16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_operations(self):
        """Test operations work on CUDA."""
        hidden = torch.randn(5, 64, device="cuda")
        vector = torch.randn(64)  # CPU vector
        vector = vector / vector.norm()

        spec = LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=1.0)])
        output = apply_steering_ops(hidden, spec)

        assert output.device.type == "cuda"
