"""Tests for steering operation parity between HF and vLLM backends.

These tests verify that the steering operations (add, cap, ablation) produce
identical results whether applied via the HuggingFace backend or the vLLM
runtime steering logic.
"""

import pytest
import torch

from steerllm import AddSpec, ProjectionCapSpec, AblationSpec, LayerSteeringSpec, SteeringSpec
from steerllm.backends.huggingface.hooks import apply_steering_ops
from steerllm.backends.vllm.runtime import (
    _apply_projection_cap,
    _apply_ablation,
    _apply_layer_steering_to_hidden,
    _ProjectionCapConfig,
    _AblationConfig,
    serialize_tensor,
    deserialize_tensor,
)


class TestTensorSerialization:
    """Tests for tensor serialization/deserialization via shared memory."""

    def test_float32_roundtrip(self):
        """Test float32 tensor roundtrip via shm."""
        original = torch.randn(64, dtype=torch.float32)
        serialized, shm = serialize_tensor(original)
        reconstructed = deserialize_tensor(serialized)
        shm.close()
        shm.unlink()
        assert torch.allclose(original, reconstructed)
        assert reconstructed.dtype == torch.float32

    def test_float16_roundtrip(self):
        """Test float16 tensor roundtrip via shm."""
        original = torch.randn(64, dtype=torch.float16)
        serialized, shm = serialize_tensor(original)
        reconstructed = deserialize_tensor(serialized)
        shm.close()
        shm.unlink()
        assert torch.allclose(original.float(), reconstructed.float(), rtol=1e-3)

    def test_bfloat16_roundtrip(self):
        """Test bfloat16 tensor roundtrip via shm preserves dtype."""
        original = torch.randn(64, dtype=torch.bfloat16)
        serialized, shm = serialize_tensor(original)
        # bfloat16 is preserved via uint8 view
        reconstructed = deserialize_tensor(serialized)
        shm.close()
        shm.unlink()
        assert reconstructed.dtype == torch.bfloat16
        assert torch.allclose(original.float(), reconstructed.float(), rtol=1e-2)

    def test_2d_tensor(self):
        """Test 2D tensor roundtrip via shm."""
        original = torch.randn(10, 64, dtype=torch.float32)
        serialized, shm = serialize_tensor(original)
        reconstructed = deserialize_tensor(serialized)
        shm.close()
        shm.unlink()
        assert torch.allclose(original, reconstructed)
        assert reconstructed.shape == (10, 64)

    def test_empty_tensor(self):
        """Test empty tensor roundtrip via shm."""
        original = torch.tensor([], dtype=torch.float32)
        serialized, shm = serialize_tensor(original)
        reconstructed = deserialize_tensor(serialized)
        shm.close()
        shm.unlink()
        assert reconstructed.numel() == 0


class TestProjectionCapParity:
    """Tests for projection cap parity between HF and vLLM."""

    def test_upper_bound_parity(self):
        """Test projection cap with upper bound produces same results."""
        direction = torch.zeros(64)
        direction[0] = 1.0  # Unit vector along first dimension

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 10.0  # Large positive projection

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            ProjectionCapSpec(vector=direction, max=1.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _ProjectionCapConfig(unit_vector=direction, min=None, max=1.0)
        vllm_output = _apply_projection_cap(hidden.clone(), vllm_config)

        assert torch.allclose(hf_output, vllm_output, atol=1e-6)
        assert torch.allclose(hf_output[:, 0], torch.ones(5), atol=1e-6)

    def test_lower_bound_parity(self):
        """Test projection cap with lower bound produces same results."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = -10.0  # Large negative projection

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            ProjectionCapSpec(vector=direction, min=-1.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _ProjectionCapConfig(unit_vector=direction, min=-1.0, max=None)
        vllm_output = _apply_projection_cap(hidden.clone(), vllm_config)

        assert torch.allclose(hf_output, vllm_output, atol=1e-6)
        assert torch.allclose(hf_output[:, 0], -torch.ones(5), atol=1e-6)

    def test_both_bounds_parity(self):
        """Test projection cap with both bounds produces same results."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(10, 64)
        hidden[:, 0] = torch.tensor([-5, -2, -0.5, 0, 0.5, 1, 2, 5, -1, 1])

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            ProjectionCapSpec(vector=direction, min=-1.0, max=1.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _ProjectionCapConfig(unit_vector=direction, min=-1.0, max=1.0)
        vllm_output = _apply_projection_cap(hidden.clone(), vllm_config)

        assert torch.allclose(hf_output, vllm_output, atol=1e-6)

        expected = torch.tensor([-1, -1, -0.5, 0, 0.5, 1, 1, 1, -1, 1])
        assert torch.allclose(hf_output[:, 0], expected, atol=1e-6)


class TestAblationParity:
    """Tests for ablation parity between HF and vLLM."""

    def test_full_ablation_parity(self):
        """Test full ablation (scale=0) produces same results."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 5.0

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=direction, scale=0.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _AblationConfig(unit_vector=direction, scale=0.0)
        vllm_output = _apply_ablation(hidden.clone(), vllm_config)

        assert torch.allclose(hf_output, vllm_output, atol=1e-6)
        assert torch.allclose(hf_output[:, 0], torch.zeros(5), atol=1e-6)

    def test_partial_ablation_parity(self):
        """Test partial ablation (scale=0.5) produces same results."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 10.0

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=direction, scale=0.5)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _AblationConfig(unit_vector=direction, scale=0.5)
        vllm_output = _apply_ablation(hidden.clone(), vllm_config)

        assert torch.allclose(hf_output, vllm_output, atol=1e-6)
        assert torch.allclose(hf_output[:, 0], torch.ones(5) * 5.0, atol=1e-6)

    def test_amplify_parity(self):
        """Test ablation with amplification (scale > 1) produces same results."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 2.0

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=direction, scale=2.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _AblationConfig(unit_vector=direction, scale=2.0)
        vllm_output = _apply_ablation(hidden.clone(), vllm_config)

        assert torch.allclose(hf_output, vllm_output, atol=1e-6)
        assert torch.allclose(hf_output[:, 0], torch.ones(5) * 4.0, atol=1e-6)


class TestAddSpecParity:
    """Tests for additive steering parity."""

    def test_basic_add_parity(self):
        """Test basic additive steering produces same results."""
        direction = torch.randn(64)
        direction = direction / direction.norm()

        hidden = torch.randn(5, 64)

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AddSpec(vector=direction, scale=2.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime uses pre-scaled vector
        scaled_vec = direction * 2.0

        # Create a mock layer spec for vLLM
        class MockLayerSpec:
            pass
        mock_spec = MockLayerSpec()
        mock_spec.operations = [("add", scaled_vec, None)]

        # Create mock state
        class MockState:
            pass
        mock_state = MockState()

        vllm_output = _apply_layer_steering_to_hidden(hidden.clone(), mock_spec, mock_state)

        assert torch.allclose(hf_output, vllm_output, atol=1e-5)


class TestMultiOperationParity:
    """Tests for multi-operation steering parity."""

    def test_add_then_cap_parity(self):
        """Test add followed by cap produces same results."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AddSpec(vector=direction, scale=10.0),
            ProjectionCapSpec(vector=direction, max=2.0),
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # Expected: add 10, cap to 2
        assert torch.allclose(hf_output[:, 0], torch.ones(5) * 2.0, atol=1e-5)

    def test_add_cap_ablate_sequence(self):
        """Test add -> cap -> ablate sequence produces expected results."""
        direction = torch.zeros(64)
        direction[0] = 1.0

        hidden = torch.zeros(5, 64)

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AddSpec(vector=direction, scale=10.0),
            ProjectionCapSpec(vector=direction, max=2.0),
            AblationSpec(vector=direction, scale=0.5),
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # Expected: add 10 -> 10, cap to 2 -> 2, scale by 0.5 -> 1
        assert torch.allclose(hf_output[:, 0], torch.ones(5), atol=1e-5)

    def test_orthogonal_operations_parity(self):
        """Test operations on orthogonal directions are independent."""
        dir1 = torch.zeros(64)
        dir1[0] = 1.0
        dir2 = torch.zeros(64)
        dir2[1] = 1.0

        hidden = torch.zeros(5, 64)
        hidden[:, 0] = 5.0
        hidden[:, 1] = 3.0

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=dir1, scale=0.0),
            ProjectionCapSpec(vector=dir2, max=1.0),
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # Dir1 component should be zeroed
        assert torch.allclose(hf_output[:, 0], torch.zeros(5), atol=1e-6)
        # Dir2 component should be capped to 1
        assert torch.allclose(hf_output[:, 1], torch.ones(5), atol=1e-6)


class TestRandomVectorParity:
    """Tests with random vectors to ensure general parity."""

    def test_random_projection_cap(self):
        """Test projection cap with random vectors."""
        torch.manual_seed(42)

        for _ in range(10):
            # Random unit vector
            direction = torch.randn(128)
            direction = direction / direction.norm()

            # Random hidden states
            hidden = torch.randn(20, 128)

            # Random bounds
            min_val = torch.randn(1).item() * 2
            max_val = min_val + abs(torch.randn(1).item()) + 0.1

            # HF backend
            hf_spec = LayerSteeringSpec(operations=[
                ProjectionCapSpec(vector=direction, min=min_val, max=max_val)
            ])
            hf_output = apply_steering_ops(hidden.clone(), hf_spec)

            # vLLM runtime
            vllm_config = _ProjectionCapConfig(unit_vector=direction, min=min_val, max=max_val)
            vllm_output = _apply_projection_cap(hidden.clone(), vllm_config)

            assert torch.allclose(hf_output, vllm_output, atol=1e-5), \
                f"Mismatch with min={min_val}, max={max_val}"

    def test_random_ablation(self):
        """Test ablation with random vectors and scales."""
        torch.manual_seed(42)

        for _ in range(10):
            # Random unit vector
            direction = torch.randn(128)
            direction = direction / direction.norm()

            # Random hidden states
            hidden = torch.randn(20, 128)

            # Random scale
            scale = abs(torch.randn(1).item()) * 2

            # HF backend
            hf_spec = LayerSteeringSpec(operations=[
                AblationSpec(vector=direction, scale=scale)
            ])
            hf_output = apply_steering_ops(hidden.clone(), hf_spec)

            # vLLM runtime
            vllm_config = _AblationConfig(unit_vector=direction, scale=scale)
            vllm_output = _apply_ablation(hidden.clone(), vllm_config)

            assert torch.allclose(hf_output, vllm_output, atol=1e-5), \
                f"Mismatch with scale={scale}"


class TestDeviceHandling:
    """Tests for device handling in parity."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_projection_cap_parity(self):
        """Test projection cap works on CUDA with same results."""
        direction = torch.zeros(64, device="cuda")
        direction[0] = 1.0

        hidden = torch.zeros(5, 64, device="cuda")
        hidden[:, 0] = 10.0

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            ProjectionCapSpec(vector=direction.cpu(), max=1.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _ProjectionCapConfig(unit_vector=direction, min=None, max=1.0)
        vllm_output = _apply_projection_cap(hidden.clone(), vllm_config)

        assert hf_output.device.type == "cuda"
        assert vllm_output.device.type == "cuda"
        assert torch.allclose(hf_output, vllm_output, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_ablation_parity(self):
        """Test ablation works on CUDA with same results."""
        direction = torch.zeros(64, device="cuda")
        direction[0] = 1.0

        hidden = torch.zeros(5, 64, device="cuda")
        hidden[:, 0] = 5.0

        # HF backend
        hf_spec = LayerSteeringSpec(operations=[
            AblationSpec(vector=direction.cpu(), scale=0.0)
        ])
        hf_output = apply_steering_ops(hidden.clone(), hf_spec)

        # vLLM runtime
        vllm_config = _AblationConfig(unit_vector=direction, scale=0.0)
        vllm_output = _apply_ablation(hidden.clone(), vllm_config)

        assert hf_output.device.type == "cuda"
        assert vllm_output.device.type == "cuda"
        assert torch.allclose(hf_output, vllm_output, atol=1e-6)
