"""Unit tests for the OrthogonalProjector class.

This test file ensures the OrthogonalProjector class works correctly and prevents
regression of the critical bugs that were previously fixed:
1. Steering vector corruption (overwriting with projection matrix)
2. Missing normalization in projection matrix formula
3. Mathematical incorrectness leading to non-idempotent matrices
"""

import pytest
import torch
from neuronpedia_inference.inference_utils.steering import OrthogonalProjector


class TestOrthogonalProjectorCorrectness:
    """Test cases that verify correct mathematical behavior."""
    
    def test_steering_vector_preservation(self):
        """Test that the steering vector is never modified during operations."""
        original_vector = torch.tensor([1.0, 2.0, 3.0])
        projector = OrthogonalProjector(original_vector.clone())
        
        # Store the original steering vector for comparison
        original_steering_vector = projector.steering_vector.clone()
        original_shape = projector.steering_vector.shape
        
        # Multiple operations that should NOT modify the steering vector
        _ = projector.get_P()
        _ = projector.get_orthogonal_complement()
        _ = projector.project(torch.tensor([1.0, 1.0, 1.0]), strength_multiplier=0.5)
        
        # Verify steering vector remains unchanged
        assert projector.steering_vector.shape == original_shape, "Steering vector shape changed"
        torch.testing.assert_close(projector.steering_vector, original_steering_vector)
        
        # Verify the flattened version matches original input
        torch.testing.assert_close(projector.steering_vector.flatten(), original_vector)
    
    def test_projection_matrix_correct_formula(self):
        """Test that projection matrix uses P = vv^T / ||v||^2."""
        # Test with unit vector
        unit_vector = torch.tensor([1.0, 0.0, 0.0])
        projector_unit = OrthogonalProjector(unit_vector)
        P_unit = projector_unit.get_P()
        
        expected_unit = torch.tensor([[1.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
        torch.testing.assert_close(P_unit, expected_unit)
        
        # Test with non-unit vector - this is where the bug was most apparent
        non_unit_vector = torch.tensor([2.0, 0.0, 0.0])
        projector_non_unit = OrthogonalProjector(non_unit_vector)
        P_non_unit = projector_non_unit.get_P()
        
        # Should be normalized: vv^T/||v||^2 = [[4,0,0],[0,0,0],[0,0,0]] / 4
        expected_non_unit = torch.tensor([[1.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]])
        torch.testing.assert_close(P_non_unit, expected_non_unit)
        
        # Test with general vector
        general_vector = torch.tensor([1.0, 1.0, 0.0])
        projector_general = OrthogonalProjector(general_vector)
        P_general = projector_general.get_P()
        
        # vv^T/||v||^2 = [[1,1,0],[1,1,0],[0,0,0]] / 2
        expected_general = torch.tensor([[0.5, 0.5, 0.0],
                                        [0.5, 0.5, 0.0],
                                        [0.0, 0.0, 0.0]])
        torch.testing.assert_close(P_general, expected_general)
    
    def test_projection_matrix_idempotent_property(self):
        """Test that projection matrix satisfies P^2 = P (idempotent property)."""
        test_vectors = [
            torch.tensor([1.0, 0.0, 0.0]),  # unit vector
            torch.tensor([2.0, 0.0, 0.0]),  # scaled unit vector
            torch.tensor([1.0, 1.0, 0.0]),  # diagonal vector
            torch.tensor([1.0, 2.0, 3.0]),  # general vector
            torch.tensor([0.5, -1.5, 2.0]) # vector with negative components
        ]
        
        for vector in test_vectors:
            projector = OrthogonalProjector(vector)
            P = projector.get_P()
            P_squared = torch.matmul(P, P)
            
            # P^2 should equal P for any valid projection matrix
            torch.testing.assert_close(P_squared, P, rtol=1e-5, atol=1e-6,
                                     msg=f"Idempotent property failed for vector {vector}")
    
    def test_orthogonal_complement_correctness(self):
        """Test that orthogonal complement I-P is computed correctly."""
        # Test with axis-aligned vector
        vector = torch.tensor([1.0, 0.0, 0.0])
        projector = OrthogonalProjector(vector)
        complement = projector.get_orthogonal_complement()
        
        expected_complement = torch.tensor([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0]])
        torch.testing.assert_close(complement, expected_complement)
        
        # Verify I-P is also idempotent
        complement_squared = torch.matmul(complement, complement)
        torch.testing.assert_close(complement_squared, complement, rtol=1e-5, atol=1e-6)
    
    def test_projection_and_complement_orthogonality(self):
        """Test that P and (I-P) are orthogonal: P(I-P) = 0."""
        vectors = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([2.0, -1.0, 0.5])
        ]
        
        for vector in vectors:
            projector = OrthogonalProjector(vector)
            P = projector.get_P()
            complement = projector.get_orthogonal_complement()
            
            # P * (I-P) should be zero matrix
            product = torch.matmul(P, complement)
            zero_matrix = torch.zeros_like(product)
            torch.testing.assert_close(product, zero_matrix, rtol=1e-5, atol=1e-6,
                                     msg=f"Orthogonality failed for vector {vector}")
    
    def test_multiple_calls_consistency(self):
        """Test that multiple calls return consistent results."""
        vector = torch.tensor([1.0, 2.0, 3.0])
        projector = OrthogonalProjector(vector)
        
        # Multiple calls should return identical results due to caching
        P1 = projector.get_P()
        P2 = projector.get_P()
        P3 = projector.get_P()
        
        torch.testing.assert_close(P1, P2)
        torch.testing.assert_close(P2, P3)
        
        complement1 = projector.get_orthogonal_complement()
        complement2 = projector.get_orthogonal_complement()
        
        torch.testing.assert_close(complement1, complement2)
        
        # Steering vector should remain unchanged
        assert projector.steering_vector.shape == torch.Size([3, 1])


class TestOrthogonalProjectorProjectionOperations:
    """Test cases for the projection operation itself."""
    
    def test_projection_with_axis_aligned_vector(self):
        """Test projection with simple axis-aligned steering vector."""
        # Project onto x-axis
        steering_vector = torch.tensor([1.0, 0.0, 0.0])
        projector = OrthogonalProjector(steering_vector)
        activations = torch.tensor([2.0, 3.0, 4.0])
        
        # strength=1.0: should preserve all components
        projected_full = projector.project(activations, strength_multiplier=1.0)
        torch.testing.assert_close(projected_full, activations)
        
        # strength=0.0: should remove x-component, keep y,z
        projected_zero = projector.project(activations, strength_multiplier=0.0)
        expected_zero = torch.tensor([0.0, 3.0, 4.0])
        torch.testing.assert_close(projected_zero, expected_zero)
        
        # strength=2.0: should amplify x-component
        projected_amplified = projector.project(activations, strength_multiplier=2.0)
        expected_amplified = torch.tensor([4.0, 3.0, 4.0])
        torch.testing.assert_close(projected_amplified, expected_amplified)
    
    def test_projection_preserves_parallel_component(self):
        """Test that projection correctly handles parallel components."""
        steering_vector = torch.tensor([1.0, 1.0, 0.0])
        projector = OrthogonalProjector(steering_vector)
        
        # Activations parallel to steering vector
        parallel_activations = torch.tensor([2.0, 2.0, 0.0])  # 2 * [1,1,0]
        
        # strength=1.0: parallel component preserved
        projected = projector.project(parallel_activations, strength_multiplier=1.0)
        torch.testing.assert_close(projected, parallel_activations, rtol=1e-5, atol=1e-6)
        
        # strength=0.0: parallel component removed
        projected_zero = projector.project(parallel_activations, strength_multiplier=0.0)
        expected_zero = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(projected_zero, expected_zero, rtol=1e-5, atol=1e-6)
    
    def test_projection_with_orthogonal_activations(self):
        """Test projection with activations orthogonal to steering vector."""
        steering_vector = torch.tensor([1.0, 0.0, 0.0])
        projector = OrthogonalProjector(steering_vector)
        
        # Activations orthogonal to steering vector (no x-component)
        orthogonal_activations = torch.tensor([0.0, 3.0, 4.0])
        
        # Any strength should preserve orthogonal components
        for strength in [0.0, 0.5, 1.0, 2.0]:
            projected = projector.project(orthogonal_activations, strength_multiplier=strength)
            torch.testing.assert_close(projected, orthogonal_activations, rtol=1e-5, atol=1e-6)


class TestOrthogonalProjectorEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_zero_vector_error(self):
        """Test that zero vector raises appropriate error."""
        zero_vector = torch.tensor([0.0, 0.0, 0.0])
        projector = OrthogonalProjector(zero_vector)
        
        with pytest.raises(ValueError, match="Cannot create projection matrix from zero vector"):
            projector.get_P()
    
    def test_very_small_vector_handling(self):
        """Test handling of very small but non-zero vectors."""
        small_vector = torch.tensor([1e-8, 0.0, 0.0])
        projector = OrthogonalProjector(small_vector)
        
        # Should not raise error and should produce valid projection matrix
        P = projector.get_P()
        assert torch.isfinite(P).all(), "Projection matrix contains inf/nan"
        
        # Should still be idempotent
        P_squared = torch.matmul(P, P)
        torch.testing.assert_close(P_squared, P, rtol=1e-4, atol=1e-8)
    
    def test_large_vector_handling(self):
        """Test handling of large vectors."""
        large_vector = torch.tensor([1e6, 2e6, 3e6])
        projector = OrthogonalProjector(large_vector)
        
        P = projector.get_P()
        assert torch.isfinite(P).all(), "Projection matrix contains inf/nan"
        
        # Should still be idempotent
        P_squared = torch.matmul(P, P)
        torch.testing.assert_close(P_squared, P, rtol=1e-4, atol=1e-4)


class TestOrthogonalProjectorRegressionTests:
    """Specific tests to prevent regression of the bugs that were fixed."""
    
    def test_steering_vector_not_overwritten_regression(self):
        """Regression test: ensure steering vector is never overwritten."""
        original_vector = torch.tensor([1.0, 2.0, 3.0])
        projector = OrthogonalProjector(original_vector.clone())
        
        # Store original for comparison
        original_steering_vector = projector.steering_vector.clone()
        
        # This operation previously overwrote steering_vector with the projection matrix
        P = projector.get_P()
        
        # REGRESSION CHECK: steering vector should be unchanged
        assert projector.steering_vector.shape == torch.Size([3, 1]), \
            "REGRESSION: steering vector shape changed from [3,1]"
        assert not projector.steering_vector.shape == torch.Size([3, 3]), \
            "REGRESSION: steering vector became a 3x3 matrix"
        torch.testing.assert_close(projector.steering_vector, original_steering_vector,
                                 msg="REGRESSION: steering vector was modified")
    
    def test_projection_matrix_normalization_regression(self):
        """Regression test: ensure projection matrix is properly normalized."""
        # Use non-unit vector where the bug was most apparent
        vector = torch.tensor([2.0, 0.0, 0.0])
        projector = OrthogonalProjector(vector)
        P = projector.get_P()
        
        # REGRESSION CHECK: should be vv^T/||v||^2, not just vv^T
        # For [2,0,0], correct result is [[1,0,0],[0,0,0],[0,0,0]], not [[4,0,0],[0,0,0],[0,0,0]]
        expected_normalized = torch.tensor([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0]])
        buggy_unnormalized = torch.tensor([[4.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]])
        
        torch.testing.assert_close(P, expected_normalized,
                                 msg="REGRESSION: projection matrix not normalized")
        assert not torch.allclose(P, buggy_unnormalized), \
            "REGRESSION: projection matrix missing normalization"
    
    def test_idempotent_property_regression(self):
        """Regression test: ensure P^2 = P (was broken due to missing normalization)."""
        # Test with vector that would expose the normalization bug
        vector = torch.tensor([1.0, 1.0, 0.0])
        projector = OrthogonalProjector(vector)
        P = projector.get_P()
        
        P_squared = torch.matmul(P, P)
        
        # REGRESSION CHECK: P^2 should equal P
        torch.testing.assert_close(P_squared, P, rtol=1e-5, atol=1e-6,
                                 msg="REGRESSION: projection matrix not idempotent")
    
    def test_multiple_get_p_calls_regression(self):
        """Regression test: multiple calls to get_P() should work without errors."""
        vector = torch.tensor([1.0, 2.0, 3.0])
        projector = OrthogonalProjector(vector)
        
        # Previously, second call would fail due to steering vector corruption
        try:
            P1 = projector.get_P()
            P2 = projector.get_P()  # This call previously failed
            P3 = projector.get_P()
        except Exception as e:
            pytest.fail(f"REGRESSION: Multiple get_P() calls failed: {e}")
        
        # All calls should return the same result
        torch.testing.assert_close(P1, P2, msg="REGRESSION: Inconsistent get_P() results")
        torch.testing.assert_close(P2, P3, msg="REGRESSION: Inconsistent get_P() results")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
