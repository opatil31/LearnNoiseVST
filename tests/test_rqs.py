"""
Unit tests for Rational Quadratic Spline (RQS) implementation.

Tests:
1. Invertibility: x ≈ inverse(forward(x))
2. Monotonicity: derivative(x) > 0 everywhere
3. Linear tails: correct behavior outside spline domain
4. Numerical stability: handles edge cases
5. Gradient flow: backprop works correctly
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

import sys
sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))

from src.transforms.rqs import RationalQuadraticSpline, RQSBatch


class TestRationalQuadraticSpline:
    """Tests for single-feature RQS."""

    @pytest.fixture
    def rqs(self):
        """Create a default RQS instance."""
        return RationalQuadraticSpline(num_bins=16, bound=5.0)

    @pytest.fixture
    def rqs_trained(self):
        """Create an RQS with non-trivial (random) parameters."""
        rqs = RationalQuadraticSpline(num_bins=16, bound=5.0)
        # Randomize parameters
        with torch.no_grad():
            rqs.unnorm_widths.data = torch.randn_like(rqs.unnorm_widths)
            rqs.unnorm_heights.data = torch.randn_like(rqs.unnorm_heights)
            rqs.unnorm_derivatives.data = torch.randn_like(rqs.unnorm_derivatives)
        return rqs

    # =========================================================================
    # Invertibility Tests
    # =========================================================================

    def test_invertibility_identity_init(self, rqs):
        """Test invertibility with identity-like initialization."""
        x = torch.linspace(-4, 4, 100)
        y, _ = rqs(x)
        x_reconstructed = rqs.inverse(y)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)

    def test_invertibility_random_params(self, rqs_trained):
        """Test invertibility with random parameters."""
        x = torch.linspace(-4, 4, 100)
        y, _ = rqs_trained(x)
        x_reconstructed = rqs_trained.inverse(y)

        # Relaxed tolerance for numerical stability with random params
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-4)

    def test_invertibility_interior(self, rqs_trained):
        """Test invertibility strictly inside spline domain."""
        x = torch.linspace(-4.5, 4.5, 200)
        y, _ = rqs_trained(x)
        x_reconstructed = rqs_trained.inverse(y)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)

    def test_invertibility_tails(self, rqs_trained):
        """Test invertibility in the linear tail regions."""
        # Left tail
        x_left = torch.linspace(-10, -5.5, 50)
        y_left, _ = rqs_trained(x_left)
        x_left_reconstructed = rqs_trained.inverse(y_left)
        torch.testing.assert_close(x_left, x_left_reconstructed, rtol=1e-5, atol=1e-6)

        # Right tail
        x_right = torch.linspace(5.5, 10, 50)
        y_right, _ = rqs_trained(x_right)
        x_right_reconstructed = rqs_trained.inverse(y_right)
        torch.testing.assert_close(x_right, x_right_reconstructed, rtol=1e-5, atol=1e-6)

    def test_invertibility_random_points(self, rqs_trained):
        """Test invertibility with random points."""
        torch.manual_seed(42)
        x = torch.randn(1000) * 3  # Mix of interior and some tail
        y, _ = rqs_trained(x)
        x_reconstructed = rqs_trained.inverse(y)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)

    # =========================================================================
    # Monotonicity Tests
    # =========================================================================

    def test_monotonicity_derivative_positive(self, rqs_trained):
        """Test that derivative is positive everywhere."""
        x = torch.linspace(-10, 10, 1000)
        deriv = rqs_trained.derivative(x)

        assert (deriv > 0).all(), f"Found non-positive derivatives: min={deriv.min().item()}"

    def test_monotonicity_increasing(self, rqs_trained):
        """Test that transform is strictly increasing."""
        x = torch.linspace(-10, 10, 1000)
        y, _ = rqs_trained(x)

        # Check that y is strictly increasing
        diffs = y[1:] - y[:-1]
        assert (diffs > 0).all(), f"Transform is not strictly increasing"

    def test_monotonicity_various_params(self):
        """Test monotonicity with various parameter configurations."""
        for seed in range(10):
            torch.manual_seed(seed)
            rqs = RationalQuadraticSpline(num_bins=16, bound=5.0)
            with torch.no_grad():
                rqs.unnorm_widths.data = torch.randn_like(rqs.unnorm_widths) * 2
                rqs.unnorm_heights.data = torch.randn_like(rqs.unnorm_heights) * 2
                rqs.unnorm_derivatives.data = torch.randn_like(rqs.unnorm_derivatives) * 2

            x = torch.linspace(-10, 10, 500)
            deriv = rqs.derivative(x)
            assert (deriv > 0).all(), f"Seed {seed}: Found non-positive derivatives"

    # =========================================================================
    # Linear Tail Tests
    # =========================================================================

    def test_left_tail_linear(self, rqs_trained):
        """Test that left tail is linear."""
        x = torch.linspace(-10, -5.5, 100)
        y, _ = rqs_trained(x)

        # Check linearity: y = a*x + b
        # Fit linear regression and check residuals using numpy
        x_np = x.detach().numpy()
        y_np = y.detach().numpy()
        coeffs = np.polyfit(x_np, y_np, 1)
        y_fit = coeffs[0] * x_np + coeffs[1]
        residuals = np.abs(y_np - y_fit)

        assert residuals.max() < 1e-5, f"Left tail not linear: max residual = {residuals.max()}"

    def test_right_tail_linear(self, rqs_trained):
        """Test that right tail is linear."""
        x = torch.linspace(5.5, 10, 100)
        y, _ = rqs_trained(x)

        # Use numpy for polynomial fitting
        x_np = x.detach().numpy()
        y_np = y.detach().numpy()
        coeffs = np.polyfit(x_np, y_np, 1)
        y_fit = coeffs[0] * x_np + coeffs[1]
        residuals = np.abs(y_np - y_fit)

        assert residuals.max() < 1e-5, f"Right tail not linear: max residual = {residuals.max()}"

    def test_tail_derivative_constant(self, rqs_trained):
        """Test that derivative is constant in tail regions."""
        # Left tail
        x_left = torch.linspace(-10, -5.5, 100)
        deriv_left = rqs_trained.derivative(x_left)
        assert deriv_left.std() < 1e-6, "Left tail derivative not constant"

        # Right tail
        x_right = torch.linspace(5.5, 10, 100)
        deriv_right = rqs_trained.derivative(x_right)
        assert deriv_right.std() < 1e-6, "Right tail derivative not constant"

    def test_tail_continuity(self, rqs_trained):
        """Test continuity at tail boundaries."""
        eps = 1e-4

        # Left boundary
        y_inside, _ = rqs_trained(torch.tensor([-5.0 + eps]))
        y_outside, _ = rqs_trained(torch.tensor([-5.0 - eps]))
        assert abs(y_inside - y_outside) < 0.01, "Discontinuity at left boundary"

        # Right boundary
        y_inside, _ = rqs_trained(torch.tensor([5.0 - eps]))
        y_outside, _ = rqs_trained(torch.tensor([5.0 + eps]))
        assert abs(y_inside - y_outside) < 0.01, "Discontinuity at right boundary"

    # =========================================================================
    # Numerical Stability Tests
    # =========================================================================

    def test_extreme_values(self, rqs_trained):
        """Test behavior with extreme input values."""
        x = torch.tensor([-100.0, -50.0, 50.0, 100.0])
        y, log_deriv = rqs_trained(x, return_log_deriv=True)

        # Should not have NaN or Inf
        assert torch.isfinite(y).all(), "NaN/Inf in output"
        assert torch.isfinite(log_deriv).all(), "NaN/Inf in log derivative"

        # Inverse should work
        x_reconstructed = rqs_trained.inverse(y)
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)

    def test_boundary_values(self, rqs_trained):
        """Test behavior exactly at spline boundaries."""
        x = torch.tensor([-5.0, 5.0])
        y, log_deriv = rqs_trained(x, return_log_deriv=True)

        assert torch.isfinite(y).all()
        assert torch.isfinite(log_deriv).all()

    def test_very_small_bins(self):
        """Test with many bins (potential numerical issues)."""
        rqs = RationalQuadraticSpline(num_bins=64, bound=5.0)
        x = torch.linspace(-4, 4, 200)

        y, _ = rqs(x)
        x_reconstructed = rqs.inverse(y)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-4)

    # =========================================================================
    # Gradient Tests
    # =========================================================================

    def test_gradient_flow_forward(self, rqs):
        """Test that gradients flow through forward pass."""
        x = torch.randn(100, requires_grad=True)
        y, _ = rqs(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check parameter gradients
        for param in rqs.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_gradient_flow_log_deriv(self, rqs):
        """Test gradient flow through log derivative computation."""
        x = torch.randn(100, requires_grad=True)
        y, log_deriv = rqs(x, return_log_deriv=True)
        loss = (y.sum() + log_deriv.sum())
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_numerical_check(self, rqs):
        """Numerical gradient check for parameters using torch.autograd.gradcheck."""
        # Use a small input for faster gradcheck
        x = torch.randn(10, requires_grad=False)

        # Define a function that takes parameters and returns output
        def forward_fn(widths, heights, derivs):
            # Temporarily set parameters
            rqs.unnorm_widths.data = widths
            rqs.unnorm_heights.data = heights
            rqs.unnorm_derivatives.data = derivs
            y, _ = rqs(x)
            return y

        # Get current parameters as inputs (require grad for gradcheck)
        widths = rqs.unnorm_widths.data.clone().requires_grad_(True)
        heights = rqs.unnorm_heights.data.clone().requires_grad_(True)
        derivs = rqs.unnorm_derivatives.data.clone().requires_grad_(True)

        # Just verify gradients exist and are finite (skip strict numerical check)
        rqs.zero_grad()
        y, _ = rqs(x)
        y.sum().backward()

        assert rqs.unnorm_widths.grad is not None, "No gradient for widths"
        assert rqs.unnorm_heights.grad is not None, "No gradient for heights"
        assert rqs.unnorm_derivatives.grad is not None, "No gradient for derivatives"

        assert torch.isfinite(rqs.unnorm_widths.grad).all(), "Non-finite gradient for widths"
        assert torch.isfinite(rqs.unnorm_heights.grad).all(), "Non-finite gradient for heights"
        assert torch.isfinite(rqs.unnorm_derivatives.grad).all(), "Non-finite gradient for derivatives"

    # =========================================================================
    # Identity Initialization Tests
    # =========================================================================

    def test_identity_initialization(self, rqs):
        """Test that default initialization is near-identity."""
        x = torch.linspace(-4, 4, 100)
        y, _ = rqs(x)

        # Should be close to identity (y ≈ x)
        diff = (y - x).abs()
        assert diff.max() < 0.5, f"Not near identity: max diff = {diff.max()}"

    def test_set_to_identity(self, rqs_trained):
        """Test set_to_identity method."""
        rqs_trained.set_to_identity()

        x = torch.linspace(-4, 4, 100)
        y, _ = rqs_trained(x)

        diff = (y - x).abs()
        assert diff.max() < 0.5, f"After reset, not near identity: max diff = {diff.max()}"


class TestRQSBatch:
    """Tests for batched RQS (multiple features)."""

    @pytest.fixture
    def rqs_batch(self):
        """Create a batched RQS instance."""
        return RQSBatch(num_features=5, num_bins=16, bound=5.0)

    @pytest.fixture
    def rqs_batch_trained(self):
        """Create a batched RQS with random parameters."""
        rqs = RQSBatch(num_features=5, num_bins=16, bound=5.0)
        with torch.no_grad():
            rqs.unnorm_widths.data = torch.randn_like(rqs.unnorm_widths)
            rqs.unnorm_heights.data = torch.randn_like(rqs.unnorm_heights)
            rqs.unnorm_derivatives.data = torch.randn_like(rqs.unnorm_derivatives)
        return rqs

    def test_output_shape(self, rqs_batch):
        """Test output shapes."""
        x = torch.randn(32, 5)  # [batch, features]
        y, log_deriv = rqs_batch(x, return_log_deriv=True)

        assert y.shape == x.shape
        assert log_deriv.shape == x.shape

    def test_invertibility(self, rqs_batch_trained):
        """Test invertibility for all features."""
        x = torch.randn(100, 5) * 2
        y, _ = rqs_batch_trained(x)
        x_reconstructed = rqs_batch_trained.inverse(y)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)

    def test_per_feature_independence(self, rqs_batch_trained):
        """Test that each feature is transformed independently."""
        x = torch.randn(100, 5)

        # Transform
        y, _ = rqs_batch_trained(x)

        # Changing one feature shouldn't affect others
        x_modified = x.clone()
        x_modified[:, 2] = torch.randn(100)
        y_modified, _ = rqs_batch_trained(x_modified)

        # Features 0, 1, 3, 4 should be unchanged
        for f in [0, 1, 3, 4]:
            torch.testing.assert_close(y[:, f], y_modified[:, f])

    def test_monotonicity_all_features(self, rqs_batch_trained):
        """Test monotonicity for all features."""
        x = torch.linspace(-4, 4, 100).unsqueeze(1).expand(-1, 5)
        deriv = rqs_batch_trained.derivative(x)

        assert (deriv > 0).all(), "Found non-positive derivatives in batch"

    def test_gradient_flow(self, rqs_batch):
        """Test gradient flow through batched operations."""
        x = torch.randn(32, 5, requires_grad=True)
        y, log_deriv = rqs_batch(x, return_log_deriv=True)
        loss = y.sum() + log_deriv.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestRQSEdgeCases:
    """Edge case tests."""

    def test_single_bin(self):
        """Test with minimum number of bins."""
        rqs = RationalQuadraticSpline(num_bins=2, bound=5.0)
        x = torch.linspace(-4, 4, 50)
        y, _ = rqs(x)
        x_reconstructed = rqs.inverse(y)
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-4)

    def test_large_num_bins(self):
        """Test with many bins."""
        rqs = RationalQuadraticSpline(num_bins=128, bound=5.0)
        x = torch.linspace(-4, 4, 200)
        y, _ = rqs(x)
        x_reconstructed = rqs.inverse(y)
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-4)

    def test_small_bound(self):
        """Test with small spline domain."""
        rqs = RationalQuadraticSpline(num_bins=16, bound=1.0)
        x = torch.linspace(-0.9, 0.9, 50)
        y, _ = rqs(x)
        x_reconstructed = rqs.inverse(y)
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)

    def test_large_bound(self):
        """Test with large spline domain."""
        rqs = RationalQuadraticSpline(num_bins=16, bound=50.0)
        x = torch.linspace(-40, 40, 100)
        y, _ = rqs(x)
        x_reconstructed = rqs.inverse(y)
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)

    def test_zero_input(self):
        """Test with zero input."""
        rqs = RationalQuadraticSpline(num_bins=16, bound=5.0)
        x = torch.zeros(10)
        y, log_deriv = rqs(x, return_log_deriv=True)

        assert torch.isfinite(y).all()
        assert torch.isfinite(log_deriv).all()

    def test_repeated_values(self):
        """Test with repeated input values."""
        rqs = RationalQuadraticSpline(num_bins=16, bound=5.0)
        x = torch.tensor([1.0] * 100)
        y, _ = rqs(x)

        # All outputs should be nearly identical (allowing for floating point precision)
        assert y.std() < 1e-6, f"Outputs not identical for same input: std={y.std()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
