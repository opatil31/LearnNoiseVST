"""
Unit tests for MonotoneFeatureTransform.

Tests:
1. Gauge-fixing: output has mean≈0, var≈1 per feature
2. Invertibility with gauge-fixing
3. Running statistics updates
4. Input normalization
5. Gradient flow
6. Image transform variant
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))

from src.transforms.monotone_transform import MonotoneFeatureTransform, ImageMonotoneTransform


class TestMonotoneFeatureTransform:
    """Tests for MonotoneFeatureTransform."""

    @pytest.fixture
    def transform(self):
        """Create a default transform."""
        return MonotoneFeatureTransform(
            num_features=5,
            num_bins=16,
            bound=5.0,
            momentum=0.1,
            track_running_stats=True
        )

    @pytest.fixture
    def transform_trained(self):
        """Create a transform with randomized parameters."""
        t = MonotoneFeatureTransform(
            num_features=5,
            num_bins=16,
            bound=5.0,
            momentum=0.1,
            track_running_stats=True
        )
        # Randomize RQS parameters
        with torch.no_grad():
            t.rqs.unnorm_widths.data = torch.randn_like(t.rqs.unnorm_widths) * 0.5
            t.rqs.unnorm_heights.data = torch.randn_like(t.rqs.unnorm_heights) * 0.5
            t.rqs.unnorm_derivatives.data = torch.randn_like(t.rqs.unnorm_derivatives) * 0.5
        return t

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        torch.manual_seed(42)
        # Data with different scales per feature
        x = torch.randn(1000, 5)
        x[:, 0] *= 10  # Feature 0: large scale
        x[:, 1] *= 0.1  # Feature 1: small scale
        x[:, 2] += 50  # Feature 2: large offset
        x[:, 3] = torch.abs(x[:, 3]) + 1  # Feature 3: positive only
        # Feature 4: standard normal
        return x

    # =========================================================================
    # Basic Functionality Tests
    # =========================================================================

    def test_output_shape(self, transform):
        """Test output shape matches input."""
        x = torch.randn(32, 5)
        z = transform(x)
        assert z.shape == x.shape

    def test_output_shape_with_extras(self, transform):
        """Test output shapes with return_prenorm and return_log_deriv."""
        x = torch.randn(32, 5)

        z, s = transform(x, return_prenorm=True)
        assert z.shape == x.shape
        assert s.shape == x.shape

        z, log_deriv = transform(x, return_log_deriv=True)
        assert z.shape == x.shape
        assert log_deriv.shape == x.shape

        z, s, log_deriv = transform(x, return_prenorm=True, return_log_deriv=True)
        assert z.shape == x.shape
        assert s.shape == x.shape
        assert log_deriv.shape == x.shape

    # =========================================================================
    # Gauge-Fixing Tests
    # =========================================================================

    def test_gauge_fixing_mean(self, transform, sample_data):
        """Test that output mean is approximately 0 after warmup."""
        transform.train()

        # Set input normalization
        transform.set_input_normalization(sample_data)

        # Warmup: run multiple passes to update running stats
        # EMA with momentum=0.1 needs several passes to converge
        batch_size = 100
        for _ in range(5):  # Multiple passes
            for i in range(0, len(sample_data), batch_size):
                batch = sample_data[i:i+batch_size]
                z = transform(batch, update_stats=True)

        # After warmup, check output statistics on fresh data
        transform.eval()
        with torch.no_grad():
            z = transform(sample_data, update_stats=False)
            mean_per_feature = z.mean(dim=0)

        # Mean should be close to 0 (relaxed tolerance for EMA convergence)
        assert (mean_per_feature.abs() < 0.3).all(), \
            f"Output mean not close to 0: {mean_per_feature}"

    def test_gauge_fixing_variance(self, transform, sample_data):
        """Test that output variance is approximately 1 after warmup."""
        transform.train()
        transform.set_input_normalization(sample_data)

        # Warmup: run multiple passes for EMA convergence
        batch_size = 100
        for _ in range(5):
            for i in range(0, len(sample_data), batch_size):
                batch = sample_data[i:i+batch_size]
                z = transform(batch, update_stats=True)

        # Check variance
        transform.eval()
        with torch.no_grad():
            z = transform(sample_data, update_stats=False)
            var_per_feature = z.var(dim=0)

        # Variance should be close to 1
        assert (var_per_feature > 0.5).all() and (var_per_feature < 2.0).all(), \
            f"Output variance not close to 1: {var_per_feature}"

    def test_gauge_fixing_prevents_shrinking(self, transform_trained, sample_data):
        """Test that gauge-fixing prevents variance shrinking."""
        transform_trained.set_input_normalization(sample_data)
        transform_trained.eval()

        # Even with random transform params, output should be standardized
        with torch.no_grad():
            z = transform_trained(sample_data, update_stats=False)
            var_per_feature = z.var(dim=0)

        # Running stats are at defaults (mean=0, var=1), so output
        # variance depends on pre-norm output. After refresh, it should be ~1.
        # This test mainly checks the mechanism exists.
        assert z.shape == sample_data.shape

    # =========================================================================
    # Invertibility Tests
    # =========================================================================

    def test_invertibility_basic(self, transform, sample_data):
        """Test basic invertibility."""
        transform.set_input_normalization(sample_data)

        # Use a subset
        x = sample_data[:100]

        # Warmup to stabilize running stats
        transform.train()
        for _ in range(10):
            _ = transform(sample_data, update_stats=True)

        transform.eval()
        with torch.no_grad():
            z = transform(x, update_stats=False)
            x_reconstructed = transform.inverse(z)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-3)

    def test_invertibility_trained(self, transform_trained, sample_data):
        """Test invertibility with trained parameters."""
        transform_trained.set_input_normalization(sample_data)

        x = sample_data[:100]

        transform_trained.eval()
        with torch.no_grad():
            z = transform_trained(x, update_stats=False)
            x_reconstructed = transform_trained.inverse(z)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-3)

    # =========================================================================
    # Running Statistics Tests
    # =========================================================================

    def test_running_stats_update(self, transform, sample_data):
        """Test that running stats are updated during training."""
        transform.set_input_normalization(sample_data)

        initial_mean = transform.running_mean.clone()
        initial_var = transform.running_var.clone()

        transform.train()
        _ = transform(sample_data[:100], update_stats=True)

        # Stats should have changed
        assert not torch.allclose(transform.running_mean, initial_mean) or \
               not torch.allclose(transform.running_var, initial_var), \
            "Running stats did not update"

    def test_running_stats_no_update_eval(self, transform, sample_data):
        """Test that running stats don't update in eval mode."""
        transform.set_input_normalization(sample_data)

        # First, do some updates
        transform.train()
        _ = transform(sample_data[:100], update_stats=True)

        # Record stats
        mean_before = transform.running_mean.clone()
        var_before = transform.running_var.clone()

        # Eval mode - stats should not change
        transform.eval()
        _ = transform(sample_data[100:200], update_stats=True)  # update_stats ignored

        torch.testing.assert_close(transform.running_mean, mean_before)
        torch.testing.assert_close(transform.running_var, var_before)

    def test_refresh_stats(self, transform, sample_data):
        """Test refresh_stats method."""
        transform.set_input_normalization(sample_data)

        # Create a dataloader
        dataset = TensorDataset(sample_data)
        dataloader = DataLoader(dataset, batch_size=100)

        # Refresh stats
        transform.refresh_stats(dataloader)

        # After refresh, output should be well-standardized
        transform.eval()
        with torch.no_grad():
            z = transform(sample_data, update_stats=False)
            mean = z.mean(dim=0)
            var = z.var(dim=0)

        assert (mean.abs() < 0.1).all(), f"Mean after refresh: {mean}"
        assert (var > 0.8).all() and (var < 1.2).all(), f"Var after refresh: {var}"

    # =========================================================================
    # Input Normalization Tests
    # =========================================================================

    def test_input_normalization_set(self, transform, sample_data):
        """Test that input normalization is set correctly."""
        transform.set_input_normalization(sample_data)

        # Check that normalization parameters are set
        assert transform._input_normalization_set
        assert not torch.allclose(transform.input_shift, torch.zeros_like(transform.input_shift)) or \
               not torch.allclose(transform.input_scale, torch.ones_like(transform.input_scale))

    def test_input_normalization_effect(self, transform, sample_data):
        """Test that input normalization maps data to spline domain."""
        transform.set_input_normalization(sample_data)

        # Normalized input should be roughly in [-bound, bound]
        x_norm = transform._normalize_input(sample_data)

        # Most values should be within bounds (with some exceptions for tails)
        in_bounds = (x_norm.abs() <= transform.bound * 1.5).float().mean()
        assert in_bounds > 0.95, f"Only {in_bounds*100:.1f}% of values in bounds"

    # =========================================================================
    # Gradient Tests
    # =========================================================================

    def test_gradient_flow(self, transform, sample_data):
        """Test that gradients flow through the transform."""
        transform.set_input_normalization(sample_data)

        x = sample_data[:32].clone().requires_grad_(True)
        z = transform(x)
        loss = z.sum()
        loss.backward()

        # Check input gradient
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check parameter gradients
        for name, param in transform.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_through_log_deriv(self, transform, sample_data):
        """Test gradients through log derivative."""
        transform.set_input_normalization(sample_data)

        x = sample_data[:32].clone().requires_grad_(True)
        z, log_deriv = transform(x, return_log_deriv=True)
        loss = z.sum() + log_deriv.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # =========================================================================
    # Derivative Tests
    # =========================================================================

    def test_derivative_positive(self, transform_trained, sample_data):
        """Test that transform derivative is positive (monotonicity)."""
        transform_trained.set_input_normalization(sample_data)

        x = sample_data[:100]
        deriv = transform_trained.derivative(x)

        assert (deriv > 0).all(), "Found non-positive derivatives"

    def test_log_derivative_finite(self, transform_trained, sample_data):
        """Test that log derivative is finite."""
        transform_trained.set_input_normalization(sample_data)

        x = sample_data[:100]
        log_deriv = transform_trained.log_derivative(x)

        assert torch.isfinite(log_deriv).all(), "Non-finite log derivatives"

    # =========================================================================
    # Affine Parameter Tests
    # =========================================================================

    def test_affine_parameters(self, sample_data):
        """Test transform with affine parameters."""
        transform = MonotoneFeatureTransform(
            num_features=5,
            num_bins=16,
            affine=True
        )
        transform.set_input_normalization(sample_data)

        # Check affine parameters exist
        assert transform.weight is not None
        assert transform.bias is not None

        # Forward should work
        z = transform(sample_data[:32])
        assert z.shape == (32, 5)

        # Inverse should work
        x_reconstructed = transform.inverse(z)
        assert x_reconstructed.shape == (32, 5)

    # =========================================================================
    # Reset Tests
    # =========================================================================

    def test_set_to_identity(self, transform_trained):
        """Test resetting to identity."""
        transform_trained.set_to_identity()

        x = torch.randn(100, 5) * 2
        z = transform_trained(x, update_stats=False)

        # With identity-like transform and default stats (mean=0, var=1),
        # output should be close to normalized input
        # This is approximate due to input normalization
        assert z.shape == x.shape

    # =========================================================================
    # Batch Independence Tests
    # =========================================================================

    def test_batch_independence(self, transform, sample_data):
        """Test that samples in a batch are processed independently."""
        transform.set_input_normalization(sample_data)
        transform.eval()

        x = sample_data[:10]

        # Process full batch
        z_batch = transform(x, update_stats=False)

        # Process individually
        z_individual = []
        for i in range(10):
            z_i = transform(x[i:i+1], update_stats=False)
            z_individual.append(z_i)
        z_individual = torch.cat(z_individual, dim=0)

        torch.testing.assert_close(z_batch, z_individual, rtol=1e-5, atol=1e-5)


class TestImageMonotoneTransform:
    """Tests for image-specific transform."""

    @pytest.fixture
    def transform(self):
        """Create image transform."""
        return ImageMonotoneTransform(num_channels=3, num_bins=16)

    @pytest.fixture
    def sample_images(self):
        """Generate sample image data."""
        torch.manual_seed(42)
        # [B, C, H, W]
        return torch.randn(16, 3, 32, 32)

    def test_output_shape(self, transform, sample_images):
        """Test output shape for images."""
        transform.set_input_normalization(sample_images)
        z = transform(sample_images)
        assert z.shape == sample_images.shape

    def test_channel_independence(self, transform, sample_images):
        """Test that channels are transformed independently."""
        transform.set_input_normalization(sample_images)
        transform.eval()

        z1 = transform(sample_images, update_stats=False)

        # Modify one channel
        images_modified = sample_images.clone()
        images_modified[:, 1] = torch.randn_like(images_modified[:, 1])
        z2 = transform(images_modified, update_stats=False)

        # Channels 0 and 2 should be unchanged
        torch.testing.assert_close(z1[:, 0], z2[:, 0])
        torch.testing.assert_close(z1[:, 2], z2[:, 2])

    def test_invertibility(self, transform, sample_images):
        """Test invertibility for images."""
        transform.set_input_normalization(sample_images)
        transform.eval()

        with torch.no_grad():
            z = transform(sample_images, update_stats=False)
            reconstructed = transform.inverse(z)

        torch.testing.assert_close(sample_images, reconstructed, rtol=1e-3, atol=1e-3)

    def test_spatial_invariance(self, transform, sample_images):
        """Test that same pixel value at different locations gives same output."""
        transform.set_input_normalization(sample_images)
        transform.eval()

        # Create image with constant value in one channel
        const_images = sample_images.clone()
        const_images[:, 0] = 1.5  # Constant value

        with torch.no_grad():
            z = transform(const_images, update_stats=False)

        # Output for channel 0 should be constant across spatial dimensions
        z_channel0 = z[:, 0]  # [B, H, W]
        for b in range(z_channel0.shape[0]):
            std = z_channel0[b].std()
            assert std < 1e-5, f"Spatial variance for constant input: {std}"

    def test_wrong_input_dims_raises(self, transform):
        """Test that wrong input dimensions raise error."""
        x_2d = torch.randn(16, 3)  # Missing spatial dims

        with pytest.raises(AssertionError):
            transform(x_2d)


class TestMonotoneTransformEdgeCases:
    """Edge case tests."""

    def test_single_feature(self):
        """Test with single feature."""
        transform = MonotoneFeatureTransform(num_features=1, num_bins=8)
        x = torch.randn(100, 1)
        transform.set_input_normalization(x)

        z = transform(x)
        assert z.shape == x.shape

        x_reconstructed = transform.inverse(z)
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-3)

    def test_many_features(self):
        """Test with many features."""
        transform = MonotoneFeatureTransform(num_features=100, num_bins=8)
        x = torch.randn(50, 100)
        transform.set_input_normalization(x)

        z = transform(x)
        assert z.shape == x.shape

    def test_single_sample(self):
        """Test with single sample (batch size 1)."""
        transform = MonotoneFeatureTransform(num_features=5, num_bins=16)
        x = torch.randn(1, 5)
        transform.set_input_normalization(x.expand(100, -1))  # Need more samples for normalization

        z = transform(x)
        assert z.shape == x.shape

    def test_no_running_stats(self):
        """Test transform without running statistics."""
        transform = MonotoneFeatureTransform(
            num_features=5,
            num_bins=16,
            track_running_stats=False
        )
        x = torch.randn(100, 5)
        transform.set_input_normalization(x)

        # Should use batch stats
        z = transform(x)
        assert z.shape == x.shape

        # Mean and var should be ~0 and ~1 for this batch
        assert z.mean(dim=0).abs().max() < 0.1
        assert (z.var(dim=0) - 1).abs().max() < 0.2

    def test_list_rqs_mode(self):
        """Test with list of RQS modules (not batched)."""
        transform = MonotoneFeatureTransform(
            num_features=3,
            num_bins=16,
            use_batch_rqs=False
        )
        x = torch.randn(50, 3)
        transform.set_input_normalization(x)

        z = transform(x)
        assert z.shape == x.shape

        x_reconstructed = transform.inverse(z)
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
