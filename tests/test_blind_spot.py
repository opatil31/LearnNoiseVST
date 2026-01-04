"""
Tests for blind-spot denoisers.

The key property to verify is:
    ∂ẑ_j/∂z_j ≈ 0 for all j

This ensures the denoiser output at position j doesn't depend on input at j.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))


class TestLeaveOneOutPooling:
    """Tests for leave-one-out pooling."""

    @pytest.fixture
    def loo_pool(self):
        from src.denoisers.tabular.loo_pooling import LeaveOneOutPooling
        return LeaveOneOutPooling()

    def test_output_shape(self, loo_pool):
        """Test output shape matches input."""
        e = torch.randn(16, 10, 64)  # [B, d, H]
        c = loo_pool(e)
        assert c.shape == e.shape

    def test_loo_property(self, loo_pool):
        """Test that c_f doesn't depend on e_f."""
        B, d, H = 4, 5, 8
        e = torch.randn(B, d, H, requires_grad=True)
        c = loo_pool(e)

        # For each feature f, check ∂c_f/∂e_f = 0
        for f in range(d):
            for b in range(B):
                for h in range(H):
                    grad = torch.autograd.grad(
                        c[b, f, h], e, retain_graph=True
                    )[0]
                    # Diagonal element should be zero
                    assert grad[b, f, h].abs() < 1e-6, \
                        f"LOO violated at b={b}, f={f}, h={h}: grad={grad[b, f, h]}"

    def test_loo_mean_value(self, loo_pool):
        """Test that LOO mean is correct."""
        # Simple case: e = [[1, 2, 3, 4, 5]]
        e = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])  # [1, 5, 1]
        c = loo_pool(e)

        # c_0 should be mean of [2,3,4,5] = 3.5
        # c_1 should be mean of [1,3,4,5] = 3.25
        # c_2 should be mean of [1,2,4,5] = 3.0
        # c_3 should be mean of [1,2,3,5] = 2.75
        # c_4 should be mean of [1,2,3,4] = 2.5

        expected = torch.tensor([[[3.5], [3.25], [3.0], [2.75], [2.5]]])
        torch.testing.assert_close(c, expected)

    def test_with_mask(self, loo_pool):
        """Test LOO pooling with missingness mask."""
        e = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])  # [1, 5, 1]
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])  # Feature 2 is missing

        c = loo_pool(e, mask)

        # For f=0: mean of [2, 4, 5] (skip 3 because masked) = 11/3 ≈ 3.67
        # Note: masked feature's embedding is zeroed out
        # So sum = 1 + 2 + 0 + 4 + 5 = 12, count = 4
        # c_0 = (12 - 1) / 3 = 11/3

        assert c[0, 0, 0].item() == pytest.approx(11/3, rel=1e-5)


class TestTabularDenoiser:
    """Tests for tabular blind-spot denoiser."""

    @pytest.fixture
    def denoiser(self):
        from src.denoisers.tabular import LightweightTabularDenoiser
        return LightweightTabularDenoiser(num_features=10, embed_dim=16, hidden_dim=32)

    def test_output_shape(self, denoiser):
        """Test output shape matches input."""
        z = torch.randn(8, 10)
        z_hat = denoiser(z)
        assert z_hat.shape == z.shape

    def test_blind_spot_property(self, denoiser):
        """Test blind-spot property: ∂ẑ_f/∂z_f = 0."""
        z = torch.randn(2, 10, requires_grad=True)
        z_hat = denoiser(z)

        # Check diagonal elements of Jacobian are near zero
        max_diag = 0.0
        for b in range(2):
            for f in range(10):
                grad = torch.autograd.grad(
                    z_hat[b, f], z, retain_graph=True
                )[0]
                diag = grad[b, f].abs().item()
                max_diag = max(max_diag, diag)

        assert max_diag < 1e-5, f"Blind-spot violated: max diagonal = {max_diag}"

    def test_gradient_flow(self, denoiser):
        """Test that gradients flow properly."""
        z = torch.randn(8, 10, requires_grad=True)
        z_hat = denoiser(z)
        loss = z_hat.sum()
        loss.backward()

        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_deterministic(self, denoiser):
        """Test that forward pass is deterministic."""
        denoiser.eval()
        z = torch.randn(4, 10)

        z_hat1 = denoiser(z)
        z_hat2 = denoiser(z)

        torch.testing.assert_close(z_hat1, z_hat2)


class TestFullTabularDenoiser:
    """Tests for the full tabular denoiser with FiLM."""

    @pytest.fixture
    def denoiser(self):
        from src.denoisers.tabular import TabularBlindSpotDenoiser
        # Import from tabular_denoiser directly to avoid name collision
        from src.denoisers.tabular.tabular_denoiser import TabularBlindSpotDenoiser as FullDenoiser
        return FullDenoiser(
            num_features=8,
            embed_dim=32,
            hidden_dim=64,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )

    def test_output_shape(self, denoiser):
        z = torch.randn(4, 8)
        z_hat = denoiser(z)
        assert z_hat.shape == z.shape

    def test_blind_spot_property(self, denoiser):
        """Verify blind-spot for full denoiser."""
        z = torch.randn(2, 8, requires_grad=True)
        z_hat = denoiser(z)

        max_diag = 0.0
        for b in range(2):
            for f in range(8):
                grad = torch.autograd.grad(
                    z_hat[b, f], z, retain_graph=True
                )[0]
                diag = grad[b, f].abs().item()
                max_diag = max(max_diag, diag)

        assert max_diag < 1e-5, f"Blind-spot violated: max diagonal = {max_diag}"


class TestImagingBlindSpotConv:
    """Tests for upward-only convolution layers."""

    @pytest.fixture
    def upward_conv(self):
        from src.denoisers.imaging.blind_spot_conv import UpwardOnlyConv2dSame
        return UpwardOnlyConv2dSame(3, 16, kernel_size=3)

    def test_output_shape(self, upward_conv):
        """Test that 'same' padding preserves spatial dims."""
        x = torch.randn(2, 3, 32, 32)
        y = upward_conv(x)
        assert y.shape[2:] == x.shape[2:], f"Expected {x.shape[2:]}, got {y.shape[2:]}"

    def test_upward_only_receptive_field(self, upward_conv):
        """Test that conv only looks upward."""
        # Create input where we modify bottom rows
        x1 = torch.randn(1, 3, 8, 8)
        x2 = x1.clone()
        x2[:, :, -2:, :] = torch.randn(1, 3, 2, 8)  # Modify bottom 2 rows

        y1 = upward_conv(x1)
        y2 = upward_conv(x2)

        # Top rows of output should be identical
        # (they can't see the modified bottom rows)
        # Actually, for upward-only, output at row i depends on input at rows <= i
        # So changing bottom rows shouldn't affect output at top rows
        top_rows = 4
        torch.testing.assert_close(
            y1[:, :, :top_rows, :],
            y2[:, :, :top_rows, :],
            rtol=1e-5, atol=1e-5
        )


class TestShiftDown:
    """Tests for shift down operation."""

    @pytest.fixture
    def shift(self):
        from src.denoisers.imaging.blind_spot_conv import ShiftDown
        return ShiftDown(shift=1)

    def test_output_shape(self, shift):
        """Test that shift preserves shape."""
        x = torch.randn(2, 3, 16, 16)
        y = shift(x)
        assert y.shape == x.shape

    def test_shift_effect(self, shift):
        """Test that content is shifted down."""
        x = torch.arange(16).float().view(1, 1, 4, 4)
        y = shift(x)

        # First row should be zeros (padded)
        assert (y[:, :, 0, :] == 0).all()

        # Other rows should be shifted versions of input
        torch.testing.assert_close(y[:, :, 1:, :], x[:, :, :-1, :])


class TestRotationBlindSpotNet:
    """Tests for rotation-based blind-spot network."""

    @pytest.fixture
    def net(self):
        from src.denoisers.imaging import LightweightRotationBlindSpotNet
        return LightweightRotationBlindSpotNet(
            in_channels=1,
            hidden_channels=8,
            num_blocks=2,
        )

    def test_output_shape(self, net):
        """Test output shape matches input."""
        z = torch.randn(2, 1, 32, 32)
        z_hat = net(z)
        assert z_hat.shape == z.shape

    def test_blind_spot_property_sampling(self, net):
        """Test blind-spot by sampling diagonal Jacobian elements."""
        z = torch.randn(1, 1, 16, 16, requires_grad=True)
        z_hat = net(z)

        # Sample random positions
        max_diag = 0.0
        num_samples = 50

        for _ in range(num_samples):
            h = torch.randint(0, 16, (1,)).item()
            w = torch.randint(0, 16, (1,)).item()

            grad = torch.autograd.grad(
                z_hat[0, 0, h, w], z, retain_graph=True
            )[0]

            diag = grad[0, 0, h, w].abs().item()
            max_diag = max(max_diag, diag)

        # Allow some tolerance due to numerical precision
        assert max_diag < 1e-4, f"Blind-spot violated: max diagonal = {max_diag}"

    def test_gradient_flow(self, net):
        """Test gradient flow."""
        z = torch.randn(2, 1, 16, 16, requires_grad=True)
        z_hat = net(z)
        loss = z_hat.sum()
        loss.backward()

        assert z.grad is not None
        assert torch.isfinite(z.grad).all()


class TestBlindSpotVerification:
    """Integration tests for blind-spot verification method."""

    def test_verify_no_leakage_tabular(self):
        """Test verify_no_leakage for tabular."""
        from src.denoisers.tabular import LightweightTabularDenoiser

        denoiser = LightweightTabularDenoiser(num_features=5)
        z = torch.randn(4, 5)

        passed, max_diag = denoiser.verify_no_leakage(z, num_samples=20)

        assert passed, f"Leakage detected: max_diag = {max_diag}"

    def test_verify_no_leakage_imaging(self):
        """Test verify_no_leakage for imaging."""
        from src.denoisers.imaging import LightweightRotationBlindSpotNet

        net = LightweightRotationBlindSpotNet(in_channels=1, hidden_channels=8, num_blocks=2)
        z = torch.randn(2, 1, 16, 16)

        passed, max_diag = net.verify_no_leakage(z, num_samples=30, threshold=1e-4)

        assert passed, f"Leakage detected: max_diag = {max_diag}"


class TestDeepTabularDenoiser:
    """Tests for deeper tabular denoiser with attention."""

    @pytest.fixture
    def denoiser(self):
        from src.denoisers.tabular.tabular_denoiser import DeepTabularDenoiser
        return DeepTabularDenoiser(
            num_features=6,
            embed_dim=32,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
        )

    def test_output_shape(self, denoiser):
        z = torch.randn(4, 6)
        z_hat = denoiser(z)
        assert z_hat.shape == z.shape

    def test_blind_spot_property(self, denoiser):
        """Verify blind-spot for deep denoiser."""
        z = torch.randn(2, 6, requires_grad=True)
        z_hat = denoiser(z)

        max_diag = 0.0
        for b in range(2):
            for f in range(6):
                grad = torch.autograd.grad(
                    z_hat[b, f], z, retain_graph=True
                )[0]
                diag = grad[b, f].abs().item()
                max_diag = max(max_diag, diag)

        # Attention-based LOO also satisfies blind-spot due to diagonal masking
        assert max_diag < 1e-5, f"Blind-spot violated: max diagonal = {max_diag}"


class TestLeaveOneOutAttentionPooling:
    """Tests for attention-based LOO pooling."""

    @pytest.fixture
    def attn_pool(self):
        from src.denoisers.tabular.loo_pooling import LeaveOneOutAttentionPooling
        return LeaveOneOutAttentionPooling(embed_dim=16, num_heads=2)

    def test_output_shape(self, attn_pool):
        e = torch.randn(4, 8, 16)
        c = attn_pool(e)
        assert c.shape == e.shape

    def test_loo_property(self, attn_pool):
        """Test that attention LOO satisfies blind-spot."""
        e = torch.randn(2, 5, 16, requires_grad=True)
        c = attn_pool(e)

        max_diag = 0.0
        for b in range(2):
            for f in range(5):
                for h in range(16):
                    grad = torch.autograd.grad(
                        c[b, f, h], e, retain_graph=True
                    )[0]
                    diag = grad[b, f, h].abs().item()
                    max_diag = max(max_diag, diag)

        assert max_diag < 1e-5, f"LOO attention violated: max diagonal = {max_diag}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
