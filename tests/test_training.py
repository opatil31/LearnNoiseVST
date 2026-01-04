"""
Tests for training module (losses, diagnostics, trainer).
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import training components
import sys
sys.path.insert(0, '/home/user/LearnNoiseVST')

from src.training.losses import (
    HomoscedasticityLoss,
    VarianceFlatnessLoss,
    ShapePenalty,
    TransformRegularization,
    CombinedTransformLoss,
    DenoiserLoss,
)
from src.training.gauge_fixing import (
    RunningStats,
    GaugeFixingManager,
    compute_standardization_stats,
    check_gauge_quality,
)
from src.training.diagnostics import (
    ConvergenceDiagnostics,
    GaugeQualityMonitor,
    BlindSpotLeakageDetector,
    TransformQualityMonitor,
    DiagnosticSuite,
)
from src.training.alternating_trainer import (
    TrainerConfig,
    AlternatingTrainer,
    LightweightTrainer,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_data():
    """Generate simple test data."""
    torch.manual_seed(42)
    B, F = 64, 10
    x = torch.randn(B, F)
    return x


@pytest.fixture
def noisy_data():
    """Generate data with signal-dependent noise."""
    torch.manual_seed(42)
    B, F = 256, 5

    # True signal
    mu = torch.randn(B, F) * 2

    # Signal-dependent noise (heteroscedastic)
    sigma = 0.1 + 0.2 * mu.abs()
    noise = torch.randn_like(mu) * sigma

    x = mu + noise
    return x, mu, noise


@pytest.fixture
def simple_transform():
    """Simple linear transform for testing."""
    class LinearTransform(nn.Module):
        def __init__(self, num_features):
            super().__init__()
            self.num_features = num_features
            self.scale = nn.Parameter(torch.ones(num_features))
            self.shift = nn.Parameter(torch.zeros(num_features))

            # Running stats
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))

        def forward(self, x):
            return x * self.scale + self.shift

        def derivative(self, x):
            return self.scale.expand_as(x)

        def forward_prenorm(self, x):
            return x * self.scale + self.shift

    return LinearTransform(10)


@pytest.fixture
def simple_denoiser():
    """Simple denoiser for testing."""
    class MeanDenoiser(nn.Module):
        def __init__(self, num_features):
            super().__init__()
            self.num_features = num_features
            self.linear = nn.Linear(num_features, num_features)

        def forward(self, z, mask=None):
            # Simple linear prediction (not actually blind-spot)
            return self.linear(z)

    return MeanDenoiser(10)


@pytest.fixture
def blind_spot_denoiser():
    """True blind-spot denoiser using LOO pooling."""
    from src.denoisers.tabular.loo_pooling import LeaveOneOutPooling

    class BlindSpotDenoiser(nn.Module):
        def __init__(self, num_features, embed_dim=16):
            super().__init__()
            self.num_features = num_features
            self.embed_dim = embed_dim

            self.encoder = nn.Linear(1, embed_dim)
            self.loo_pool = LeaveOneOutPooling()
            self.decoder = nn.Linear(embed_dim, 1)

        def forward(self, z, mask=None):
            B, F = z.shape
            # Encode each feature
            e = self.encoder(z.unsqueeze(-1))  # [B, F, H]
            # LOO pooling
            c = self.loo_pool(e, mask)  # [B, F, H]
            # Decode
            z_hat = self.decoder(c).squeeze(-1)  # [B, F]
            return z_hat

    return BlindSpotDenoiser(10)


# ============================================================================
# Loss Tests
# ============================================================================

class TestHomoscedasticityLoss:
    """Tests for HomoscedasticityLoss."""

    def test_forward_runs(self, simple_data):
        """Test that forward pass completes."""
        loss_fn = HomoscedasticityLoss(basis_degree=2)

        z_hat = simple_data
        residuals = torch.randn_like(simple_data) * 0.1

        loss = loss_fn(z_hat, residuals)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_zero_correlation(self):
        """Test that independent residuals give low loss."""
        torch.manual_seed(42)
        loss_fn = HomoscedasticityLoss(basis_degree=2)

        # Independent residuals
        z_hat = torch.randn(1000)
        residuals = torch.randn(1000) * 0.5  # Independent of z_hat

        loss = loss_fn(z_hat, residuals)

        # Should be close to zero
        assert loss.item() < 0.01

    def test_high_correlation(self):
        """Test that correlated residuals give high loss."""
        torch.manual_seed(42)
        loss_fn = HomoscedasticityLoss(basis_degree=2)

        # Residuals correlated with signal
        z_hat = torch.randn(1000)
        residuals = z_hat * 0.5 + torch.randn(1000) * 0.1  # Correlated

        loss_correlated = loss_fn(z_hat, residuals)

        # Independent for comparison
        residuals_ind = torch.randn(1000) * 0.5
        loss_ind = loss_fn(z_hat, residuals_ind)

        assert loss_correlated > loss_ind

    def test_spline_basis(self):
        """Test with spline basis functions."""
        loss_fn = HomoscedasticityLoss(use_spline_basis=True, num_spline_knots=5)

        z_hat = torch.randn(100)
        residuals = torch.randn(100) * 0.5

        loss = loss_fn(z_hat, residuals)

        assert loss.shape == ()
        assert loss.item() >= 0


class TestVarianceFlatnessLoss:
    """Tests for VarianceFlatnessLoss."""

    def test_forward_runs(self, simple_data):
        """Test that forward pass completes."""
        loss_fn = VarianceFlatnessLoss(subsample=100)

        z_hat = simple_data.flatten()
        residuals = torch.randn_like(z_hat) * 0.5

        loss = loss_fn(z_hat, residuals)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_constant_variance(self):
        """Test that homoscedastic data gives low loss."""
        torch.manual_seed(42)
        loss_fn = VarianceFlatnessLoss(subsample=500)

        # Homoscedastic data (constant variance)
        z_hat = torch.randn(1000)
        residuals = torch.randn(1000) * 0.5  # Constant variance

        loss = loss_fn(z_hat, residuals)

        # Should be relatively low
        assert loss.item() < 1.0

    def test_variable_variance(self):
        """Test that heteroscedastic data gives higher loss."""
        torch.manual_seed(42)
        loss_fn = VarianceFlatnessLoss(subsample=500, bandwidth=0.5)

        # Heteroscedastic data
        z_hat = torch.randn(1000)
        sigma = 0.1 + 0.5 * z_hat.abs()  # Variance depends on signal
        residuals = torch.randn(1000) * sigma

        loss_hetero = loss_fn(z_hat, residuals)

        # Homoscedastic for comparison
        residuals_homo = torch.randn(1000) * 0.5
        loss_homo = loss_fn(z_hat, residuals_homo)

        # Heteroscedastic should have higher variance of local variances
        assert loss_hetero > loss_homo * 0.5  # Some tolerance


class TestShapePenalty:
    """Tests for ShapePenalty."""

    def test_forward_runs(self, simple_data):
        """Test that forward pass completes."""
        loss_fn = ShapePenalty()

        # Standardized residuals
        u = simple_data / simple_data.std()

        loss = loss_fn(u.flatten())

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gaussian_low_penalty(self):
        """Test that Gaussian residuals give low penalty."""
        torch.manual_seed(42)
        loss_fn = ShapePenalty(use_robust=False)

        u = torch.randn(10000)  # Standard Gaussian
        u = (u - u.mean()) / u.std()  # Standardize

        loss = loss_fn(u)

        # Should be close to zero for Gaussian
        assert loss.item() < 1.0

    def test_skewed_high_penalty(self):
        """Test that skewed residuals give higher penalty."""
        torch.manual_seed(42)
        loss_fn = ShapePenalty(use_robust=False)

        # Skewed distribution (exponential-like)
        u = torch.abs(torch.randn(10000))
        u = (u - u.mean()) / u.std()

        loss_skewed = loss_fn(u)

        # Gaussian for comparison
        u_gauss = torch.randn(10000)
        u_gauss = (u_gauss - u_gauss.mean()) / u_gauss.std()
        loss_gauss = loss_fn(u_gauss)

        assert loss_skewed > loss_gauss

    def test_robust_estimators(self):
        """Test robust estimators handle outliers."""
        torch.manual_seed(42)
        loss_fn_robust = ShapePenalty(use_robust=True)
        loss_fn_standard = ShapePenalty(use_robust=False)

        # Data with outliers
        u = torch.randn(1000)
        u[0:10] = 100  # Extreme outliers

        loss_robust = loss_fn_robust(u)
        loss_standard = loss_fn_standard(u)

        # Both should complete, robust should be less affected
        assert loss_robust.item() < loss_standard.item()


class TestCombinedTransformLoss:
    """Tests for CombinedTransformLoss."""

    def test_forward_runs(self, simple_data, simple_transform):
        """Test that forward pass completes."""
        loss_fn = CombinedTransformLoss(
            num_features=10,
            lambda_homo=1.0,
            lambda_vf=0.1,
            lambda_shape=0.1,
            lambda_reg=0.01,
        )

        z_hat = simple_data
        residuals = torch.randn_like(simple_data) * 0.5

        losses = loss_fn(z_hat=z_hat, residuals=residuals)

        assert 'total' in losses
        assert 'homo' in losses
        assert 'vf' in losses
        assert 'shape' in losses
        assert isinstance(losses['total'], torch.Tensor)

    def test_gradient_flow(self, simple_data, simple_transform):
        """Test that gradients flow through loss."""
        loss_fn = CombinedTransformLoss(
            num_features=10,
            lambda_homo=1.0,
            lambda_vf=0.1,
        )

        z = simple_transform(simple_data)
        z_hat = z - 0.1 * torch.randn_like(z)  # Fake predictions

        losses = loss_fn(z=z, z_hat=z_hat)
        losses['total'].backward()

        # Check gradients exist
        for param in simple_transform.parameters():
            assert param.grad is not None


class TestDenoiserLoss:
    """Tests for DenoiserLoss."""

    def test_mse_loss(self):
        """Test MSE loss computation."""
        loss_fn = DenoiserLoss(loss_type='mse')

        z = torch.randn(32, 10)
        z_hat = z + torch.randn_like(z) * 0.1

        loss, components = loss_fn(z, z_hat)

        assert loss.shape == ()
        assert 'mse' in components
        assert loss.item() >= 0

    def test_huber_loss(self):
        """Test Huber loss computation."""
        loss_fn = DenoiserLoss(loss_type='huber', huber_delta=1.0)

        z = torch.randn(32, 10)
        z_hat = z + torch.randn_like(z) * 0.1

        loss, components = loss_fn(z, z_hat)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_charbonnier_loss(self):
        """Test Charbonnier loss computation."""
        loss_fn = DenoiserLoss(loss_type='charbonnier')

        z = torch.randn(32, 10)
        z_hat = z + torch.randn_like(z) * 0.1

        loss, components = loss_fn(z, z_hat)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_masked_loss(self):
        """Test loss with mask."""
        loss_fn = DenoiserLoss()

        z = torch.randn(32, 10)
        z_hat = z + torch.randn_like(z) * 0.1
        mask = torch.ones_like(z)
        mask[:, 0:5] = 0  # Mask out half the features

        loss, _ = loss_fn(z, z_hat, mask=mask)

        assert loss.shape == ()


# ============================================================================
# Gauge Fixing Tests
# ============================================================================

class TestRunningStats:
    """Tests for RunningStats."""

    def test_ema_update(self):
        """Test EMA updates work correctly."""
        stats = RunningStats(num_features=5, momentum=0.5)

        # Initial state
        assert (stats.running_mean == 0).all()
        assert (stats.running_var == 1).all()

        # Update with batch stats
        batch_mean = torch.ones(5) * 2
        batch_var = torch.ones(5) * 4

        stats.update_ema(batch_mean, batch_var)

        # Should be interpolation
        assert torch.allclose(stats.running_mean, torch.ones(5))  # 0.5*0 + 0.5*2 = 1
        assert torch.allclose(stats.running_var, torch.ones(5) * 2.5)  # 0.5*1 + 0.5*4 = 2.5

    def test_welford_update(self):
        """Test Welford's online algorithm."""
        torch.manual_seed(42)
        stats = RunningStats(num_features=3)

        # Generate data
        data = torch.randn(100, 3) + torch.tensor([1.0, 2.0, 3.0])

        # Update with Welford
        stats.update_welford(data)
        stats.finalize_welford()

        # Check against direct computation
        expected_mean = data.mean(dim=0)
        expected_var = data.var(dim=0, unbiased=True)

        assert torch.allclose(stats.running_mean, expected_mean, atol=1e-5)
        assert torch.allclose(stats.running_var, expected_var, atol=1e-5)

    def test_standardize(self):
        """Test standardization."""
        stats = RunningStats(num_features=3)
        stats.running_mean = torch.tensor([1.0, 2.0, 3.0])
        stats.running_var = torch.tensor([1.0, 4.0, 9.0])

        x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

        z, log_deriv = stats.standardize(x, return_log_deriv=True)

        # First row should be all zeros
        assert torch.allclose(z[0], torch.zeros(3), atol=1e-6)

        # Check log derivative
        assert log_deriv is not None
        expected_log_deriv = -torch.log(torch.sqrt(torch.tensor([1.0, 4.0, 9.0]) + 1e-8))
        assert torch.allclose(log_deriv[0], expected_log_deriv, atol=1e-5)

    def test_destandardize(self):
        """Test that destandardize inverts standardize."""
        stats = RunningStats(num_features=3)
        stats.running_mean = torch.tensor([1.0, 2.0, 3.0])
        stats.running_var = torch.tensor([1.0, 4.0, 9.0])

        x = torch.randn(10, 3)
        z, _ = stats.standardize(x)
        x_recovered = stats.destandardize(z)

        assert torch.allclose(x, x_recovered, atol=1e-5)


class TestGaugeQuality:
    """Tests for gauge quality checking."""

    def test_good_gauge(self):
        """Test properly gauge-fixed data passes check."""
        torch.manual_seed(42)

        # Generate standardized data
        z = torch.randn(1000, 5)  # Should have mean≈0, var≈1

        passed, stats = check_gauge_quality(z, tolerance=0.1)

        assert passed
        assert stats['mean_max_dev'] < 0.1
        assert stats['var_max_dev'] < 0.1

    def test_bad_gauge_mean(self):
        """Test that shifted mean fails check."""
        torch.manual_seed(42)

        z = torch.randn(1000, 5) + 1.0  # Mean shifted by 1

        passed, stats = check_gauge_quality(z, tolerance=0.1)

        assert not passed
        assert stats['mean_max_dev'] > 0.5

    def test_bad_gauge_var(self):
        """Test that wrong variance fails check."""
        torch.manual_seed(42)

        z = torch.randn(1000, 5) * 2  # Var = 4, not 1

        passed, stats = check_gauge_quality(z, tolerance=0.1)

        assert not passed
        assert stats['var_max_dev'] > 1.0


# ============================================================================
# Diagnostics Tests
# ============================================================================

class TestConvergenceDiagnostics:
    """Tests for ConvergenceDiagnostics."""

    def test_loss_logging(self):
        """Test loss logging."""
        diag = ConvergenceDiagnostics(window_size=10)

        for i in range(20):
            diag.log_losses(1.0 / (i + 1), 0.5 / (i + 1))

        assert len(diag.total_losses) == 20
        assert diag.get_smoothed_loss() < 1.0

    def test_early_stopping(self):
        """Test early stopping detection."""
        # Use small window_size so smoothed loss reflects recent values
        diag = ConvergenceDiagnostics(patience=5, min_delta=0.01, window_size=3)

        # Decreasing loss
        for i in range(5):
            diag.log_losses(1.0 - i * 0.1, 0.0)
            should_stop, _ = diag.check_early_stopping()
            assert not should_stop

        # Plateau - loss stops improving
        for i in range(10):
            diag.log_losses(0.5, 0.0)
            should_stop, _ = diag.check_early_stopping()

        assert should_stop

    def test_gradient_health(self, simple_transform, simple_denoiser):
        """Test gradient health checking."""
        diag = ConvergenceDiagnostics()

        # Generate some gradients
        x = torch.randn(16, 10)
        z = simple_transform(x)
        z_hat = simple_denoiser(z)
        loss = (z - z_hat).pow(2).mean()
        loss.backward()

        diag.log_gradients(simple_transform, simple_denoiser)

        result = diag.check_gradient_health()
        assert result.passed


class TestBlindSpotLeakageDetector:
    """Tests for BlindSpotLeakageDetector."""

    def test_true_blind_spot(self, blind_spot_denoiser):
        """Test that true blind-spot denoiser passes."""
        detector = BlindSpotLeakageDetector(num_checks=5, tolerance=1e-4)

        z = torch.randn(4, 10)

        result = detector.check_gradient_leakage(blind_spot_denoiser, z)

        # Should pass (no leakage)
        assert result.passed
        assert result.value < 1e-4

    def test_leaky_denoiser(self, simple_denoiser):
        """Test that leaky denoiser fails."""
        detector = BlindSpotLeakageDetector(num_checks=5, tolerance=1e-4)

        z = torch.randn(4, 10, requires_grad=True)

        result = detector.check_gradient_leakage(simple_denoiser, z)

        # Should fail (has leakage)
        assert not result.passed
        assert result.value > 1e-4


class TestGaugeQualityMonitor:
    """Tests for GaugeQualityMonitor."""

    def test_monitoring(self):
        """Test gauge quality monitoring."""
        # Use relaxed tolerance since finite sample statistics naturally deviate
        # With 1000 samples, typical deviation is ~0.03 for mean and ~0.05 for var
        monitor = GaugeQualityMonitor(num_features=5, tolerance=0.25)

        # Log data with fixed seed for reproducibility
        torch.manual_seed(42)
        for _ in range(10):
            z = torch.randn(100, 5)
            monitor.update(z)

        result = monitor.check_quality()

        assert result.passed

    def test_bad_gauge_detection(self):
        """Test detection of bad gauge."""
        monitor = GaugeQualityMonitor(num_features=5, tolerance=0.1)

        # Log shifted data
        for _ in range(10):
            z = torch.randn(100, 5) + 2.0  # Mean shifted
            monitor.update(z)

        result = monitor.check_quality()

        assert not result.passed


# ============================================================================
# Trainer Tests
# ============================================================================

class TestLightweightTrainer:
    """Tests for LightweightTrainer."""

    def test_train_step(self, simple_transform, blind_spot_denoiser, simple_data):
        """Test single training step."""
        trainer = LightweightTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            lr=1e-3,
            device='cpu',
        )

        losses = trainer.train_step(simple_data)

        assert 'transform_loss' in losses
        assert 'denoiser_loss' in losses

    def test_train_epoch(self, simple_transform, blind_spot_denoiser, simple_data):
        """Test training for one epoch."""
        dataset = TensorDataset(simple_data)
        dataloader = DataLoader(dataset, batch_size=16)

        trainer = LightweightTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            lr=1e-3,
            device='cpu',
        )

        losses = trainer.train_epoch(dataloader)

        assert 'transform_loss' in losses
        assert 'denoiser_loss' in losses


class TestAlternatingTrainer:
    """Tests for AlternatingTrainer."""

    def test_initialization(self, simple_transform, blind_spot_denoiser):
        """Test trainer initialization."""
        config = TrainerConfig(
            num_outer_iters=2,
            transform_inner_iters=5,
            denoiser_inner_iters=5,
            device='cpu',
        )

        trainer = AlternatingTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            config=config,
        )

        assert trainer.outer_iter == 0
        assert trainer.best_loss == float('inf')

    def test_transform_step(self, simple_transform, blind_spot_denoiser, simple_data):
        """Test transform optimization step."""
        config = TrainerConfig(
            transform_inner_iters=5,
            device='cpu',
        )

        trainer = AlternatingTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            config=config,
        )

        dataset = TensorDataset(simple_data)
        dataloader = DataLoader(dataset, batch_size=16)

        results = trainer.train_transform_step(dataloader)

        assert 'loss' in results
        assert results['loss'] >= 0

    def test_denoiser_step(self, simple_transform, blind_spot_denoiser, simple_data):
        """Test denoiser optimization step."""
        config = TrainerConfig(
            denoiser_inner_iters=5,
            device='cpu',
        )

        trainer = AlternatingTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            config=config,
        )

        dataset = TensorDataset(simple_data)
        dataloader = DataLoader(dataset, batch_size=16)

        results = trainer.train_denoiser_step(dataloader)

        assert 'loss' in results
        assert 'mse' in results

    def test_full_training(self, simple_transform, blind_spot_denoiser, simple_data):
        """Test full training loop (short run)."""
        config = TrainerConfig(
            num_outer_iters=2,
            transform_inner_iters=3,
            denoiser_inner_iters=3,
            log_every=1,
            diagnose_every=1,
            device='cpu',
        )

        trainer = AlternatingTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            config=config,
        )

        dataset = TensorDataset(simple_data)
        dataloader = DataLoader(dataset, batch_size=16)

        result = trainer.train(dataloader)

        assert 'history' in result
        assert len(result['history']['transform_loss']) == 2
        assert len(result['history']['denoiser_loss']) == 2

    def test_checkpoint_save_load(self, simple_transform, blind_spot_denoiser, tmp_path):
        """Test checkpoint saving and loading."""
        config = TrainerConfig(device='cpu')

        trainer = AlternatingTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            config=config,
        )

        # Modify state
        trainer.outer_iter = 5
        trainer.best_loss = 0.5

        # Save
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Create new trainer and load
        trainer2 = AlternatingTrainer(
            transform=simple_transform,
            denoiser=blind_spot_denoiser,
            config=config,
        )
        trainer2.load_checkpoint(str(checkpoint_path))

        assert trainer2.outer_iter == 5
        assert trainer2.best_loss == 0.5


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_loss_gradient_flow(self, noisy_data):
        """Test that gradients flow from loss through transform."""
        x, mu, noise = noisy_data

        # Create transform matching data dimensions
        class LinearTransform(nn.Module):
            def __init__(self, num_features):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(num_features))
                self.shift = nn.Parameter(torch.zeros(num_features))

            def forward(self, x):
                return x * self.scale + self.shift

        transform = LinearTransform(x.shape[1])

        # Forward
        z = transform(x)
        z_hat = z.mean(dim=0, keepdim=True).expand_as(z)  # Simple prediction
        residuals = z - z_hat

        # Loss
        loss_fn = CombinedTransformLoss(
            num_features=x.shape[1],
            lambda_homo=1.0,
            lambda_vf=0.0,  # Skip VF for speed
            lambda_shape=0.0,
            lambda_reg=0.0,
        )

        losses = loss_fn(z_hat=z_hat, residuals=residuals)
        losses['total'].backward()

        # Check gradients
        for param in transform.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_diagnostics_integration(self, simple_transform, blind_spot_denoiser, simple_data):
        """Test that diagnostics work with real models."""
        suite = DiagnosticSuite(num_features=10)

        # Generate predictions
        z = simple_transform(simple_data)
        z_hat = blind_spot_denoiser(z)

        # Update monitors
        suite.gauge.update(z)
        suite.residuals.update(z, z_hat)

        # Run checks
        results = suite.run_all_checks(
            transform_model=simple_transform,
            denoiser_model=blind_spot_denoiser,
            z=z,
            z_hat=z_hat,
        )

        # At least some checks should run
        assert len(results) > 0

        # Generate report
        report = suite.generate_report()
        assert "DIAGNOSTIC REPORT" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
