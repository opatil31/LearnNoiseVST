"""
Tests for noise model module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '/home/user/LearnNoiseVST')

from src.noise_model.calibration import (
    ResidualData,
    CalibrationResult,
    CalibrationDatasetGenerator,
    ResidualAnalyzer,
)
from src.noise_model.location_scale import (
    LocationScaleModel,
    LocationScaleModelCollection,
    DeltaMethodVariancePropagator,
)
from src.noise_model.marginal_fit import (
    EmpiricalMarginal,
    GaussianMarginal,
    MarginalCollection,
)
from src.noise_model.copula import (
    GaussianCopula,
    IndependenceCopula,
    test_independence,
    choose_copula,
)
from src.noise_model.sampler import (
    NoiseModelConfig,
    NoiseModelSampler,
    NoiseAugmenter,
    validate_noise_model,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_residuals():
    """Generate simple residual data."""
    np.random.seed(42)
    n = 1000

    # Homoscedastic Gaussian
    z_hat = np.random.randn(n)
    r = np.random.randn(n) * 0.5

    return z_hat, r


@pytest.fixture
def heteroscedastic_residuals():
    """Generate residuals with signal-dependent variance."""
    np.random.seed(42)
    n = 1000

    z_hat = np.random.uniform(-3, 3, n)
    sigma = 0.2 + 0.1 * np.abs(z_hat)  # Variance depends on signal
    r = np.random.randn(n) * sigma

    return z_hat, r


@pytest.fixture
def multigroup_data():
    """Generate multi-group data (like image channels)."""
    np.random.seed(42)
    n = 500
    num_groups = 3

    data = {}
    for g in range(num_groups):
        z_hat = np.random.randn(n)
        r = np.random.randn(n) * (0.3 + 0.1 * g)
        data[g] = ResidualData(z_hat=z_hat, r=r)

    return CalibrationResult(
        residuals=data,
        is_image=False,
        num_groups=num_groups,
        total_samples=n,
    )


@pytest.fixture
def correlated_multigroup():
    """Generate multi-group data with correlation."""
    np.random.seed(42)
    n = 500
    d = 3

    # Correlation matrix
    rho = 0.6
    cov = np.array([
        [1.0, rho, rho/2],
        [rho, 1.0, rho],
        [rho/2, rho, 1.0],
    ])

    # Generate correlated residuals
    L = np.linalg.cholesky(cov)
    z = np.random.randn(n, d)
    u = z @ L.T

    data = {}
    for g in range(d):
        z_hat = np.random.randn(n)
        data[g] = ResidualData(z_hat=z_hat, r=u[:, g])

    return CalibrationResult(
        residuals=data,
        is_image=False,
        num_groups=d,
        total_samples=n,
    )


@pytest.fixture
def simple_transform():
    """Simple transform for testing."""
    class SimpleTransform(nn.Module):
        def __init__(self, num_features):
            super().__init__()
            self.num_features = num_features
            self.scale = nn.Parameter(torch.ones(num_features))

        def forward(self, x):
            return x * self.scale

        def log_derivative(self, x):
            return torch.log(self.scale).expand_as(x)

    return SimpleTransform(5)


@pytest.fixture
def simple_denoiser():
    """Simple denoiser for testing."""
    class SimpleDenoiser(nn.Module):
        def forward(self, z, mask=None):
            # Return mean across batch (not realistic but simple)
            return z - 0.1 * torch.randn_like(z)

    return SimpleDenoiser()


# ============================================================================
# Location-Scale Tests
# ============================================================================

class TestLocationScaleModel:
    """Tests for LocationScaleModel."""

    def test_fit_homoscedastic(self, simple_residuals):
        """Test fitting with constant variance."""
        z_hat, r = simple_residuals

        model = LocationScaleModel(
            variance_flatness_threshold=0.3,
        )
        result = model.fit(z_hat, r)

        # Should detect flat variance
        assert result.is_variance_flat
        assert result.sigma_constant is not None
        assert abs(result.sigma_constant - 0.5) < 0.1

    def test_fit_heteroscedastic(self, heteroscedastic_residuals):
        """Test fitting with signal-dependent variance."""
        z_hat, r = heteroscedastic_residuals

        model = LocationScaleModel(
            variance_flatness_threshold=0.1,
        )
        result = model.fit(z_hat, r)

        # May or may not detect as flat depending on strength
        # Just check it runs without error
        assert result.mu_function is not None
        assert result.sigma_function is not None

    def test_standardize_destandardize(self, simple_residuals):
        """Test that standardize/destandardize are inverses."""
        z_hat, r = simple_residuals

        model = LocationScaleModel()
        model.fit(z_hat, r)

        u = model.standardize(z_hat, r)
        r_recovered = model.destandardize(z_hat, u)

        np.testing.assert_allclose(r, r_recovered, rtol=1e-5)

    def test_standardized_stats(self, simple_residuals):
        """Test that standardized residuals have mean≈0, var≈1."""
        z_hat, r = simple_residuals

        model = LocationScaleModel()
        model.fit(z_hat, r)

        u = model.standardize(z_hat, r)

        assert abs(np.mean(u)) < 0.1
        assert abs(np.var(u) - 1) < 0.2

    def test_get_mu_sigma(self, simple_residuals):
        """Test mu and sigma accessors."""
        z_hat, r = simple_residuals

        model = LocationScaleModel()
        model.fit(z_hat, r)

        mu = model.get_mu(z_hat)
        sigma = model.get_sigma(z_hat)

        assert mu.shape == z_hat.shape
        assert sigma.shape == z_hat.shape
        assert (sigma > 0).all()


class TestLocationScaleCollection:
    """Tests for LocationScaleModelCollection."""

    def test_fit_multiple_groups(self, multigroup_data):
        """Test fitting models for multiple groups."""
        collection = LocationScaleModelCollection()

        data = {
            g: {'z_hat': d.z_hat, 'r': d.r}
            for g, d in multigroup_data
        }
        results = collection.fit(data)

        assert len(results) == 3
        assert 0 in collection
        assert 1 in collection
        assert 2 in collection

    def test_standardize_per_group(self, multigroup_data):
        """Test standardization for each group."""
        collection = LocationScaleModelCollection()

        data = {
            g: {'z_hat': d.z_hat, 'r': d.r}
            for g, d in multigroup_data
        }
        collection.fit(data)

        for g in range(3):
            u = collection.standardize(g, data[g]['z_hat'], data[g]['r'])
            assert abs(np.mean(u)) < 0.2
            assert abs(np.var(u) - 1) < 0.3


# ============================================================================
# Marginal Tests
# ============================================================================

class TestEmpiricalMarginal:
    """Tests for EmpiricalMarginal."""

    def test_fit(self):
        """Test basic fitting."""
        np.random.seed(42)
        u = np.random.randn(1000)

        marginal = EmpiricalMarginal()
        result = marginal.fit(u)

        assert result.mean is not None
        assert result.std is not None
        assert len(result.quantiles) > 0

    def test_cdf_range(self):
        """Test that CDF returns values in [0, 1]."""
        np.random.seed(42)
        u = np.random.randn(1000)

        marginal = EmpiricalMarginal()
        marginal.fit(u)

        test_values = np.linspace(-5, 5, 100)
        cdf_values = marginal.cdf(test_values)

        assert (cdf_values >= 0).all()
        assert (cdf_values <= 1).all()
        # CDF should be monotonically increasing
        assert (np.diff(cdf_values) >= -1e-6).all()

    def test_cdf_quantile_inverse(self):
        """Test that quantile is inverse of CDF."""
        np.random.seed(42)
        u = np.random.randn(1000)

        marginal = EmpiricalMarginal()
        marginal.fit(u)

        # Test at various probabilities
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = marginal.quantile(p)
        p_recovered = marginal.cdf(q)

        np.testing.assert_allclose(p, p_recovered, atol=0.05)

    def test_sample(self):
        """Test sampling."""
        np.random.seed(42)
        u = np.random.randn(10000)

        marginal = EmpiricalMarginal()
        marginal.fit(u)

        samples = marginal.sample(1000)

        assert samples.shape == (1000,)
        # Samples should have similar statistics to original
        assert abs(np.mean(samples)) < 0.2
        assert abs(np.std(samples) - 1) < 0.2

    def test_non_gaussian(self):
        """Test with non-Gaussian distribution."""
        np.random.seed(42)
        # Heavy-tailed distribution (t-distribution approximation)
        u = np.random.standard_t(df=5, size=1000)
        u = (u - np.mean(u)) / np.std(u)

        marginal = EmpiricalMarginal()
        result = marginal.fit(u)

        # Should detect non-Gaussian
        assert abs(result.kurtosis) > 0.5  # Excess kurtosis > 0 for t-dist


class TestGaussianMarginal:
    """Tests for GaussianMarginal."""

    def test_fit(self):
        """Test Gaussian fitting."""
        np.random.seed(42)
        u = np.random.randn(1000)

        marginal = GaussianMarginal()
        result = marginal.fit(u)

        assert abs(result.mean) < 0.1
        assert abs(result.std - 1) < 0.1

    def test_sample(self):
        """Test Gaussian sampling."""
        marginal = GaussianMarginal()
        marginal.mean = 0
        marginal.std = 1

        samples = marginal.sample(1000)
        assert abs(np.mean(samples)) < 0.2


class TestMarginalCollection:
    """Tests for MarginalCollection."""

    def test_fit_multiple(self):
        """Test fitting marginals for multiple groups."""
        np.random.seed(42)

        data = {
            0: np.random.randn(500),
            1: np.random.randn(500) * 2,
            2: np.random.randn(500) + 1,
        }

        collection = MarginalCollection()
        results = collection.fit(data)

        assert len(results) == 3
        assert 0 in collection
        assert 1 in collection
        assert 2 in collection


# ============================================================================
# Copula Tests
# ============================================================================

class TestGaussianCopula:
    """Tests for GaussianCopula."""

    def test_fit_independent(self):
        """Test fitting with independent data."""
        np.random.seed(42)
        n = 500
        d = 3

        u = np.random.randn(n, d)

        copula = GaussianCopula(shrinkage=0.1)
        result = copula.fit(u)

        # Correlation should be close to identity
        off_diag = result.correlation_matrix - np.eye(d)
        assert np.abs(off_diag).max() < 0.3

    def test_fit_correlated(self, correlated_multigroup):
        """Test fitting with correlated data."""
        # Build u_matrix from calibration data
        u_matrix = np.column_stack([
            correlated_multigroup.residuals[g].r
            for g in range(3)
        ])

        copula = GaussianCopula(shrinkage=0.1)
        result = copula.fit(u_matrix)

        # Should detect correlation
        assert result.correlation_matrix[0, 1] > 0.3

    def test_sample(self):
        """Test copula sampling."""
        np.random.seed(42)
        n = 1000
        d = 3

        # Create correlated data
        rho = 0.5
        cov = np.array([
            [1.0, rho, 0],
            [rho, 1.0, rho],
            [0, rho, 1.0],
        ])
        L = np.linalg.cholesky(cov)
        u = np.random.randn(n, d) @ L.T

        copula = GaussianCopula()
        copula.fit(u)

        samples = copula.sample(500)

        assert samples.shape == (500, d)
        # Samples should preserve correlation structure
        sample_corr = np.corrcoef(samples, rowvar=False)
        assert sample_corr[0, 1] > 0.2

    def test_positive_definiteness(self):
        """Test that correlation matrix is always positive definite."""
        np.random.seed(42)

        # Edge case: near-singular
        u = np.random.randn(50, 10)

        copula = GaussianCopula(shrinkage=0.2)
        result = copula.fit(u)

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(result.correlation_matrix)
        assert (eigenvalues > 0).all()


class TestIndependenceCopula:
    """Tests for IndependenceCopula."""

    def test_fit(self):
        """Test independence copula fitting."""
        u = np.random.randn(100, 3)

        copula = IndependenceCopula()
        result = copula.fit(u)

        # Correlation should be identity
        np.testing.assert_array_equal(result.correlation_matrix, np.eye(3))


class TestIndependenceTest:
    """Tests for test_independence function."""

    def test_independent_data(self):
        """Test detection of independent data."""
        np.random.seed(42)
        u = np.random.randn(500, 3)

        is_independent, details = test_independence(u)

        assert is_independent
        assert len(details['significant_pairs']) == 0

    def test_correlated_data(self):
        """Test detection of correlated data."""
        np.random.seed(42)
        n = 500

        # Create correlated data
        x = np.random.randn(n)
        u = np.column_stack([
            x,
            0.8 * x + 0.2 * np.random.randn(n),
            np.random.randn(n),
        ])

        is_independent, details = test_independence(u)

        assert not is_independent
        assert len(details['significant_pairs']) > 0


# ============================================================================
# Sampler Tests
# ============================================================================

class TestNoiseModelSampler:
    """Tests for NoiseModelSampler."""

    def test_fit(self, multigroup_data):
        """Test fitting the full noise model."""
        config = NoiseModelConfig(
            fit_copula=False,
        )
        sampler = NoiseModelSampler(config)
        result = sampler.fit(multigroup_data)

        assert result.num_groups == 3
        assert sampler.is_fitted

    def test_sample_shape(self, multigroup_data):
        """Test that sampled noise has correct shape."""
        sampler = NoiseModelSampler()
        sampler.fit(multigroup_data)

        z_hat = np.random.randn(10, 3)
        eps = sampler.sample(z_hat)

        assert eps.shape == z_hat.shape

    def test_sample_tensor(self, multigroup_data):
        """Test sampling with torch tensors."""
        sampler = NoiseModelSampler()
        sampler.fit(multigroup_data)

        z_hat = torch.randn(10, 3)
        eps = sampler.sample(z_hat)

        assert isinstance(eps, torch.Tensor)
        assert eps.shape == z_hat.shape

    def test_sample_statistics(self, multigroup_data):
        """Test that sampled noise has reasonable statistics."""
        sampler = NoiseModelSampler()
        sampler.fit(multigroup_data)

        # Sample many times
        z_hat = np.random.randn(1000, 3)
        eps = sampler.sample(z_hat)

        # Mean should be close to 0
        assert abs(np.mean(eps)) < 0.2

    def test_get_noise_std(self, multigroup_data):
        """Test noise standard deviation accessor."""
        sampler = NoiseModelSampler()
        sampler.fit(multigroup_data)

        z_hat = np.random.randn(10, 3)
        sigma = sampler.get_noise_std(z_hat)

        assert sigma.shape == z_hat.shape
        assert (sigma > 0).all()


class TestNoiseAugmenter:
    """Tests for NoiseAugmenter module."""

    def test_forward(self, multigroup_data):
        """Test forward pass."""
        sampler = NoiseModelSampler()
        sampler.fit(multigroup_data)

        augmenter = NoiseAugmenter(sampler)

        z_hat = torch.randn(10, 3)
        z = augmenter(z_hat)

        assert z.shape == z_hat.shape

    def test_noise_scale(self, multigroup_data):
        """Test noise scaling."""
        sampler = NoiseModelSampler()
        sampler.fit(multigroup_data)

        augmenter_full = NoiseAugmenter(sampler, noise_scale=1.0)
        augmenter_half = NoiseAugmenter(sampler, noise_scale=0.5)

        torch.manual_seed(42)
        z_hat = torch.randn(100, 3)

        # The variance of noise should scale with noise_scale^2
        # This is a statistical test, so we use large samples
        z_full = augmenter_full(z_hat)
        z_half = augmenter_half(z_hat)

        # Just check they produce different outputs
        assert not torch.allclose(z_full, z_half)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_tabular(self, simple_transform, simple_denoiser):
        """Test full pipeline for tabular data."""
        # Create test data
        torch.manual_seed(42)
        x = torch.randn(100, 5)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=32)

        # Generate calibration data
        generator = CalibrationDatasetGenerator(
            transform=simple_transform,
            denoiser=simple_denoiser,
            device='cpu',
        )
        calibration = generator.generate(loader, is_image=False)

        assert calibration.num_groups == 5
        assert calibration.total_samples == 100

        # Fit noise model
        sampler = NoiseModelSampler()
        result = sampler.fit(calibration)

        assert result.num_groups == 5
        assert sampler.is_fitted

        # Sample noise
        z_hat = torch.randn(10, 5)
        eps = sampler.sample(z_hat)

        assert eps.shape == z_hat.shape

    def test_residual_analyzer(self, multigroup_data):
        """Test residual analysis."""
        analyzer = ResidualAnalyzer(multigroup_data)

        stats = analyzer.compute_statistics()
        assert len(stats) == 3

        homo_scores = analyzer.check_homoscedasticity()
        assert len(homo_scores) == 3

        normality = analyzer.check_normality()
        assert len(normality) == 3

        report = analyzer.get_diagnostic_report()
        assert "DIAGNOSTIC REPORT" in report

    def test_validation(self, multigroup_data):
        """Test noise model validation."""
        sampler = NoiseModelSampler()
        sampler.fit(multigroup_data)

        results = validate_noise_model(sampler, multigroup_data)

        assert len(results) == 3
        for g, r in results.items():
            assert 'real_mean' in r
            assert 'sampled_mean' in r


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_small_sample(self):
        """Test with very small sample size."""
        np.random.seed(42)

        z_hat = np.random.randn(20)
        r = np.random.randn(20) * 0.5

        model = LocationScaleModel()
        result = model.fit(z_hat, r)

        # Should still work
        assert result.sigma_function is not None

    def test_constant_residuals(self):
        """Test with near-constant residuals."""
        z_hat = np.random.randn(100)
        r = np.zeros(100) + 0.001 * np.random.randn(100)

        model = LocationScaleModel()
        result = model.fit(z_hat, r)

        assert result.is_variance_flat

    def test_outliers(self):
        """Test robustness to outliers."""
        np.random.seed(42)

        z_hat = np.random.randn(1000)
        r = np.random.randn(1000) * 0.5

        # Add outliers
        r[0:10] = 100

        model = LocationScaleModel(use_robust=True)
        result = model.fit(z_hat, r)

        # Robust estimate should not be too affected
        assert result.sigma_constant is not None
        assert result.sigma_constant < 2  # Should not be inflated by outliers

    def test_single_group(self):
        """Test with single group."""
        np.random.seed(42)

        data = {
            0: ResidualData(
                z_hat=np.random.randn(100),
                r=np.random.randn(100) * 0.5,
            )
        }
        calibration = CalibrationResult(
            residuals=data,
            is_image=False,
            num_groups=1,
            total_samples=100,
        )

        sampler = NoiseModelSampler()
        result = sampler.fit(calibration)

        assert result.num_groups == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
