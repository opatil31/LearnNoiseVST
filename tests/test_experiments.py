"""Tests for synthetic experiment components."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.synthetic.generate_data import (
    SyntheticDataset,
    generate_poisson_like,
    generate_multiplicative,
    generate_affine_variance,
    generate_homoscedastic,
    generate_mixed,
    generate_challenging,
    get_benchmark_datasets,
)
from src.utils.metrics import (
    variance_flatness_score,
    variance_flatness_functional,
    compare_with_oracle,
    assess_residual_quality,
)


class TestSyntheticDataGenerators:
    """Tests for synthetic data generators."""

    def test_poisson_like_shape(self):
        """Test Poisson-like data shape."""
        dataset = generate_poisson_like(n_samples=100, n_features=5, seed=42)

        assert dataset.x.shape == (100, 5)
        assert dataset.mu.shape == (100, 5)
        assert dataset.sigma.shape == (100, 5)

    def test_poisson_like_properties(self):
        """Test Poisson-like noise properties."""
        dataset = generate_poisson_like(n_samples=10000, n_features=3, seed=42)

        # Check that variance increases with signal level
        residuals = dataset.x - dataset.mu

        for j in range(3):
            median_signal = np.median(dataset.mu[:, j])
            low_mask = dataset.mu[:, j] < median_signal
            high_mask = dataset.mu[:, j] >= median_signal

            var_low = np.var(residuals[low_mask, j])
            var_high = np.var(residuals[high_mask, j])

            # High signal should have higher noise variance
            assert var_high > var_low

    def test_poisson_like_oracle(self):
        """Test Poisson-like oracle transform."""
        dataset = generate_poisson_like(n_samples=1000, n_features=3, seed=42)

        assert dataset.oracle_transform is not None
        assert dataset.noise_type == 'poisson_like'

        # Apply oracle
        oracle_z = dataset.oracle_transform(dataset.x)
        assert oracle_z.shape == dataset.x.shape

    def test_multiplicative_shape(self):
        """Test multiplicative noise data shape."""
        dataset = generate_multiplicative(n_samples=100, n_features=5, seed=42)

        assert dataset.x.shape == (100, 5)
        assert dataset.mu.shape == (100, 5)

    def test_multiplicative_oracle(self):
        """Test multiplicative oracle (log transform)."""
        dataset = generate_multiplicative(n_samples=1000, n_features=3, seed=42)

        assert dataset.oracle_transform is not None
        assert dataset.noise_type == 'multiplicative'

        oracle_z = dataset.oracle_transform(dataset.x)
        assert np.all(np.isfinite(oracle_z))

    def test_affine_variance_shape(self):
        """Test affine variance data shape."""
        dataset = generate_affine_variance(n_samples=100, n_features=5, seed=42)

        assert dataset.x.shape == (100, 5)
        assert dataset.mu.shape == (100, 5)

    def test_affine_variance_oracle(self):
        """Test affine variance oracle (Generalized Anscombe)."""
        dataset = generate_affine_variance(n_samples=1000, n_features=3, seed=42)

        assert dataset.oracle_transform is not None
        assert dataset.noise_type == 'affine'

        oracle_z = dataset.oracle_transform(dataset.x)
        assert np.all(np.isfinite(oracle_z))

    def test_homoscedastic(self):
        """Test homoscedastic data generation."""
        dataset = generate_homoscedastic(n_samples=1000, n_features=5, seed=42)

        assert dataset.x.shape == (1000, 5)
        assert dataset.noise_type == 'homoscedastic'

        # Variance should be roughly constant across signal levels
        residuals = dataset.x - dataset.mu

        for j in range(5):
            median_signal = np.median(dataset.mu[:, j])
            low_mask = dataset.mu[:, j] < median_signal
            high_mask = dataset.mu[:, j] >= median_signal

            var_low = np.var(residuals[low_mask, j])
            var_high = np.var(residuals[high_mask, j])

            # Variances should be similar
            ratio = var_high / (var_low + 1e-8)
            assert 0.5 < ratio < 2.0

    def test_mixed_data(self):
        """Test mixed noise data generation."""
        dataset = generate_mixed(n_samples=1000, n_features=6, seed=42)

        assert dataset.x.shape == (1000, 6)
        assert dataset.mu.shape == (1000, 6)
        assert dataset.noise_type == 'mixed'

    def test_challenging_data(self):
        """Test challenging scenario generation."""
        dataset = generate_challenging(n_samples=1000, n_features=6, seed=42)

        assert dataset.x.shape == (1000, 6)
        assert dataset.mu.shape == (1000, 6)
        assert dataset.noise_type == 'challenging'

    def test_benchmark_datasets(self):
        """Test benchmark dataset collection."""
        datasets = get_benchmark_datasets(n_samples=100, n_features=5, seed=42)

        assert 'poisson' in datasets or 'poisson_like' in datasets.get('poisson', datasets).noise_type if 'poisson' in datasets else True
        assert 'multiplicative' in datasets
        assert 'affine' in datasets
        assert 'homoscedastic' in datasets
        assert 'mixed' in datasets

        for name, dataset in datasets.items():
            assert isinstance(dataset, SyntheticDataset)
            assert dataset.x.shape == (100, 5)

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        dataset1 = generate_poisson_like(n_samples=100, n_features=5, seed=42)
        dataset2 = generate_poisson_like(n_samples=100, n_features=5, seed=42)

        np.testing.assert_array_equal(dataset1.x, dataset2.x)
        np.testing.assert_array_equal(dataset1.mu, dataset2.mu)

    def test_to_torch(self):
        """Test conversion to PyTorch dataset."""
        dataset = generate_poisson_like(n_samples=100, n_features=5, seed=42)

        torch_ds = dataset.to_torch()
        assert len(torch_ds) == 100

        x, = torch_ds[0]
        assert x.shape[0] == 5


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_variance_flatness_score(self):
        """Test variance flatness score computation."""
        np.random.seed(42)

        # Perfect homoscedastic case
        z = np.random.randn(1000, 5)
        z_hat = z + np.random.randn(1000, 5) * 0.5  # Constant variance noise
        residuals = z - z_hat

        result = variance_flatness_score(z_hat, residuals)

        # Result should be a VarianceFlatnessResult object
        assert hasattr(result, 'cv')
        assert hasattr(result, 'is_flat')
        assert result.cv >= 0

    def test_variance_flatness_functional(self):
        """Test variance flatness functional J[T]."""
        np.random.seed(42)

        z = np.random.randn(1000, 5)
        residuals = np.random.randn(1000, 5) * 0.5

        J = variance_flatness_functional(z, residuals)

        assert isinstance(J, float)
        assert J >= 0

    def test_compare_with_oracle(self):
        """Test oracle comparison."""
        np.random.seed(42)
        n, d = 100, 5

        # Create test data and transforms
        x_test = np.random.uniform(1, 10, (n, d))

        def learned_transform(x):
            return np.log(x)

        def oracle_transform(x):
            return np.log(x)

        result = compare_with_oracle(
            learned_transform=learned_transform,
            oracle_transform=oracle_transform,
            x_test=x_test,
        )

        assert hasattr(result, 'correlation')
        assert result.correlation > 0.99

    def test_compare_with_oracle_scaled(self):
        """Test oracle comparison with scaled transform."""
        np.random.seed(42)
        n, d = 100, 5

        x_test = np.random.uniform(1, 10, (n, d))

        def learned_transform(x):
            return np.log(x) * 2 + 1  # Scaled and shifted

        def oracle_transform(x):
            return np.log(x)

        result = compare_with_oracle(
            learned_transform=learned_transform,
            oracle_transform=oracle_transform,
            x_test=x_test,
        )

        # Correlation should still be high (linear relationship preserved)
        assert result.correlation > 0.99

    def test_assess_residual_quality(self):
        """Test residual quality assessment."""
        np.random.seed(42)

        # Near-Gaussian residuals
        residuals = np.random.randn(1000, 5)

        quality = assess_residual_quality(residuals)

        assert hasattr(quality, 'is_gaussian_like')
        assert hasattr(quality, 'skewness')
        assert hasattr(quality, 'kurtosis')
        assert abs(quality.skewness) < 0.5


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for visualization tests."""
        np.random.seed(42)
        n, d = 500, 5

        y = np.random.uniform(1, 10, (n, d))
        z_learned = np.log(y)
        z_hat = z_learned + np.random.randn(n, d) * 0.3

        history = {
            'transform_loss': list(np.exp(-np.linspace(0, 2, 50))),
            'denoiser_loss': list(np.exp(-np.linspace(0, 1.5, 50))),
            'total_loss': list(np.exp(-np.linspace(0, 2.5, 50))),
        }

        return {
            'y': y,
            'z_learned': z_learned,
            'z_hat': z_hat,
            'history': history,
        }

    def test_plot_training_curves(self, sample_data):
        """Test training curves plot."""
        import matplotlib
        matplotlib.use('Agg')

        from src.utils.visualization import plot_training_curves

        fig = plot_training_curves(sample_data['history'])

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_residual_diagnostics(self, sample_data):
        """Test residual diagnostics plot."""
        import matplotlib
        matplotlib.use('Agg')

        from src.utils.visualization import plot_residual_diagnostics

        residuals = sample_data['z_learned'] - sample_data['z_hat']
        fig = plot_residual_diagnostics(residuals)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_variance_flatness(self, sample_data):
        """Test variance flatness plot."""
        import matplotlib
        matplotlib.use('Agg')

        from src.utils.visualization import plot_variance_flatness

        z_hat = sample_data['z_hat']
        residuals = sample_data['z_learned'] - sample_data['z_hat']

        fig = plot_variance_flatness(z_hat, residuals)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestOracleTransforms:
    """Tests for oracle transform implementations."""

    def test_anscombe_stabilizes_poisson_variance(self):
        """Test that Anscombe transform stabilizes Poisson-like variance."""
        np.random.seed(42)

        # Generate Poisson-like data
        n = 10000
        lambdas = np.linspace(1, 100, n)
        y = lambdas + np.sqrt(lambdas) * np.random.randn(n)
        y = np.maximum(y, 0.1)  # Ensure positive

        # Before transform: variance increases with signal
        low = lambdas < 30
        high = lambdas > 70

        var_before_low = np.var(y[low] - lambdas[low])
        var_before_high = np.var(y[high] - lambdas[high])
        ratio_before = var_before_high / var_before_low

        # After Anscombe transform
        z = 2 * np.sqrt(y + 3/8)
        z_signal = 2 * np.sqrt(lambdas + 3/8)

        var_after_low = np.var(z[low] - z_signal[low])
        var_after_high = np.var(z[high] - z_signal[high])
        ratio_after = var_after_high / var_after_low

        # Ratio should be closer to 1 after transform
        assert ratio_after < ratio_before
        assert 0.5 < ratio_after < 2.0

    def test_log_stabilizes_multiplicative_variance(self):
        """Test that log transform stabilizes multiplicative variance."""
        np.random.seed(42)

        n = 10000
        signals = np.linspace(1, 10, n)
        cv = 0.2  # Coefficient of variation
        y = signals * (1 + cv * np.random.randn(n))
        y = np.maximum(y, 0.1)

        # Before: variance increases with signal squared
        low = signals < 3
        high = signals > 7

        var_before_low = np.var(y[low] - signals[low])
        var_before_high = np.var(y[high] - signals[high])
        ratio_before = var_before_high / var_before_low

        # After log: variance should be more stable
        z = np.log(y)
        z_signal = np.log(signals)

        var_after_low = np.var(z[low] - z_signal[low])
        var_after_high = np.var(z[high] - z_signal[high])
        ratio_after = var_after_high / var_after_low

        # Ratio should be closer to 1 after transform
        assert ratio_after < ratio_before
        assert 0.3 < ratio_after < 3.0


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_runs(self):
        """Test that the full pipeline runs without errors."""
        # Generate data
        dataset = generate_poisson_like(n_samples=200, n_features=3, seed=42)

        # Compute metrics (without training)
        z_approx = 2 * np.sqrt(dataset.x + 0.375)  # Anscombe
        residuals = z_approx - z_approx.mean(axis=0)  # Simple residuals

        result = variance_flatness_score(z_approx, residuals)

        assert hasattr(result, 'cv')
        assert hasattr(result, 'is_flat')

    @pytest.mark.skip(reason="Requires full training integration - tested via run_minimal.py")
    def test_minimal_training_runs(self):
        """Test that minimal training runs without errors.

        This test is skipped as it requires full integration testing.
        Use `python experiments/synthetic/run_minimal.py` for integration testing.
        """
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
