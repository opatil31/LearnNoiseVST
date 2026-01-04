"""
Complete noise sampler combining all noise model components.

This module provides the main interface for:
1. Fitting the full noise model from calibration data
2. Sampling noise given predicted clean signals
3. Generating noise-augmented data for self-supervised learning

The noise model consists of:
- Location-scale model: r = μ(ẑ) + σ(ẑ) * u
- Marginal distributions: per-group empirical CDF of standardized residuals u
- Copula (optional): Gaussian copula for modeling dependence between groups

Given a predicted clean signal ẑ, we can sample noise ε such that z = ẑ + ε
matches the distribution of true transformed noisy data.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
import warnings

from .calibration import CalibrationDatasetGenerator, CalibrationResult
from .location_scale import LocationScaleModel, LocationScaleModelCollection
from .marginal_fit import EmpiricalMarginal, MarginalCollection, GaussianMarginal
from .copula import GaussianCopula, IndependenceCopula, choose_copula


@dataclass
class NoiseModelConfig:
    """Configuration for noise model fitting."""

    # Location-scale
    num_knots: int = 8
    variance_flatness_threshold: float = 0.2
    use_robust: bool = True

    # Marginal
    num_quantiles: int = 1023
    use_gaussian_marginals: bool = False

    # Copula
    fit_copula: bool = True
    copula_shrinkage: float = 0.1
    independence_threshold: float = 0.1

    # Calibration
    margin: int = 8  # For images


@dataclass
class NoiseModelFitResult:
    """Result of noise model fitting."""
    num_groups: int
    num_samples: int
    is_image: bool
    variance_flat_groups: List[int]
    copula_used: bool
    diagnostics: Dict


class NoiseModelSampler:
    """
    Complete noise sampler combining all components.

    Given ẑ, sample ε such that z = ẑ + ε matches true noise.

    The sampling procedure:
    1. For each group, compute μ(ẑ) and σ(ẑ) from location-scale model
    2. Sample standardized noise u from marginal (with optional copula for dependence)
    3. Compute ε = μ(ẑ) + σ(ẑ) * u

    Args:
        config: NoiseModelConfig with fitting parameters.
    """

    def __init__(self, config: Optional[NoiseModelConfig] = None):
        self.config = config or NoiseModelConfig()

        # Components
        self.location_scale: Optional[LocationScaleModelCollection] = None
        self.marginals: Optional[MarginalCollection] = None
        self.copula: Optional[Union[GaussianCopula, IndependenceCopula]] = None

        # Metadata
        self.is_image: bool = False
        self.num_groups: int = 0
        self.group_ids: List[int] = []
        self.is_fitted: bool = False

    def fit(
        self,
        calibration_data: CalibrationResult,
    ) -> NoiseModelFitResult:
        """
        Fit from calibration residuals.

        Args:
            calibration_data: CalibrationResult from CalibrationDatasetGenerator.

        Returns:
            NoiseModelFitResult with fitting diagnostics.
        """
        self.is_image = calibration_data.is_image
        self.num_groups = calibration_data.num_groups
        self.group_ids = calibration_data.get_group_ids()

        # 1. Fit location-scale models
        self.location_scale = LocationScaleModelCollection(
            num_knots=self.config.num_knots,
            variance_flatness_threshold=self.config.variance_flatness_threshold,
            use_robust=self.config.use_robust,
        )

        ls_data = {
            g: {'z_hat': data.z_hat, 'r': data.r}
            for g, data in calibration_data
        }
        ls_results = self.location_scale.fit(ls_data)

        # Track which groups have flat variance
        variance_flat_groups = [
            g for g, result in ls_results.items() if result.is_variance_flat
        ]

        # 2. Compute standardized residuals
        standardized = {}
        for g, data in calibration_data:
            u = self.location_scale.standardize(g, data.z_hat, data.r)
            standardized[g] = u

        # 3. Fit marginal distributions
        self.marginals = MarginalCollection(
            use_gaussian=self.config.use_gaussian_marginals,
            num_quantiles=self.config.num_quantiles,
        )
        marginal_results = self.marginals.fit(standardized)

        # 4. Fit copula for dependence (if requested and multiple groups)
        copula_used = False
        if self.config.fit_copula and self.num_groups > 1:
            # Build matrix of standardized residuals
            # Need to align samples across groups
            min_samples = min(len(u) for u in standardized.values())
            u_matrix = np.column_stack([
                standardized[g][:min_samples] for g in self.group_ids
            ])

            self.copula = choose_copula(
                u_matrix,
                self.marginals,
                independence_threshold=self.config.independence_threshold,
                shrinkage=self.config.copula_shrinkage,
            )
            copula_used = isinstance(self.copula, GaussianCopula)
        else:
            self.copula = IndependenceCopula()
            self.copula.fit(
                np.zeros((1, self.num_groups)),
                group_ids=self.group_ids,
            )

        self.is_fitted = True

        # Compile diagnostics
        diagnostics = {
            'location_scale': {g: r.diagnostics for g, r in ls_results.items()},
            'marginals': {
                g: {
                    'skewness': r.skewness,
                    'kurtosis': r.kurtosis,
                    'is_gaussian_like': r.is_gaussian_like,
                }
                for g, r in marginal_results.items()
            },
        }

        return NoiseModelFitResult(
            num_groups=self.num_groups,
            num_samples=calibration_data.total_samples,
            is_image=self.is_image,
            variance_flat_groups=variance_flat_groups,
            copula_used=copula_used,
            diagnostics=diagnostics,
        )

    def sample(
        self,
        z_hat: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample noise given predicted clean signal.

        Args:
            z_hat: [B, d] or [B, C, H, W] predicted clean signal.

        Returns:
            ε with same shape, such that z = ẑ + ε.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle tensor input
        is_tensor = isinstance(z_hat, torch.Tensor)
        if is_tensor:
            device = z_hat.device
            z_hat_np = z_hat.detach().cpu().numpy()
        else:
            z_hat_np = np.asarray(z_hat)

        # Sample
        if self.is_image:
            eps_np = self._sample_image(z_hat_np)
        else:
            eps_np = self._sample_tabular(z_hat_np)

        # Convert back to tensor if needed
        if is_tensor:
            eps = torch.from_numpy(eps_np).to(device)
            return eps
        else:
            return eps_np

    def _sample_tabular(self, z_hat: np.ndarray) -> np.ndarray:
        """Sample noise for tabular data [B, d]."""
        B, d = z_hat.shape
        eps = np.zeros_like(z_hat)

        if self.copula is not None and isinstance(self.copula, GaussianCopula):
            # Sample correlated standardized noise
            u = self.copula.sample(B, self.marginals)
        else:
            # Sample independently
            u = np.zeros((B, d))
            for j, g in enumerate(self.group_ids):
                u[:, j] = self.marginals.sample(g, B)

        # Apply location-scale transform for each group
        for j, g in enumerate(self.group_ids):
            z_hat_g = z_hat[:, g]
            u_g = u[:, j]
            eps[:, g] = self.location_scale.destandardize(g, z_hat_g, u_g)

        return eps

    def _sample_image(self, z_hat: np.ndarray) -> np.ndarray:
        """Sample noise for image data [B, C, H, W]."""
        B, C, H, W = z_hat.shape
        eps = np.zeros_like(z_hat)

        for c in range(C):
            z_hat_c = z_hat[:, c]  # [B, H, W]

            # Sample standardized noise for this channel
            u_c = self.marginals.sample(c, (B, H, W))

            # Apply location-scale (vectorized)
            z_hat_flat = z_hat_c.flatten()
            u_flat = u_c.flatten()

            eps_flat = self.location_scale.destandardize(c, z_hat_flat, u_flat)
            eps[:, c] = eps_flat.reshape(B, H, W)

        return eps

    def sample_standardized(
        self,
        shape: Union[int, Tuple[int, ...]],
        group_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample standardized noise u (mean=0, var=1).

        Useful for understanding the noise distribution without location-scale.

        Args:
            shape: Output shape.
            group_id: If specified, sample from that group's marginal.
                If None, samples from first group.

        Returns:
            Standardized noise samples.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if group_id is None:
            group_id = self.group_ids[0]

        return self.marginals.sample(group_id, shape)

    def get_noise_std(
        self,
        z_hat: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get noise standard deviation at given z_hat values.

        Args:
            z_hat: Predicted clean signal.

        Returns:
            σ(ẑ) with same shape.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        is_tensor = isinstance(z_hat, torch.Tensor)
        if is_tensor:
            device = z_hat.device
            z_hat_np = z_hat.detach().cpu().numpy()
        else:
            z_hat_np = np.asarray(z_hat)

        if self.is_image:
            B, C, H, W = z_hat_np.shape
            sigma = np.zeros_like(z_hat_np)
            for c in range(C):
                z_hat_flat = z_hat_np[:, c].flatten()
                sigma_flat = self.location_scale.get_sigma(c, z_hat_flat)
                sigma[:, c] = sigma_flat.reshape(B, H, W)
        else:
            B, d = z_hat_np.shape
            sigma = np.zeros_like(z_hat_np)
            for j, g in enumerate(self.group_ids):
                sigma[:, g] = self.location_scale.get_sigma(g, z_hat_np[:, g])

        if is_tensor:
            return torch.from_numpy(sigma).to(device)
        return sigma


class NoiseAugmenter(nn.Module):
    """
    PyTorch module for noise augmentation in training pipelines.

    Wraps NoiseModelSampler for use in data augmentation.

    Args:
        sampler: Fitted NoiseModelSampler.
        noise_scale: Scaling factor for noise (1.0 = original noise level).
    """

    def __init__(
        self,
        sampler: NoiseModelSampler,
        noise_scale: float = 1.0,
    ):
        super().__init__()
        self.sampler = sampler
        self.noise_scale = noise_scale

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Add sampled noise to clean signal.

        Args:
            z_hat: Predicted clean signal.

        Returns:
            z = z_hat + noise_scale * ε
        """
        eps = self.sampler.sample(z_hat)
        return z_hat + self.noise_scale * eps

    def sample_noise(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Sample noise without adding to signal."""
        return self.sampler.sample(z_hat)


def fit_noise_model(
    transform: nn.Module,
    denoiser: nn.Module,
    calibration_loader: DataLoader,
    is_image: bool = False,
    config: Optional[NoiseModelConfig] = None,
    device: str = "cpu",
) -> Tuple[NoiseModelSampler, NoiseModelFitResult]:
    """
    Convenience function to fit complete noise model.

    Args:
        transform: Trained transform module.
        denoiser: Trained denoiser module.
        calibration_loader: DataLoader for calibration data.
        is_image: Whether data is image format.
        config: NoiseModelConfig.
        device: Device for computation.

    Returns:
        (fitted_sampler, fit_result)
    """
    config = config or NoiseModelConfig()

    # Generate calibration residuals
    generator = CalibrationDatasetGenerator(
        transform=transform,
        denoiser=denoiser,
        margin=config.margin,
        device=device,
    )

    calibration = generator.generate(
        calibration_loader,
        is_image=is_image,
    )

    # Fit noise model
    sampler = NoiseModelSampler(config)
    result = sampler.fit(calibration)

    return sampler, result


class NoiseModelIO:
    """
    Save and load noise models.
    """

    @staticmethod
    def save(
        sampler: NoiseModelSampler,
        path: str,
    ):
        """Save noise model to file."""
        import pickle

        state = {
            'config': sampler.config,
            'is_image': sampler.is_image,
            'num_groups': sampler.num_groups,
            'group_ids': sampler.group_ids,
            'location_scale': sampler.location_scale,
            'marginals': sampler.marginals,
            'copula': sampler.copula,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str) -> NoiseModelSampler:
        """Load noise model from file."""
        import pickle

        with open(path, 'rb') as f:
            state = pickle.load(f)

        sampler = NoiseModelSampler(state['config'])
        sampler.is_image = state['is_image']
        sampler.num_groups = state['num_groups']
        sampler.group_ids = state['group_ids']
        sampler.location_scale = state['location_scale']
        sampler.marginals = state['marginals']
        sampler.copula = state['copula']
        sampler.is_fitted = True

        return sampler


def validate_noise_model(
    sampler: NoiseModelSampler,
    calibration_data: CalibrationResult,
    n_samples: int = 1000,
) -> Dict:
    """
    Validate noise model via two-sample tests.

    Compares real residuals vs sampled residuals using:
    - Kolmogorov-Smirnov test per group
    - Moment matching (mean, std, skewness, kurtosis)

    Args:
        sampler: Fitted noise model.
        calibration_data: Original calibration data.
        n_samples: Number of samples for comparison.

    Returns:
        Dict with validation results per group.
    """
    try:
        from scipy.stats import ks_2samp
        has_scipy = True
    except ImportError:
        has_scipy = False
        warnings.warn("scipy not available, using simplified validation")

    results = {}

    for group_id, data in calibration_data:
        z_hat = data.z_hat[:n_samples]
        r_real = data.r[:n_samples]

        # Sample from model
        if sampler.is_image:
            # Reshape for image format
            z_hat_reshaped = z_hat.reshape(-1, 1, 1, 1)
            r_sampled = sampler._sample_image(z_hat_reshaped).flatten()
        else:
            z_hat_reshaped = z_hat.reshape(-1, 1)
            r_sampled = sampler._sample_tabular(z_hat_reshaped).flatten()

        # Moment comparison
        result = {
            'real_mean': np.mean(r_real),
            'sampled_mean': np.mean(r_sampled),
            'real_std': np.std(r_real),
            'sampled_std': np.std(r_sampled),
        }

        # KS test
        if has_scipy:
            ks_stat, p_value = ks_2samp(r_real, r_sampled)
            result['ks_stat'] = ks_stat
            result['p_value'] = p_value
            result['passed'] = p_value > 0.05
        else:
            result['passed'] = abs(result['real_mean'] - result['sampled_mean']) < 0.1

        results[group_id] = result

    return results
