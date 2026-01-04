"""
Marginal distribution fitting for standardized residuals.

After location-scale normalization, standardized residuals u should have:
- Mean ≈ 0
- Variance ≈ 1

If the learned VST is perfect and noise is Gaussian, u ~ N(0,1).
In practice, the marginal may be non-Gaussian (heavier tails, skewed, etc.).

This module fits the empirical CDF of u and provides:
1. CDF evaluation F(u)
2. Inverse CDF (quantile function) F^{-1}(p)
3. Efficient sampling via inverse transform

Uses a quantile table with interpolation for speed.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass


@dataclass
class MarginalFitResult:
    """Result of marginal distribution fitting."""
    mean: float
    std: float
    skewness: float
    kurtosis: float
    is_gaussian_like: bool
    quantiles: np.ndarray
    probabilities: np.ndarray


class EmpiricalMarginal:
    """
    Fit and sample from empirical CDF of standardized residuals.

    Uses quantile table for fast inverse CDF sampling via linear interpolation.
    Handles edge cases with extrapolation using Gaussian tails.

    Args:
        num_quantiles: Number of quantile points to store (higher = more accurate).
        tail_extension: Number of standard deviations for Gaussian tail extension.
        regularize: If True, re-standardize quantiles to ensure mean=0, std=1.
    """

    def __init__(
        self,
        num_quantiles: int = 1023,
        tail_extension: float = 4.0,
        regularize: bool = True,
    ):
        self.K = num_quantiles
        self.tail_extension = tail_extension
        self.regularize = regularize

        # Quantile table: K values corresponding to probabilities (1..K)/(K+1)
        self.quantiles: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None

        # Statistics
        self.mean = 0.0
        self.std = 1.0
        self.skewness = 0.0
        self.kurtosis = 0.0

    def fit(self, u: np.ndarray) -> MarginalFitResult:
        """
        Fit from standardized residual samples.

        Args:
            u: Standardized residuals (should have mean≈0, var≈1).

        Returns:
            MarginalFitResult with statistics and quantiles.
        """
        u = np.asarray(u).flatten()
        n = len(u)

        if n < 10:
            raise ValueError(f"Need at least 10 samples, got {n}")

        # Compute statistics
        self.mean = np.mean(u)
        self.std = np.std(u)
        self.skewness = self._compute_skewness(u)
        self.kurtosis = self._compute_kurtosis(u)

        # Sort for quantile computation
        u_sorted = np.sort(u)

        # Compute quantiles at regular probability points
        self.probabilities = (np.arange(1, self.K + 1)) / (self.K + 1)

        # Find quantile values via interpolation
        indices_float = self.probabilities * (n - 1)
        indices_low = np.floor(indices_float).astype(int)
        indices_high = np.minimum(indices_low + 1, n - 1)
        frac = indices_float - indices_low

        self.quantiles = (1 - frac) * u_sorted[indices_low] + frac * u_sorted[indices_high]

        # Regularize to ensure mean=0, std=1
        if self.regularize:
            q_mean = self.quantiles.mean()
            q_std = self.quantiles.std()
            if q_std > 1e-8:
                self.quantiles = (self.quantiles - q_mean) / q_std

        # Check if approximately Gaussian
        is_gaussian_like = (
            abs(self.skewness) < 0.5 and
            abs(self.kurtosis) < 1.0
        )

        return MarginalFitResult(
            mean=self.mean,
            std=self.std,
            skewness=self.skewness,
            kurtosis=self.kurtosis,
            is_gaussian_like=is_gaussian_like,
            quantiles=self.quantiles.copy(),
            probabilities=self.probabilities.copy(),
        )

    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness."""
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return np.mean(((x - mean) / std) ** 3)

    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return np.mean(((x - mean) / std) ** 4) - 3

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Evaluate CDF at given values.

        F(u) = P(U ≤ u)

        Args:
            u: Values at which to evaluate CDF.

        Returns:
            Probability values in [0, 1].
        """
        u = np.asarray(u)
        original_shape = u.shape
        u_flat = u.flatten()

        if self.quantiles is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use searchsorted for efficient lookup
        indices = np.searchsorted(self.quantiles, u_flat)

        # Convert to probabilities with interpolation
        p = np.zeros_like(u_flat, dtype=float)

        # Values below first quantile
        below_mask = indices == 0
        if below_mask.any():
            # Gaussian tail extrapolation
            z_below = (u_flat[below_mask] - self.quantiles[0]) / max(
                self.quantiles[1] - self.quantiles[0], 1e-8
            )
            p[below_mask] = self.probabilities[0] * np.exp(
                np.minimum(z_below, 0)
            )

        # Values above last quantile
        above_mask = indices >= self.K
        if above_mask.any():
            # Gaussian tail extrapolation
            z_above = (u_flat[above_mask] - self.quantiles[-1]) / max(
                self.quantiles[-1] - self.quantiles[-2], 1e-8
            )
            p[above_mask] = self.probabilities[-1] + (1 - self.probabilities[-1]) * (
                1 - np.exp(-np.maximum(z_above, 0))
            )

        # Values in range - linear interpolation
        in_range = ~below_mask & ~above_mask
        if in_range.any():
            idx = indices[in_range]
            idx_low = np.maximum(idx - 1, 0)

            q_low = self.quantiles[idx_low]
            q_high = self.quantiles[np.minimum(idx, self.K - 1)]
            p_low = self.probabilities[idx_low]
            p_high = self.probabilities[np.minimum(idx, self.K - 1)]

            # Linear interpolation
            dq = q_high - q_low
            dq = np.where(dq < 1e-8, 1e-8, dq)
            frac = (u_flat[in_range] - q_low) / dq
            frac = np.clip(frac, 0, 1)

            p[in_range] = p_low + frac * (p_high - p_low)

        # Clip to valid probability range
        p = np.clip(p, 1e-10, 1 - 1e-10)

        return p.reshape(original_shape)

    def quantile(self, p: np.ndarray) -> np.ndarray:
        """
        Evaluate inverse CDF (quantile function) at given probabilities.

        F^{-1}(p) = inf{u : F(u) ≥ p}

        Args:
            p: Probabilities in (0, 1).

        Returns:
            Quantile values.
        """
        p = np.asarray(p)
        original_shape = p.shape
        p_flat = p.flatten()

        if self.quantiles is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Clip probabilities
        p_flat = np.clip(p_flat, 1e-10, 1 - 1e-10)

        u = np.zeros_like(p_flat, dtype=float)

        # Values in the table range
        in_range = (p_flat >= self.probabilities[0]) & (p_flat <= self.probabilities[-1])

        if in_range.any():
            # Linear interpolation in quantile table
            idx_float = np.interp(
                p_flat[in_range],
                self.probabilities,
                np.arange(self.K)
            )
            idx_low = np.floor(idx_float).astype(int)
            idx_high = np.minimum(idx_low + 1, self.K - 1)
            frac = idx_float - idx_low

            u[in_range] = (
                (1 - frac) * self.quantiles[idx_low] +
                frac * self.quantiles[idx_high]
            )

        # Below table range - Gaussian tail
        below_range = p_flat < self.probabilities[0]
        if below_range.any():
            # Approximate using Gaussian quantile
            from scipy.stats import norm
            try:
                u[below_range] = norm.ppf(p_flat[below_range])
            except ImportError:
                # Fallback: linear extrapolation
                slope = (self.quantiles[1] - self.quantiles[0]) / (
                    self.probabilities[1] - self.probabilities[0]
                )
                u[below_range] = self.quantiles[0] + slope * (
                    p_flat[below_range] - self.probabilities[0]
                )

        # Above table range - Gaussian tail
        above_range = p_flat > self.probabilities[-1]
        if above_range.any():
            from scipy.stats import norm
            try:
                u[above_range] = norm.ppf(p_flat[above_range])
            except ImportError:
                # Fallback: linear extrapolation
                slope = (self.quantiles[-1] - self.quantiles[-2]) / (
                    self.probabilities[-1] - self.probabilities[-2]
                )
                u[above_range] = self.quantiles[-1] + slope * (
                    p_flat[above_range] - self.probabilities[-1]
                )

        return u.reshape(original_shape)

    def sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Sample via inverse CDF.

        Args:
            size: Output shape.

        Returns:
            Random samples from the fitted distribution.
        """
        if isinstance(size, int):
            size = (size,)

        # Sample uniform
        p = np.random.uniform(0, 1, size)

        # Apply inverse CDF
        return self.quantile(p)

    def sample_from_uniform(self, p: np.ndarray) -> np.ndarray:
        """
        Sample given uniform random values (for copula integration).

        Args:
            p: Uniform random values in (0, 1).

        Returns:
            Samples from the fitted distribution.
        """
        return self.quantile(p)

    def log_pdf(self, u: np.ndarray) -> np.ndarray:
        """
        Estimate log PDF via numerical differentiation of CDF.

        Args:
            u: Values at which to evaluate log PDF.

        Returns:
            Log probability density estimates.
        """
        eps = 1e-4
        cdf_plus = self.cdf(u + eps)
        cdf_minus = self.cdf(u - eps)

        pdf = (cdf_plus - cdf_minus) / (2 * eps)
        pdf = np.maximum(pdf, 1e-10)

        return np.log(pdf)


class GaussianMarginal:
    """
    Standard Gaussian marginal (baseline).

    Useful for comparison or when empirical distribution is close to Gaussian.
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, u: np.ndarray) -> MarginalFitResult:
        """Fit Gaussian to data."""
        u = np.asarray(u).flatten()

        self.mean = np.mean(u)
        self.std = np.std(u)

        skewness = np.mean(((u - self.mean) / self.std) ** 3)
        kurtosis = np.mean(((u - self.mean) / self.std) ** 4) - 3

        return MarginalFitResult(
            mean=self.mean,
            std=self.std,
            skewness=skewness,
            kurtosis=kurtosis,
            is_gaussian_like=True,
            quantiles=np.array([]),
            probabilities=np.array([]),
        )

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """Gaussian CDF."""
        try:
            from scipy.stats import norm
            return norm.cdf(u, loc=self.mean, scale=self.std)
        except ImportError:
            # Fallback using error function approximation
            z = (u - self.mean) / (self.std * np.sqrt(2))
            return 0.5 * (1 + np.tanh(z * 1.2))  # Approximation

    def quantile(self, p: np.ndarray) -> np.ndarray:
        """Gaussian quantile function."""
        try:
            from scipy.stats import norm
            return norm.ppf(p, loc=self.mean, scale=self.std)
        except ImportError:
            # Fallback approximation
            p = np.clip(p, 1e-10, 1 - 1e-10)
            z = np.arctanh(2 * p - 1) / 1.2
            return self.mean + self.std * np.sqrt(2) * z

    def sample(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """Sample from Gaussian."""
        if isinstance(size, int):
            size = (size,)
        return np.random.normal(self.mean, self.std, size)

    def sample_from_uniform(self, p: np.ndarray) -> np.ndarray:
        """Convert uniform to Gaussian via quantile."""
        return self.quantile(p)


class MarginalCollection:
    """
    Collection of marginal distributions for multiple groups.
    """

    def __init__(
        self,
        use_gaussian: bool = False,
        **kwargs,
    ):
        """
        Args:
            use_gaussian: If True, use Gaussian marginals instead of empirical.
            **kwargs: Arguments passed to each EmpiricalMarginal.
        """
        self.use_gaussian = use_gaussian
        self.marginal_kwargs = kwargs
        self.marginals: Dict[int, Union[EmpiricalMarginal, GaussianMarginal]] = {}

    def fit(
        self,
        standardized_residuals: Dict[int, np.ndarray],
    ) -> Dict[int, MarginalFitResult]:
        """
        Fit marginals for all groups.

        Args:
            standardized_residuals: Dict {group_id: u_array}

        Returns:
            Dict of fit results per group.
        """
        results = {}

        for group_id, u in standardized_residuals.items():
            if self.use_gaussian:
                marginal = GaussianMarginal()
            else:
                marginal = EmpiricalMarginal(**self.marginal_kwargs)

            result = marginal.fit(u)

            self.marginals[group_id] = marginal
            results[group_id] = result

        return results

    def sample(
        self,
        group_id: int,
        size: Union[int, Tuple[int, ...]],
    ) -> np.ndarray:
        """Sample from a group's marginal."""
        return self.marginals[group_id].sample(size)

    def sample_from_uniform(
        self,
        group_id: int,
        p: np.ndarray,
    ) -> np.ndarray:
        """Convert uniform to marginal samples."""
        return self.marginals[group_id].sample_from_uniform(p)

    def cdf(self, group_id: int, u: np.ndarray) -> np.ndarray:
        """Evaluate CDF for a group."""
        return self.marginals[group_id].cdf(u)

    def quantile(self, group_id: int, p: np.ndarray) -> np.ndarray:
        """Evaluate quantile function for a group."""
        return self.marginals[group_id].quantile(p)

    def __getitem__(self, group_id: int):
        return self.marginals[group_id]

    def __contains__(self, group_id: int) -> bool:
        return group_id in self.marginals

    def get_group_ids(self) -> list:
        return list(self.marginals.keys())
