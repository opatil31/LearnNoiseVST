"""
Location-scale model fitting for noise characterization.

After obtaining calibration residuals r = z - ẑ, we fit a location-scale model:

    r = μ(ẑ) + σ(ẑ) * u

where:
- μ(ẑ): location function (ideally ~0 if denoiser is unbiased)
- σ(ẑ): scale function (ideally constant if VST worked perfectly)
- u: standardized residual with mean 0, variance 1

This module fits μ and σ as smooth functions of ẑ using:
- Robust regression for μ (resistant to outliers)
- Local variance estimation for σ
- Optional constancy test to decide if σ should be constant

If the learned VST is perfect, σ(ẑ) ≈ constant. We test for this
and use a simpler constant model when appropriate.
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict
from dataclasses import dataclass
import warnings


@dataclass
class LocationScaleFitResult:
    """Result of location-scale model fitting."""
    is_variance_flat: bool
    sigma_constant: Optional[float]
    mu_function: Callable[[np.ndarray], np.ndarray]
    sigma_function: Callable[[np.ndarray], np.ndarray]
    diagnostics: Dict


class LocationScaleModel:
    """
    Fit r = μ_g(ẑ) + σ_g(ẑ) * u location-scale model.

    μ_g: bias function (ideally ~0)
    σ_g: scale function (ideally constant if VST worked)

    Uses robust spline fitting for both functions.

    Args:
        num_knots: Number of knots for spline fitting.
        variance_flatness_threshold: CV threshold for declaring variance flat.
        min_samples_per_bin: Minimum samples per bin for variance estimation.
        use_robust: Use robust (median-based) estimation.
    """

    def __init__(
        self,
        num_knots: int = 8,
        variance_flatness_threshold: float = 0.2,
        min_samples_per_bin: int = 50,
        use_robust: bool = True,
    ):
        self.num_knots = num_knots
        self.variance_flatness_threshold = variance_flatness_threshold
        self.min_samples_per_bin = min_samples_per_bin
        self.use_robust = use_robust

        # Fitted components
        self.mu_spline = None
        self.log_sigma_spline = None
        self.sigma_const = None
        self.is_variance_flat = False

        # Normalization
        self.z_hat_min = None
        self.z_hat_max = None

    def fit(
        self,
        z_hat: np.ndarray,
        r: np.ndarray,
    ) -> LocationScaleFitResult:
        """
        Fit location and scale functions.

        Args:
            z_hat: Predicted clean signal values.
            r: Residuals (z - z_hat).

        Returns:
            LocationScaleFitResult with fitted functions and diagnostics.
        """
        z_hat = np.asarray(z_hat).flatten()
        r = np.asarray(r).flatten()

        if len(z_hat) != len(r):
            raise ValueError("z_hat and r must have same length")

        # Store range for normalization
        self.z_hat_min = np.min(z_hat)
        self.z_hat_max = np.max(z_hat)

        # 1. Fit μ(ẑ) robustly
        self.mu_spline = self._fit_location(z_hat, r)

        # 2. Compute de-meaned residuals
        mu_pred = self.mu_spline(z_hat)
        r_tilde = r - mu_pred

        # 3. Check if variance is flat
        self.is_variance_flat, cv = self._test_variance_flatness(z_hat, r_tilde)

        # 4. Fit σ(ẑ) or use constant
        if self.is_variance_flat:
            if self.use_robust:
                self.sigma_const = 1.4826 * np.median(np.abs(r_tilde))  # MAD estimator
            else:
                self.sigma_const = np.std(r_tilde)
            self.log_sigma_spline = None
        else:
            self.sigma_const = None
            self.log_sigma_spline = self._fit_scale(z_hat, r_tilde)

        # Diagnostics
        diagnostics = {
            'variance_cv': cv,
            'is_flat': self.is_variance_flat,
            'mu_mean': np.mean(mu_pred),
            'sigma_mean': self.sigma_const if self.is_variance_flat else np.exp(self.log_sigma_spline(z_hat)).mean(),
            'n_samples': len(z_hat),
        }

        return LocationScaleFitResult(
            is_variance_flat=self.is_variance_flat,
            sigma_constant=self.sigma_const,
            mu_function=self.get_mu,
            sigma_function=self.get_sigma,
            diagnostics=diagnostics,
        )

    def _fit_location(
        self,
        z_hat: np.ndarray,
        r: np.ndarray,
    ) -> Callable:
        """Fit location function μ(ẑ) with robust regression."""
        # Use local median for robustness
        sorted_idx = np.argsort(z_hat)
        z_sorted = z_hat[sorted_idx]
        r_sorted = r[sorted_idx]

        n = len(z_hat)
        num_bins = min(self.num_knots * 2, n // self.min_samples_per_bin)
        num_bins = max(num_bins, 3)

        bin_edges = np.percentile(z_sorted, np.linspace(0, 100, num_bins + 1))
        bin_centers = []
        bin_medians = []

        for i in range(num_bins):
            mask = (z_sorted >= bin_edges[i]) & (z_sorted < bin_edges[i + 1])
            if i == num_bins - 1:  # Include last point
                mask = (z_sorted >= bin_edges[i]) & (z_sorted <= bin_edges[i + 1])

            if mask.sum() >= self.min_samples_per_bin // 2:
                if self.use_robust:
                    bin_medians.append(np.median(r_sorted[mask]))
                else:
                    bin_medians.append(np.mean(r_sorted[mask]))
                bin_centers.append(np.median(z_sorted[mask]))

        if len(bin_centers) < 2:
            # Fall back to constant
            const_val = np.median(r) if self.use_robust else np.mean(r)
            return lambda x: np.full_like(x, const_val, dtype=float)

        bin_centers = np.array(bin_centers)
        bin_medians = np.array(bin_medians)

        # Linear interpolation with extrapolation
        return self._create_interpolator(bin_centers, bin_medians)

    def _fit_scale(
        self,
        z_hat: np.ndarray,
        r_tilde: np.ndarray,
    ) -> Callable:
        """Fit scale function σ(ẑ) via log variance."""
        sorted_idx = np.argsort(z_hat)
        z_sorted = z_hat[sorted_idx]
        r_sorted = r_tilde[sorted_idx]

        n = len(z_hat)
        num_bins = min(self.num_knots * 2, n // self.min_samples_per_bin)
        num_bins = max(num_bins, 3)

        bin_edges = np.percentile(z_sorted, np.linspace(0, 100, num_bins + 1))
        bin_centers = []
        bin_log_vars = []

        for i in range(num_bins):
            mask = (z_sorted >= bin_edges[i]) & (z_sorted < bin_edges[i + 1])
            if i == num_bins - 1:
                mask = (z_sorted >= bin_edges[i]) & (z_sorted <= bin_edges[i + 1])

            if mask.sum() >= self.min_samples_per_bin // 2:
                if self.use_robust:
                    # MAD-based variance estimate
                    mad = 1.4826 * np.median(np.abs(r_sorted[mask]))
                    var_est = mad ** 2
                else:
                    var_est = np.var(r_sorted[mask])

                bin_log_vars.append(np.log(var_est + 1e-8))
                bin_centers.append(np.median(z_sorted[mask]))

        if len(bin_centers) < 2:
            # Fall back to constant
            const_log_var = np.log(np.var(r_tilde) + 1e-8)
            return lambda x: np.full_like(x, const_log_var, dtype=float)

        bin_centers = np.array(bin_centers)
        bin_log_vars = np.array(bin_log_vars)

        return self._create_interpolator(bin_centers, bin_log_vars)

    def _create_interpolator(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Callable:
        """Create linear interpolator with constant extrapolation."""
        def interpolate(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new)
            result = np.interp(x_new, x, y)
            # Constant extrapolation at boundaries
            result = np.where(x_new < x[0], y[0], result)
            result = np.where(x_new > x[-1], y[-1], result)
            return result

        return interpolate

    def _test_variance_flatness(
        self,
        z_hat: np.ndarray,
        r_tilde: np.ndarray,
        num_bins: int = 10,
    ) -> Tuple[bool, float]:
        """
        Test if variance is approximately constant across ẑ values.

        Returns (is_flat, coefficient_of_variation).
        """
        sorted_idx = np.argsort(z_hat)
        z_sorted = z_hat[sorted_idx]
        r_sorted = r_tilde[sorted_idx]

        bin_edges = np.percentile(z_sorted, np.linspace(0, 100, num_bins + 1))
        bin_vars = []

        for i in range(num_bins):
            mask = (z_sorted >= bin_edges[i]) & (z_sorted < bin_edges[i + 1])
            if i == num_bins - 1:
                mask = (z_sorted >= bin_edges[i]) & (z_sorted <= bin_edges[i + 1])

            if mask.sum() >= self.min_samples_per_bin:
                if self.use_robust:
                    mad = 1.4826 * np.median(np.abs(r_sorted[mask]))
                    bin_vars.append(mad ** 2)
                else:
                    bin_vars.append(np.var(r_sorted[mask]))

        if len(bin_vars) < 3:
            return True, 0.0

        bin_vars = np.array(bin_vars)
        cv = np.std(bin_vars) / (np.mean(bin_vars) + 1e-8)

        is_flat = cv < self.variance_flatness_threshold
        return is_flat, cv

    def get_mu(self, z_hat: np.ndarray) -> np.ndarray:
        """Get location (mean) estimate at z_hat values."""
        z_hat = np.asarray(z_hat)
        if self.mu_spline is None:
            return np.zeros_like(z_hat)
        return self.mu_spline(z_hat)

    def get_sigma(self, z_hat: np.ndarray) -> np.ndarray:
        """Get scale (std) estimate at z_hat values."""
        z_hat = np.asarray(z_hat)
        if self.sigma_const is not None:
            return np.full_like(z_hat, self.sigma_const, dtype=float)
        elif self.log_sigma_spline is not None:
            return np.exp(0.5 * self.log_sigma_spline(z_hat))
        else:
            return np.ones_like(z_hat)

    def standardize(
        self,
        z_hat: np.ndarray,
        r: np.ndarray,
    ) -> np.ndarray:
        """
        Return standardized residuals u = (r - μ(ẑ)) / σ(ẑ).

        These should have approximately mean 0 and variance 1.
        """
        z_hat = np.asarray(z_hat)
        r = np.asarray(r)

        mu = self.get_mu(z_hat)
        sigma = self.get_sigma(z_hat)

        u = (r - mu) / (sigma + 1e-8)
        return u

    def destandardize(
        self,
        z_hat: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """
        Convert standardized residuals back to original scale.

        r = μ(ẑ) + σ(ẑ) * u
        """
        z_hat = np.asarray(z_hat)
        u = np.asarray(u)

        mu = self.get_mu(z_hat)
        sigma = self.get_sigma(z_hat)

        r = mu + sigma * u
        return r


class DeltaMethodVariancePropagator:
    """
    Propagate variance through the transform using the delta method.

    For a transform z = T(x), the variance in z-space is approximately:

        Var(z) ≈ T'(x)² * Var(x)

    This allows us to relate noise in the original space to noise
    in the transformed space.

    Useful for:
    - Understanding how the transform affects noise
    - Generating noise in original x-space from z-space model
    """

    def __init__(
        self,
        transform_derivative: Callable[[np.ndarray], np.ndarray],
    ):
        """
        Args:
            transform_derivative: Function T'(x) returning derivatives.
        """
        self.T_prime = transform_derivative

    def variance_to_z_space(
        self,
        x: np.ndarray,
        var_x: np.ndarray,
    ) -> np.ndarray:
        """
        Propagate variance from x-space to z-space.

        Var(z) ≈ T'(x)² * Var(x)
        """
        deriv = self.T_prime(x)
        var_z = (deriv ** 2) * var_x
        return var_z

    def variance_to_x_space(
        self,
        x: np.ndarray,
        var_z: np.ndarray,
    ) -> np.ndarray:
        """
        Propagate variance from z-space to x-space.

        Var(x) ≈ Var(z) / T'(x)²
        """
        deriv = self.T_prime(x)
        var_x = var_z / (deriv ** 2 + 1e-8)
        return var_x

    def std_to_z_space(
        self,
        x: np.ndarray,
        std_x: np.ndarray,
    ) -> np.ndarray:
        """Propagate standard deviation from x-space to z-space."""
        deriv = self.T_prime(x)
        std_z = np.abs(deriv) * std_x
        return std_z

    def std_to_x_space(
        self,
        x: np.ndarray,
        std_z: np.ndarray,
    ) -> np.ndarray:
        """Propagate standard deviation from z-space to x-space."""
        deriv = self.T_prime(x)
        std_x = std_z / (np.abs(deriv) + 1e-8)
        return std_x


class LocationScaleModelCollection:
    """
    Collection of location-scale models for multiple groups.

    Manages fitting and inference for all groups (channels/features).
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arguments passed to each LocationScaleModel.
        """
        self.model_kwargs = kwargs
        self.models: Dict[int, LocationScaleModel] = {}

    def fit(
        self,
        calibration_data: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, LocationScaleFitResult]:
        """
        Fit models for all groups.

        Args:
            calibration_data: Dict {group_id: {'z_hat': array, 'r': array}}

        Returns:
            Dict of fit results per group.
        """
        results = {}

        for group_id, data in calibration_data.items():
            z_hat = data['z_hat']
            r = data['r']

            model = LocationScaleModel(**self.model_kwargs)
            result = model.fit(z_hat, r)

            self.models[group_id] = model
            results[group_id] = result

        return results

    def get_mu(self, group_id: int, z_hat: np.ndarray) -> np.ndarray:
        """Get location estimate for a group."""
        return self.models[group_id].get_mu(z_hat)

    def get_sigma(self, group_id: int, z_hat: np.ndarray) -> np.ndarray:
        """Get scale estimate for a group."""
        return self.models[group_id].get_sigma(z_hat)

    def standardize(
        self,
        group_id: int,
        z_hat: np.ndarray,
        r: np.ndarray,
    ) -> np.ndarray:
        """Standardize residuals for a group."""
        return self.models[group_id].standardize(z_hat, r)

    def destandardize(
        self,
        group_id: int,
        z_hat: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Destandardize residuals for a group."""
        return self.models[group_id].destandardize(z_hat, u)

    def __getitem__(self, group_id: int) -> LocationScaleModel:
        return self.models[group_id]

    def __contains__(self, group_id: int) -> bool:
        return group_id in self.models
