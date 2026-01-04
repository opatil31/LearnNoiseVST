"""
Evaluation metrics for learnable VST experiments.

This module provides metrics for:
1. Variance flatness assessment
2. Oracle transform comparison
3. Residual quality metrics
4. Noise model validation

All metrics can be computed with both numpy arrays and torch tensors.
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple, Callable, Union
from dataclasses import dataclass


@dataclass
class VarianceFlatnessResult:
    """Result of variance flatness evaluation."""
    cv: float  # Coefficient of variation of bin variances
    bin_centers: np.ndarray
    bin_variances: np.ndarray
    is_flat: bool  # CV < threshold
    log_variance_std: float  # Std of log(variance)


@dataclass
class OracleComparisonResult:
    """Result of comparison with oracle transform."""
    correlation: float  # Pearson correlation
    spearman_correlation: float  # Spearman (rank) correlation
    mse: float  # Mean squared error
    mae: float  # Mean absolute error
    relative_error: float  # Mean relative error
    monotonicity_preserved: bool


@dataclass
class ResidualQualityResult:
    """Result of residual quality assessment."""
    mean: float
    std: float
    skewness: float
    kurtosis: float  # Excess kurtosis
    is_gaussian_like: bool
    normality_pvalue: Optional[float]


# ============================================================================
# Variance Flatness Metrics
# ============================================================================

def variance_flatness_score(
    z_hat: Union[np.ndarray, torch.Tensor],
    residuals: Union[np.ndarray, torch.Tensor],
    num_bins: int = 20,
    min_samples_per_bin: int = 10,
) -> VarianceFlatnessResult:
    """
    Compute variance flatness score via binned variance estimation.

    A good VST should produce residuals with constant variance across
    all signal levels. This metric bins predictions by value and
    computes variance in each bin.

    Args:
        z_hat: Predicted clean signal [N] or [N, d].
        residuals: Residuals r = z - ẑ.
        num_bins: Number of bins for signal level.
        min_samples_per_bin: Minimum samples for valid bin.

    Returns:
        VarianceFlatnessResult with CV and per-bin variances.
    """
    # Convert to numpy
    if isinstance(z_hat, torch.Tensor):
        z_hat = z_hat.detach().cpu().numpy()
    if isinstance(residuals, torch.Tensor):
        residuals = residuals.detach().cpu().numpy()

    z_hat_flat = z_hat.flatten()
    r_flat = residuals.flatten()

    # Compute bin edges using percentiles (handles uneven distributions)
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(z_hat_flat, percentiles)

    bin_centers = []
    bin_variances = []

    for i in range(num_bins):
        if i < num_bins - 1:
            mask = (z_hat_flat >= bin_edges[i]) & (z_hat_flat < bin_edges[i + 1])
        else:
            mask = (z_hat_flat >= bin_edges[i]) & (z_hat_flat <= bin_edges[i + 1])

        if mask.sum() >= min_samples_per_bin:
            bin_centers.append(z_hat_flat[mask].mean())
            bin_variances.append(r_flat[mask].var())

    if len(bin_variances) < 2:
        return VarianceFlatnessResult(
            cv=float('nan'),
            bin_centers=np.array([]),
            bin_variances=np.array([]),
            is_flat=False,
            log_variance_std=float('nan'),
        )

    bin_centers = np.array(bin_centers)
    bin_variances = np.array(bin_variances)

    # Coefficient of variation
    cv = bin_variances.std() / (bin_variances.mean() + 1e-8)

    # Log variance std (another measure of flatness)
    log_var_std = np.std(np.log(bin_variances + 1e-8))

    return VarianceFlatnessResult(
        cv=cv,
        bin_centers=bin_centers,
        bin_variances=bin_variances,
        is_flat=cv < 0.3,  # Threshold for "flat"
        log_variance_std=log_var_std,
    )


def variance_flatness_functional(
    z_hat: Union[np.ndarray, torch.Tensor],
    residuals: Union[np.ndarray, torch.Tensor],
    bandwidth: float = 0.5,
    subsample: int = 1000,
) -> float:
    """
    Compute J[T] = Var(log σ²(ẑ)) using kernel smoothing.

    This is the variance flatness functional from VST theory.
    Lower values indicate better variance stabilization.

    Args:
        z_hat: Predicted clean signal.
        residuals: Residuals.
        bandwidth: Kernel bandwidth.
        subsample: Max samples for efficiency.

    Returns:
        J[T] value (lower is better).
    """
    if isinstance(z_hat, torch.Tensor):
        z_hat = z_hat.detach().cpu().numpy()
    if isinstance(residuals, torch.Tensor):
        residuals = residuals.detach().cpu().numpy()

    z_hat_flat = z_hat.flatten()
    r_flat = residuals.flatten()

    n = len(z_hat_flat)
    if n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        z_hat_flat = z_hat_flat[idx]
        r_flat = r_flat[idx]
        n = subsample

    # Kernel weights
    diffs = z_hat_flat[:, None] - z_hat_flat[None, :]
    weights = np.exp(-diffs**2 / (2 * bandwidth**2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

    # Local variance
    r_sq = r_flat ** 2
    local_var = (weights * r_sq[None, :]).sum(axis=1)
    local_var = np.maximum(local_var, 1e-8)

    # Variance of log local variance
    log_var = np.log(local_var)
    J_T = np.var(log_var)

    return J_T


# ============================================================================
# Oracle Comparison Metrics
# ============================================================================

def compare_with_oracle(
    learned_transform: Callable,
    oracle_transform: Callable,
    x_test: Union[np.ndarray, torch.Tensor],
    normalize: bool = True,
) -> OracleComparisonResult:
    """
    Compare learned transform with oracle (ground truth) transform.

    Since transforms are unique only up to affine transformation,
    we optionally normalize both to zero mean and unit variance
    before comparison.

    Args:
        learned_transform: Learned transform function.
        oracle_transform: Ground truth transform function.
        x_test: Test data points.
        normalize: If True, normalize both transforms before comparison.

    Returns:
        OracleComparisonResult with various comparison metrics.
    """
    # Convert to numpy
    if isinstance(x_test, torch.Tensor):
        x_np = x_test.detach().cpu().numpy()
    else:
        x_np = x_test

    # Apply transforms
    if isinstance(x_test, torch.Tensor):
        with torch.no_grad():
            z_learned = learned_transform(x_test)
            if isinstance(z_learned, torch.Tensor):
                z_learned = z_learned.detach().cpu().numpy()
    else:
        z_learned = learned_transform(x_test)
        if isinstance(z_learned, torch.Tensor):
            z_learned = z_learned.detach().cpu().numpy()

    z_oracle = oracle_transform(x_np)

    # Flatten
    z_learned_flat = z_learned.flatten()
    z_oracle_flat = z_oracle.flatten()

    # Normalize (transforms are unique up to affine)
    if normalize:
        z_learned_norm = (z_learned_flat - z_learned_flat.mean()) / (z_learned_flat.std() + 1e-8)
        z_oracle_norm = (z_oracle_flat - z_oracle_flat.mean()) / (z_oracle_flat.std() + 1e-8)

        # Handle potential sign flip
        corr = np.corrcoef(z_learned_norm, z_oracle_norm)[0, 1]
        if corr < 0:
            z_learned_norm = -z_learned_norm

        z_learned_flat = z_learned_norm
        z_oracle_flat = z_oracle_norm

    # Pearson correlation
    correlation = np.corrcoef(z_learned_flat, z_oracle_flat)[0, 1]

    # Spearman (rank) correlation
    try:
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(z_learned_flat, z_oracle_flat)
    except ImportError:
        # Fallback: compute rank correlation manually
        ranks_l = np.argsort(np.argsort(z_learned_flat))
        ranks_o = np.argsort(np.argsort(z_oracle_flat))
        spearman_corr = np.corrcoef(ranks_l, ranks_o)[0, 1]

    # Error metrics
    mse = np.mean((z_learned_flat - z_oracle_flat) ** 2)
    mae = np.mean(np.abs(z_learned_flat - z_oracle_flat))
    relative_error = np.mean(np.abs(z_learned_flat - z_oracle_flat) / (np.abs(z_oracle_flat) + 1e-8))

    # Monotonicity check: does learned preserve ordering?
    x_sorted_idx = np.argsort(x_np.flatten())
    z_learned_sorted = z_learned.flatten()[x_sorted_idx]
    monotonicity_preserved = np.all(np.diff(z_learned_sorted) >= -1e-6)

    return OracleComparisonResult(
        correlation=abs(correlation),  # Take abs because sign can flip
        spearman_correlation=abs(spearman_corr),
        mse=mse,
        mae=mae,
        relative_error=relative_error,
        monotonicity_preserved=monotonicity_preserved,
    )


def variance_ratio_comparison(
    learned_transform: Callable,
    oracle_transform: Callable,
    x_test: np.ndarray,
    mu_test: np.ndarray,
    sigma_test: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compare variance ratio V(z)/V(oracle_z) across signal levels.

    A perfect VST would give ratio = 1 everywhere.

    Args:
        learned_transform: Learned transform.
        oracle_transform: Oracle transform.
        x_test: Test observations.
        mu_test: True clean signals.
        sigma_test: True noise std.
        num_bins: Number of signal level bins.

    Returns:
        Dict with bin_centers, learned_var, oracle_var, ratio.
    """
    # Apply transforms
    if isinstance(x_test, torch.Tensor):
        with torch.no_grad():
            z_learned = learned_transform(x_test)
            if isinstance(z_learned, torch.Tensor):
                z_learned = z_learned.detach().cpu().numpy()
    else:
        z_learned = learned_transform(x_test)
        if isinstance(z_learned, torch.Tensor):
            z_learned = z_learned.detach().cpu().numpy()

    z_oracle = oracle_transform(x_test if isinstance(x_test, np.ndarray) else x_test.numpy())

    # Also transform clean signal for reference
    mu_z_learned = learned_transform(
        torch.from_numpy(mu_test).float() if not isinstance(mu_test, torch.Tensor) else mu_test
    )
    if isinstance(mu_z_learned, torch.Tensor):
        mu_z_learned = mu_z_learned.detach().cpu().numpy()
    mu_z_oracle = oracle_transform(mu_test)

    # Compute residuals in z-space
    r_learned = z_learned - mu_z_learned
    r_oracle = z_oracle - mu_z_oracle

    # Bin by signal level
    mu_flat = mu_test.flatten()
    bin_edges = np.percentile(mu_flat, np.linspace(0, 100, num_bins + 1))

    results = {
        'bin_centers': [],
        'learned_var': [],
        'oracle_var': [],
        'ratio': [],
    }

    for i in range(num_bins):
        if i < num_bins - 1:
            mask = (mu_flat >= bin_edges[i]) & (mu_flat < bin_edges[i + 1])
        else:
            mask = (mu_flat >= bin_edges[i]) & (mu_flat <= bin_edges[i + 1])

        if mask.sum() > 10:
            results['bin_centers'].append(mu_flat[mask].mean())
            var_l = r_learned.flatten()[mask].var()
            var_o = r_oracle.flatten()[mask].var()
            results['learned_var'].append(var_l)
            results['oracle_var'].append(var_o)
            results['ratio'].append(var_l / (var_o + 1e-8))

    for k in results:
        results[k] = np.array(results[k])

    return results


# ============================================================================
# Residual Quality Metrics
# ============================================================================

def assess_residual_quality(
    residuals: Union[np.ndarray, torch.Tensor],
    standardize: bool = True,
) -> ResidualQualityResult:
    """
    Assess quality of residuals (should be Gaussian-like after good VST).

    Args:
        residuals: Residual values.
        standardize: If True, standardize residuals first.

    Returns:
        ResidualQualityResult with distribution metrics.
    """
    if isinstance(residuals, torch.Tensor):
        residuals = residuals.detach().cpu().numpy()

    r = residuals.flatten()

    if standardize:
        r = (r - r.mean()) / (r.std() + 1e-8)

    mean = r.mean()
    std = r.std()

    # Skewness
    skewness = np.mean((r - mean) ** 3) / (std ** 3 + 1e-8)

    # Excess kurtosis
    kurtosis = np.mean((r - mean) ** 4) / (std ** 4 + 1e-8) - 3

    # Normality test
    try:
        from scipy.stats import jarque_bera
        _, pvalue = jarque_bera(r)
    except ImportError:
        pvalue = None

    # Gaussian-like if low skewness and kurtosis
    is_gaussian_like = (abs(skewness) < 0.5) and (abs(kurtosis) < 1.0)

    return ResidualQualityResult(
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        is_gaussian_like=is_gaussian_like,
        normality_pvalue=pvalue,
    )


def noise_sampler_ks_test(
    real_residuals: np.ndarray,
    sampled_residuals: np.ndarray,
) -> Dict[str, float]:
    """
    Kolmogorov-Smirnov two-sample test for noise model validation.

    Args:
        real_residuals: True residuals from data.
        sampled_residuals: Residuals sampled from noise model.

    Returns:
        Dict with ks_statistic, p_value, passed (p > 0.05).
    """
    try:
        from scipy.stats import ks_2samp
        stat, pvalue = ks_2samp(real_residuals.flatten(), sampled_residuals.flatten())
    except ImportError:
        # Simplified version without scipy
        r1 = np.sort(real_residuals.flatten())
        r2 = np.sort(sampled_residuals.flatten())

        # Empirical CDFs
        n1, n2 = len(r1), len(r2)
        all_values = np.sort(np.concatenate([r1, r2]))

        cdf1 = np.searchsorted(r1, all_values, side='right') / n1
        cdf2 = np.searchsorted(r2, all_values, side='right') / n2

        stat = np.max(np.abs(cdf1 - cdf2))
        # Approximate p-value (not accurate but usable)
        n = n1 * n2 / (n1 + n2)
        pvalue = 2 * np.exp(-2 * n * stat ** 2)

    return {
        'ks_statistic': stat,
        'p_value': pvalue,
        'passed': pvalue > 0.05,
    }


# ============================================================================
# Aggregate Metrics
# ============================================================================

def compute_all_metrics(
    z: Union[np.ndarray, torch.Tensor],
    z_hat: Union[np.ndarray, torch.Tensor],
    learned_transform: Optional[Callable] = None,
    oracle_transform: Optional[Callable] = None,
    x_test: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute all relevant metrics in one call.

    Args:
        z: Transformed data.
        z_hat: Denoiser predictions.
        learned_transform: Optional learned transform for oracle comparison.
        oracle_transform: Optional oracle transform.
        x_test: Optional test data for oracle comparison.

    Returns:
        Dict with all metrics.
    """
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    if isinstance(z_hat, torch.Tensor):
        z_hat = z_hat.detach().cpu().numpy()

    residuals = z - z_hat

    results = {}

    # Variance flatness
    vf_result = variance_flatness_score(z_hat, residuals)
    results['variance_flatness'] = {
        'cv': vf_result.cv,
        'is_flat': vf_result.is_flat,
        'log_var_std': vf_result.log_variance_std,
    }

    # J[T] functional
    results['J_T'] = variance_flatness_functional(z_hat, residuals)

    # Residual quality
    rq_result = assess_residual_quality(residuals)
    results['residual_quality'] = {
        'mean': rq_result.mean,
        'std': rq_result.std,
        'skewness': rq_result.skewness,
        'kurtosis': rq_result.kurtosis,
        'is_gaussian_like': rq_result.is_gaussian_like,
    }

    # Oracle comparison (if available)
    if learned_transform is not None and oracle_transform is not None and x_test is not None:
        oracle_result = compare_with_oracle(learned_transform, oracle_transform, x_test)
        results['oracle_comparison'] = {
            'correlation': oracle_result.correlation,
            'spearman': oracle_result.spearman_correlation,
            'mse': oracle_result.mse,
            'monotonicity': oracle_result.monotonicity_preserved,
        }

    return results
