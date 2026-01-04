"""
Visualization utilities for learnable VST experiments.

This module provides plotting functions for:
1. Transform comparison (learned vs oracle)
2. Variance flatness visualization
3. Training curves
4. Residual diagnostics
5. Noise model validation

All functions return matplotlib figures that can be saved or displayed.
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Tuple, Callable, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Visualization functions will not work.")


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


# ============================================================================
# Transform Comparison Plots
# ============================================================================

def plot_transform_comparison(
    learned_transform: Callable,
    oracle_transform: Callable,
    x_range: Tuple[float, float] = (0.1, 100),
    num_points: int = 500,
    title: str = "Learned vs Oracle Transform",
    figsize: Tuple[int, int] = (12, 4),
) -> 'plt.Figure':
    """
    Plot learned transform vs oracle transform.

    Creates three subplots:
    1. T(x) for both transforms
    2. Normalized comparison (after affine alignment)
    3. T'(x) derivative comparison

    Args:
        learned_transform: Learned transform function.
        oracle_transform: Oracle transform function.
        x_range: Range of x values to plot.
        num_points: Number of points.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    # Generate test points
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Apply transforms
    if callable(learned_transform):
        x_tensor = torch.from_numpy(x).float().unsqueeze(1)
        with torch.no_grad():
            z_learned = learned_transform(x_tensor)
            if isinstance(z_learned, torch.Tensor):
                z_learned = z_learned.detach().cpu().numpy().flatten()
    else:
        z_learned = learned_transform

    z_oracle = oracle_transform(x)

    # Normalize for comparison
    z_learned_norm = (z_learned - z_learned.mean()) / (z_learned.std() + 1e-8)
    z_oracle_norm = (z_oracle - z_oracle.mean()) / (z_oracle.std() + 1e-8)

    # Flip sign if negatively correlated
    if np.corrcoef(z_learned_norm, z_oracle_norm)[0, 1] < 0:
        z_learned_norm = -z_learned_norm

    # Compute derivatives numerically
    dx = x[1] - x[0]
    dz_learned = np.gradient(z_learned, dx)
    dz_oracle = np.gradient(z_oracle, dx)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Raw transforms
    ax1 = axes[0]
    ax1.plot(x, z_oracle, 'b-', linewidth=2, label='Oracle T(x)')
    ax1.plot(x, z_learned, 'r--', linewidth=2, label='Learned T(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('T(x)')
    ax1.set_title('Transform Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Normalized comparison
    ax2 = axes[1]
    ax2.plot(z_oracle_norm, z_learned_norm, 'k.', alpha=0.3, markersize=2)
    ax2.plot([-3, 3], [-3, 3], 'r--', linewidth=1, label='y=x (perfect)')
    ax2.set_xlabel('Oracle T(x) (normalized)')
    ax2.set_ylabel('Learned T(x) (normalized)')
    ax2.set_title('Normalized Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Derivatives
    ax3 = axes[2]
    ax3.semilogy(x, dz_oracle, 'b-', linewidth=2, label="Oracle T'(x)")
    ax3.semilogy(x, np.abs(dz_learned), 'r--', linewidth=2, label="|Learned T'(x)|")
    ax3.set_xlabel('x')
    ax3.set_ylabel("T'(x)")
    ax3.set_title('Transform Derivatives')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_per_feature_transforms(
    learned_transform: Callable,
    oracle_transform: Callable,
    x_range: Tuple[float, float],
    num_features: int,
    feature_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> 'plt.Figure':
    """
    Plot transforms for each feature separately.

    Args:
        learned_transform: Learned transform.
        oracle_transform: Oracle transform (per-feature).
        x_range: Range for x values.
        num_features: Number of features.
        feature_names: Optional names for features.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    if figsize is None:
        ncols = min(4, num_features)
        nrows = (num_features + ncols - 1) // ncols
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        nrows=(num_features + 3) // 4,
        ncols=min(4, num_features),
        figsize=figsize,
    )
    axes = np.atleast_2d(axes).flatten()

    x = np.linspace(x_range[0], x_range[1], 200)

    for f in range(num_features):
        ax = axes[f]

        # Create input with zeros except for feature f
        x_input = np.zeros((len(x), num_features))
        x_input[:, f] = x

        # Apply transforms
        x_tensor = torch.from_numpy(x_input).float()
        with torch.no_grad():
            z_learned = learned_transform(x_tensor)
            if isinstance(z_learned, torch.Tensor):
                z_learned = z_learned.detach().cpu().numpy()

        z_oracle = oracle_transform(x_input)

        # Plot
        ax.plot(x, z_oracle[:, f], 'b-', linewidth=2, label='Oracle')
        ax.plot(x, z_learned[:, f], 'r--', linewidth=2, label='Learned')

        name = feature_names[f] if feature_names else f"Feature {f}"
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('T(x)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


# ============================================================================
# Variance Flatness Plots
# ============================================================================

def plot_variance_flatness(
    z_hat: np.ndarray,
    residuals: np.ndarray,
    num_bins: int = 20,
    title: str = "Variance Flatness Analysis",
    figsize: Tuple[int, int] = (12, 4),
) -> 'plt.Figure':
    """
    Visualize variance flatness of residuals.

    Creates three subplots:
    1. Scatter of residuals vs predictions
    2. Binned variance plot
    3. Log-variance vs signal level

    Args:
        z_hat: Predicted clean signal.
        residuals: Residuals r = z - ẑ.
        num_bins: Number of bins.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    z_hat_flat = z_hat.flatten()
    r_flat = residuals.flatten()

    # Compute binned statistics
    bin_edges = np.percentile(z_hat_flat, np.linspace(0, 100, num_bins + 1))
    bin_centers = []
    bin_vars = []
    bin_stds = []

    for i in range(num_bins):
        if i < num_bins - 1:
            mask = (z_hat_flat >= bin_edges[i]) & (z_hat_flat < bin_edges[i + 1])
        else:
            mask = (z_hat_flat >= bin_edges[i]) & (z_hat_flat <= bin_edges[i + 1])

        if mask.sum() > 10:
            bin_centers.append(z_hat_flat[mask].mean())
            bin_vars.append(r_flat[mask].var())
            bin_stds.append(r_flat[mask].std())

    bin_centers = np.array(bin_centers)
    bin_vars = np.array(bin_vars)
    bin_stds = np.array(bin_stds)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Scatter
    ax1 = axes[0]
    subsample = min(5000, len(z_hat_flat))
    idx = np.random.choice(len(z_hat_flat), subsample, replace=False)
    ax1.scatter(z_hat_flat[idx], r_flat[idx], alpha=0.1, s=1)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_xlabel('Predicted ẑ')
    ax1.set_ylabel('Residual r')
    ax1.set_title('Residuals vs Predictions')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Binned variance
    ax2 = axes[1]
    ax2.bar(range(len(bin_vars)), bin_vars, alpha=0.7, color='steelblue')
    ax2.axhline(y=bin_vars.mean(), color='r', linestyle='--',
                linewidth=2, label=f'Mean = {bin_vars.mean():.3f}')
    ax2.set_xlabel('Bin index')
    ax2.set_ylabel('Variance')
    ax2.set_title(f'Binned Variance (CV = {bin_vars.std()/bin_vars.mean():.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Log variance vs signal
    ax3 = axes[2]
    ax3.plot(bin_centers, np.log(bin_vars), 'bo-', linewidth=2, markersize=6)
    ax3.axhline(y=np.log(bin_vars).mean(), color='r', linestyle='--',
                linewidth=2, label='Mean log(var)')
    ax3.set_xlabel('Signal level ẑ')
    ax3.set_ylabel('log(Variance)')
    ax3.set_title('Log-Variance vs Signal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_before_after_vst(
    x: np.ndarray,
    z: np.ndarray,
    mu: np.ndarray,
    title: str = "Before vs After VST",
    figsize: Tuple[int, int] = (12, 4),
) -> 'plt.Figure':
    """
    Compare variance structure before and after applying VST.

    Args:
        x: Original noisy data.
        z: Transformed data (after VST).
        mu: True clean signal (for x-axis).
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Before VST
    ax1 = axes[0]
    residuals_before = x - mu
    mu_flat = mu.flatten()
    r_before_flat = residuals_before.flatten()

    # Bin and compute variance
    num_bins = 15
    bin_edges = np.percentile(mu_flat, np.linspace(0, 100, num_bins + 1))
    bin_centers_before = []
    bin_vars_before = []

    for i in range(num_bins):
        mask = (mu_flat >= bin_edges[i]) & (mu_flat < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_centers_before.append(mu_flat[mask].mean())
            bin_vars_before.append(r_before_flat[mask].var())

    ax1.plot(bin_centers_before, bin_vars_before, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Signal level μ')
    ax1.set_ylabel('Variance')
    ax1.set_title('BEFORE VST: Heteroscedastic')
    ax1.grid(True, alpha=0.3)

    # After VST
    ax2 = axes[1]
    z_mu = z.mean()  # Approximate clean z
    residuals_after = z - z_mu
    z_flat = z.flatten()
    r_after_flat = residuals_after.flatten()

    bin_edges = np.percentile(z_flat, np.linspace(0, 100, num_bins + 1))
    bin_centers_after = []
    bin_vars_after = []

    for i in range(num_bins):
        mask = (z_flat >= bin_edges[i]) & (z_flat < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_centers_after.append(z_flat[mask].mean())
            bin_vars_after.append(r_after_flat[mask].var())

    ax2.plot(bin_centers_after, bin_vars_after, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=np.mean(bin_vars_after), color='r', linestyle='--',
                label=f'Mean = {np.mean(bin_vars_after):.3f}')
    ax2.set_xlabel('Transformed signal z')
    ax2.set_ylabel('Variance')
    ax2.set_title('AFTER VST: Homoscedastic')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


# ============================================================================
# Training Curve Plots
# ============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5),
) -> 'plt.Figure':
    """
    Plot training curves from history dict.

    Args:
        history: Dict with loss names as keys and lists of values.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    # Separate transform and denoiser losses
    transform_keys = [k for k in history if 'transform' in k.lower() or 'homo' in k or 'vf' in k]
    denoiser_keys = [k for k in history if 'denoiser' in k.lower() or 'mse' in k.lower()]
    other_keys = [k for k in history if k not in transform_keys and k not in denoiser_keys]

    num_plots = sum([
        len(transform_keys) > 0,
        len(denoiser_keys) > 0,
        len(other_keys) > 0,
    ])
    num_plots = max(num_plots, 1)

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Transform losses
    if transform_keys:
        ax = axes[plot_idx]
        for key in transform_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Transform Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Denoiser losses
    if denoiser_keys:
        ax = axes[plot_idx]
        for key in denoiser_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Denoiser Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Other metrics
    if other_keys:
        ax = axes[plot_idx]
        for key in other_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title('Other Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_convergence_diagnostics(
    history: Dict[str, List[float]],
    window_size: int = 20,
    figsize: Tuple[int, int] = (14, 8),
) -> 'plt.Figure':
    """
    Plot detailed convergence diagnostics.

    Args:
        history: Training history dict.
        window_size: Window for smoothing.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    def smooth(x, w):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    # Total loss
    ax1 = fig.add_subplot(gs[0, 0])
    if 'transform_loss' in history:
        total = np.array(history['transform_loss']) + np.array(history.get('denoiser_loss', [0]*len(history['transform_loss'])))
        ax1.plot(total, alpha=0.3, color='blue')
        ax1.plot(smooth(total, window_size), color='blue', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss (smoothed)')
    ax1.grid(True, alpha=0.3)

    # J[T] if available
    ax2 = fig.add_subplot(gs[0, 1])
    if 'vf_loss' in history:
        ax2.plot(history['vf_loss'], alpha=0.3, color='green')
        ax2.plot(smooth(history['vf_loss'], window_size), color='green', linewidth=2)
        ax2.set_ylabel('J[T]')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Variance Flatness J[T]')
    ax2.grid(True, alpha=0.3)

    # Homo loss
    ax3 = fig.add_subplot(gs[0, 2])
    if 'homo_loss' in history:
        ax3.plot(history['homo_loss'], alpha=0.3, color='orange')
        ax3.plot(smooth(history['homo_loss'], window_size), color='orange', linewidth=2)
        ax3.set_ylabel('L_homo')
    ax3.set_xlabel('Iteration')
    ax3.set_title('Homoscedasticity Loss')
    ax3.grid(True, alpha=0.3)

    # MSE loss
    ax4 = fig.add_subplot(gs[1, 0])
    if 'mse_loss' in history:
        ax4.plot(history['mse_loss'], alpha=0.3, color='red')
        ax4.plot(smooth(history['mse_loss'], window_size), color='red', linewidth=2)
        ax4.set_ylabel('MSE')
    ax4.set_xlabel('Iteration')
    ax4.set_title('Denoiser MSE')
    ax4.grid(True, alpha=0.3)

    # Loss ratio
    ax5 = fig.add_subplot(gs[1, 1])
    if 'transform_loss' in history and 'denoiser_loss' in history:
        ratio = np.array(history['transform_loss']) / (np.array(history['denoiser_loss']) + 1e-8)
        ax5.semilogy(ratio, alpha=0.3, color='purple')
        ax5.semilogy(smooth(ratio, window_size), color='purple', linewidth=2)
        ax5.set_ylabel('Ratio')
    ax5.set_xlabel('Iteration')
    ax5.set_title('Transform/Denoiser Loss Ratio')
    ax5.grid(True, alpha=0.3)

    # Learning progress
    ax6 = fig.add_subplot(gs[1, 2])
    if 'transform_loss' in history:
        progress = 1 - np.array(history['transform_loss']) / (history['transform_loss'][0] + 1e-8)
        ax6.plot(progress * 100, color='teal', linewidth=2)
        ax6.set_ylabel('Progress (%)')
        ax6.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90%')
    ax6.set_xlabel('Iteration')
    ax6.set_title('Learning Progress')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Residual Diagnostic Plots
# ============================================================================

def plot_residual_diagnostics(
    residuals: np.ndarray,
    title: str = "Residual Diagnostics",
    figsize: Tuple[int, int] = (12, 8),
) -> 'plt.Figure':
    """
    Comprehensive residual diagnostic plots.

    Args:
        residuals: Residual values.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    r = residuals.flatten()
    r_std = (r - r.mean()) / (r.std() + 1e-8)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r_std, bins=50, density=True, alpha=0.7, color='steelblue')
    # Overlay Gaussian
    x_gauss = np.linspace(-4, 4, 100)
    ax1.plot(x_gauss, np.exp(-x_gauss**2/2) / np.sqrt(2*np.pi),
             'r-', linewidth=2, label='N(0,1)')
    ax1.set_xlabel('Standardized residual')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_r = np.sort(r_std)
    n = len(sorted_r)
    theoretical = np.linspace(0.001, 0.999, n)
    try:
        from scipy.stats import norm
        expected = norm.ppf(theoretical)
    except ImportError:
        expected = np.arctanh(2 * theoretical - 1) / 0.85  # Approximation

    ax2.scatter(expected, sorted_r, alpha=0.3, s=1)
    ax2.plot([-4, 4], [-4, 4], 'r--', linewidth=2)
    ax2.set_xlabel('Theoretical quantiles')
    ax2.set_ylabel('Sample quantiles')
    ax2.set_title('Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)

    # Box plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.boxplot(r_std, vert=True)
    ax3.set_ylabel('Standardized residual')
    ax3.set_title('Box Plot')
    ax3.grid(True, alpha=0.3)

    # Autocorrelation (if ordered)
    ax4 = fig.add_subplot(gs[1, 0])
    max_lag = min(50, len(r) // 10)
    acf = [np.corrcoef(r[:-lag], r[lag:])[0, 1] for lag in range(1, max_lag)]
    ax4.bar(range(1, max_lag), acf, color='steelblue', alpha=0.7)
    ax4.axhline(y=0, color='k', linewidth=1)
    ax4.axhline(y=1.96/np.sqrt(len(r)), color='r', linestyle='--', alpha=0.5)
    ax4.axhline(y=-1.96/np.sqrt(len(r)), color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Autocorrelation')
    ax4.set_title('Autocorrelation')
    ax4.grid(True, alpha=0.3)

    # Statistics text
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    skew = np.mean(r_std ** 3)
    kurt = np.mean(r_std ** 4) - 3

    stats_text = f"""
    Sample Statistics:

    N = {len(r):,}
    Mean = {r.mean():.4f}
    Std = {r.std():.4f}

    Skewness = {skew:.4f}  (Gaussian = 0)
    Excess Kurtosis = {kurt:.4f}  (Gaussian = 0)

    Min = {r.min():.4f}
    Max = {r.max():.4f}

    Assessment: {"Gaussian-like ✓" if abs(skew) < 0.5 and abs(kurt) < 1.0 else "Non-Gaussian ✗"}
    """

    ax5.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace',
             verticalalignment='center', transform=ax5.transAxes)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


# ============================================================================
# Ablation Study Plots
# ============================================================================

def plot_ablation_results(
    results: Dict[str, Dict[str, float]],
    metric: str = 'correlation',
    title: str = "Ablation Study Results",
    figsize: Tuple[int, int] = (10, 6),
) -> 'plt.Figure':
    """
    Plot ablation study results as a bar chart.

    Args:
        results: Dict mapping experiment name to metrics dict.
        metric: Which metric to plot.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(names)), values, color='steelblue', alpha=0.8)

    # Highlight best
    best_idx = np.argmax(values)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['correlation', 'cv', 'mse'],
    figsize: Tuple[int, int] = (14, 5),
) -> 'plt.Figure':
    """
    Plot multiple metrics for ablation comparison.

    Args:
        results: Dict mapping experiment name to metrics dict.
        metrics: List of metrics to compare.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.8

    for ax, metric in zip(axes, metrics):
        values = [results[n].get(metric, 0) for n in names]

        colors = ['steelblue'] * len(values)
        best_idx = np.argmax(values) if metric in ['correlation', 'spearman'] else np.argmin(values)
        colors[best_idx] = 'green'

        ax.bar(x, values, width, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


# ============================================================================
# Summary Dashboard
# ============================================================================

def create_experiment_dashboard(
    dataset_name: str,
    history: Dict[str, List[float]],
    z_hat: np.ndarray,
    residuals: np.ndarray,
    oracle_comparison: Optional[Dict] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> 'plt.Figure':
    """
    Create a comprehensive dashboard for a single experiment.

    Args:
        dataset_name: Name of the dataset.
        history: Training history.
        z_hat: Predictions.
        residuals: Residuals.
        oracle_comparison: Optional oracle comparison results.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # Training curve
    ax1 = fig.add_subplot(gs[0, :2])
    if 'transform_loss' in history:
        ax1.plot(history['transform_loss'], label='Transform', alpha=0.7)
    if 'denoiser_loss' in history:
        ax1.plot(history['denoiser_loss'], label='Denoiser', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Variance flatness
    ax2 = fig.add_subplot(gs[0, 2:])
    z_flat = z_hat.flatten()
    r_flat = residuals.flatten()
    idx = np.random.choice(len(z_flat), min(3000, len(z_flat)), replace=False)
    ax2.scatter(z_flat[idx], r_flat[idx], alpha=0.1, s=1)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted ẑ')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residuals vs Predictions')
    ax2.grid(True, alpha=0.3)

    # Residual histogram
    ax3 = fig.add_subplot(gs[1, 0])
    r_std = (r_flat - r_flat.mean()) / (r_flat.std() + 1e-8)
    ax3.hist(r_std, bins=50, density=True, alpha=0.7)
    x_g = np.linspace(-4, 4, 100)
    ax3.plot(x_g, np.exp(-x_g**2/2) / np.sqrt(2*np.pi), 'r-', linewidth=2)
    ax3.set_xlabel('Standardized residual')
    ax3.set_title('Residual Distribution')
    ax3.grid(True, alpha=0.3)

    # Binned variance
    ax4 = fig.add_subplot(gs[1, 1])
    num_bins = 15
    bin_edges = np.percentile(z_flat, np.linspace(0, 100, num_bins + 1))
    bin_vars = []
    for i in range(num_bins):
        mask = (z_flat >= bin_edges[i]) & (z_flat < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_vars.append(r_flat[mask].var())
    ax4.bar(range(len(bin_vars)), bin_vars, alpha=0.7)
    cv = np.std(bin_vars) / (np.mean(bin_vars) + 1e-8)
    ax4.axhline(y=np.mean(bin_vars), color='r', linestyle='--')
    ax4.set_xlabel('Bin')
    ax4.set_ylabel('Variance')
    ax4.set_title(f'Binned Variance (CV={cv:.3f})')
    ax4.grid(True, alpha=0.3)

    # Metrics summary
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis('off')

    skew = np.mean(r_std ** 3)
    kurt = np.mean(r_std ** 4) - 3

    metrics_text = f"""
    Dataset: {dataset_name}

    Variance Flatness:
      CV = {cv:.4f}

    Residual Quality:
      Skewness = {skew:.4f}
      Kurtosis = {kurt:.4f}
    """

    if oracle_comparison:
        metrics_text += f"""
    Oracle Comparison:
      Correlation = {oracle_comparison.get('correlation', 'N/A'):.4f}
      Spearman = {oracle_comparison.get('spearman', 'N/A'):.4f}
    """

    ax5.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center', transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Component losses
    ax6 = fig.add_subplot(gs[2, :2])
    component_keys = ['homo_loss', 'vf_loss', 'mse_loss']
    for key in component_keys:
        if key in history:
            ax6.plot(history[key], label=key, alpha=0.7)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Loss')
    ax6.set_title('Component Losses')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Q-Q plot
    ax7 = fig.add_subplot(gs[2, 2])
    sorted_r = np.sort(r_std)
    n = len(sorted_r)
    theoretical = np.linspace(0.001, 0.999, min(n, 1000))
    expected = np.arctanh(2 * theoretical - 1) / 0.85
    sample_q = np.percentile(r_std, theoretical * 100)
    ax7.scatter(expected, sample_q, alpha=0.3, s=5)
    ax7.plot([-3, 3], [-3, 3], 'r--', linewidth=2)
    ax7.set_xlabel('Theoretical')
    ax7.set_ylabel('Sample')
    ax7.set_title('Q-Q Plot')
    ax7.grid(True, alpha=0.3)

    # Box plot by bin
    ax8 = fig.add_subplot(gs[2, 3])
    bin_data = []
    for i in range(min(5, num_bins)):
        mask = (z_flat >= bin_edges[i]) & (z_flat < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_data.append(r_flat[mask])
    if bin_data:
        ax8.boxplot(bin_data)
        ax8.set_xlabel('Bin')
        ax8.set_ylabel('Residual')
        ax8.set_title('Residuals by Bin')
        ax8.grid(True, alpha=0.3)

    fig.suptitle(f'Experiment Dashboard: {dataset_name}', fontsize=16)
    plt.tight_layout()

    return fig
