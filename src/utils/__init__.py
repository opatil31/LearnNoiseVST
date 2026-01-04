"""Utility modules for metrics and visualization."""

from src.utils.metrics import (
    variance_flatness_score,
    variance_flatness_functional,
    compare_with_oracle,
    variance_ratio_comparison,
    assess_residual_quality,
    noise_sampler_ks_test,
    compute_all_metrics,
)

from src.utils.visualization import (
    plot_transform_comparison,
    plot_per_feature_transforms,
    plot_variance_flatness,
    plot_before_after_vst,
    plot_training_curves,
    plot_convergence_diagnostics,
    plot_residual_diagnostics,
    plot_ablation_results,
    plot_ablation_comparison,
    create_experiment_dashboard,
)

__all__ = [
    # Metrics
    'variance_flatness_score',
    'variance_flatness_functional',
    'compare_with_oracle',
    'variance_ratio_comparison',
    'assess_residual_quality',
    'noise_sampler_ks_test',
    'compute_all_metrics',
    # Visualization
    'plot_transform_comparison',
    'plot_per_feature_transforms',
    'plot_variance_flatness',
    'plot_before_after_vst',
    'plot_training_curves',
    'plot_convergence_diagnostics',
    'plot_residual_diagnostics',
    'plot_ablation_results',
    'plot_ablation_comparison',
    'create_experiment_dashboard',
]
