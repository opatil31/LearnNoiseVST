#!/usr/bin/env python
"""
Full synthetic experiment for learnable VST with ablations.

This script runs comprehensive experiments including:
- All synthetic dataset types
- Oracle transform comparisons
- Loss component ablations
- Architecture ablations
- Training strategy ablations
- Complete visualization suite

Usage:
    python run_full.py --output_dir results/full
    python run_full.py --ablations loss architecture --epochs 100
    python run_full.py --skip_ablations --datasets poisson_like multiplicative
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.transform.rqs import RQSTransform
from src.transform.gauge_fixing import GaugeFixingModule
from src.denoiser.blind_spot import ColumnMaskedMLP
from src.training.losses import CombinedTransformLoss, DenoiserLoss
from src.training.alternating_trainer import AlternatingTrainer, TrainerConfig
from src.noise_model import NoiseModelSampler
from src.utils.metrics import (
    compute_all_metrics,
    variance_flatness_score,
    variance_flatness_functional,
    compare_with_oracle,
    variance_ratio_comparison,
    assess_residual_quality,
    noise_sampler_ks_test,
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
from experiments.synthetic.generate_data import (
    generate_poisson_like,
    generate_multiplicative,
    generate_affine_variance,
    generate_homoscedastic,
    generate_mixed,
    generate_challenging,
    get_benchmark_datasets,
    SyntheticDataset,
)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    n_bins: int = 8
    hidden_dims: Tuple[int, ...] = (64, 64)
    lambda_homo: float = 1.0
    lambda_gauge: float = 0.1
    lambda_deriv: float = 0.01
    transform_lr: float = 1e-3
    denoiser_lr: float = 1e-3
    transform_epochs_per_cycle: int = 1
    denoiser_epochs_per_cycle: int = 1
    use_gauge: bool = True
    use_homo_loss: bool = True
    use_deriv_reg: bool = True


@dataclass
class AblationSuite:
    """Collection of ablation configurations."""

    @staticmethod
    def loss_ablations() -> List[ExperimentConfig]:
        """Ablations for loss components."""
        return [
            ExperimentConfig(
                name='full_loss',
                lambda_homo=1.0,
                lambda_gauge=0.1,
                lambda_deriv=0.01,
                use_homo_loss=True,
                use_deriv_reg=True,
            ),
            ExperimentConfig(
                name='no_homo',
                lambda_homo=0.0,
                lambda_gauge=0.1,
                lambda_deriv=0.01,
                use_homo_loss=False,
                use_deriv_reg=True,
            ),
            ExperimentConfig(
                name='no_deriv',
                lambda_homo=1.0,
                lambda_gauge=0.1,
                lambda_deriv=0.0,
                use_homo_loss=True,
                use_deriv_reg=False,
            ),
            ExperimentConfig(
                name='no_gauge',
                lambda_homo=1.0,
                lambda_gauge=0.0,
                lambda_deriv=0.01,
                use_gauge=False,
                use_deriv_reg=True,
            ),
            ExperimentConfig(
                name='homo_only',
                lambda_homo=1.0,
                lambda_gauge=0.0,
                lambda_deriv=0.0,
                use_gauge=False,
                use_homo_loss=True,
                use_deriv_reg=False,
            ),
        ]

    @staticmethod
    def architecture_ablations() -> List[ExperimentConfig]:
        """Ablations for architecture choices."""
        return [
            ExperimentConfig(
                name='bins_4',
                n_bins=4,
            ),
            ExperimentConfig(
                name='bins_8',
                n_bins=8,
            ),
            ExperimentConfig(
                name='bins_16',
                n_bins=16,
            ),
            ExperimentConfig(
                name='hidden_32',
                hidden_dims=(32, 32),
            ),
            ExperimentConfig(
                name='hidden_64',
                hidden_dims=(64, 64),
            ),
            ExperimentConfig(
                name='hidden_128',
                hidden_dims=(128, 128),
            ),
            ExperimentConfig(
                name='hidden_deep',
                hidden_dims=(64, 64, 64),
            ),
        ]

    @staticmethod
    def training_ablations() -> List[ExperimentConfig]:
        """Ablations for training strategies."""
        return [
            ExperimentConfig(
                name='lr_high',
                transform_lr=1e-2,
                denoiser_lr=1e-2,
            ),
            ExperimentConfig(
                name='lr_medium',
                transform_lr=1e-3,
                denoiser_lr=1e-3,
            ),
            ExperimentConfig(
                name='lr_low',
                transform_lr=1e-4,
                denoiser_lr=1e-4,
            ),
            ExperimentConfig(
                name='alt_1_1',
                transform_epochs_per_cycle=1,
                denoiser_epochs_per_cycle=1,
            ),
            ExperimentConfig(
                name='alt_1_3',
                transform_epochs_per_cycle=1,
                denoiser_epochs_per_cycle=3,
            ),
            ExperimentConfig(
                name='alt_3_1',
                transform_epochs_per_cycle=3,
                denoiser_epochs_per_cycle=1,
            ),
        ]

    @staticmethod
    def homo_weight_sweep() -> List[ExperimentConfig]:
        """Sweep over homoscedasticity loss weights."""
        weights = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        return [
            ExperimentConfig(
                name=f'homo_{w}',
                lambda_homo=w,
            )
            for w in weights
        ]


def create_model(
    n_features: int,
    config: ExperimentConfig,
    device: str = 'cpu',
) -> Dict[str, nn.Module]:
    """Create models based on experiment config."""
    # RQS Transform
    rqs = RQSTransform(
        dim=n_features,
        n_bins=config.n_bins,
        init_scale=0.1,
    )

    # Gauge fixing (optional)
    gauge = GaugeFixingModule(
        dim=n_features,
        method='soft' if config.use_gauge else 'none',
        target_mean=0.0,
        target_std=1.0,
    )

    # Blind-spot denoiser
    denoiser = ColumnMaskedMLP(
        input_dim=n_features,
        hidden_dims=list(config.hidden_dims),
        output_dim=n_features,
    )

    # Move to device
    rqs = rqs.to(device)
    gauge = gauge.to(device)
    denoiser = denoiser.to(device)

    return {
        'rqs': rqs,
        'gauge': gauge,
        'denoiser': denoiser,
    }


def run_single_experiment(
    dataset: SyntheticDataset,
    exp_config: ExperimentConfig,
    n_epochs: int,
    batch_size: int,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single experiment with given configuration."""
    n_samples, n_features = dataset.x.shape

    if verbose:
        print(f"  Running: {exp_config.name}")

    # Create models
    models = create_model(
        n_features=n_features,
        config=exp_config,
        device=device,
    )

    # Create trainer config
    trainer_config = TrainerConfig(
        n_epochs=n_epochs,
        batch_size=batch_size,
        transform_lr=exp_config.transform_lr,
        denoiser_lr=exp_config.denoiser_lr,
        transform_epochs_per_cycle=exp_config.transform_epochs_per_cycle,
        denoiser_epochs_per_cycle=exp_config.denoiser_epochs_per_cycle,
        lambda_homo=exp_config.lambda_homo,
        lambda_gauge=exp_config.lambda_gauge,
    )

    # Prepare data
    y_train = torch.tensor(dataset.x, dtype=torch.float32).to(device)

    # Create trainer
    trainer = AlternatingTrainer(
        transform=models['rqs'],
        denoiser=models['denoiser'],
        gauge_module=models['gauge'],
        config=trainer_config,
    )

    # Train
    start_time = time.time()
    history = trainer.train(y_train, verbose=False)
    train_time = time.time() - start_time

    # Evaluate
    with torch.no_grad():
        z_learned = models['rqs'](y_train)
        z_learned = models['gauge'](z_learned)
        z_learned_np = z_learned.cpu().numpy()

        z_hat = models['denoiser'](z_learned)
        z_hat_np = z_hat.cpu().numpy()

    # Compute all metrics
    residuals = z_learned_np - z_hat_np
    metrics = {
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_type': dataset.noise_type,
    }

    # Additional metrics
    vf_result = variance_flatness_score(z_hat_np, residuals)
    metrics['variance_flatness_cv'] = float(vf_result.cv)
    metrics['variance_flatness_is_flat'] = bool(vf_result.is_flat)
    metrics['variance_flatness_functional'] = float(variance_flatness_functional(z_hat_np, residuals))
    metrics['train_time'] = train_time

    # Oracle comparison
    if dataset.oracle_transform is not None:
        oracle_result = compare_with_oracle(
            learned_transform=lambda x: models['rqs'](torch.tensor(x, dtype=torch.float32).to(device)).detach().cpu().numpy(),
            oracle_transform=dataset.oracle_transform,
            x_test=dataset.x[:min(1000, n_samples)],
        )
        metrics['oracle_correlation'] = float(oracle_result.correlation)
        metrics['oracle_mse'] = float(oracle_result.mse)

    # Residual quality
    residual_quality = assess_residual_quality(residuals)
    metrics['residual_is_gaussian'] = residual_quality.is_gaussian_like
    metrics['residual_skewness'] = residual_quality.skewness
    metrics['residual_kurtosis'] = residual_quality.kurtosis

    return {
        'config': asdict(exp_config),
        'models': models,
        'history': history,
        'metrics': metrics,
        'z_learned': z_learned_np,
        'z_hat': z_hat_np,
    }


def run_dataset_experiments(
    dataset: SyntheticDataset,
    configs: List[ExperimentConfig],
    n_epochs: int,
    batch_size: int,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run all experiment configs on a single dataset."""
    results = {}

    for config in configs:
        try:
            result = run_single_experiment(
                dataset=dataset,
                exp_config=config,
                n_epochs=n_epochs,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )
            results[config.name] = result
        except Exception as e:
            print(f"    Error in {config.name}: {e}")
            results[config.name] = {'error': str(e)}

    return results


def run_ablation_study(
    datasets: Dict[str, SyntheticDataset],
    ablation_type: str,
    n_epochs: int,
    batch_size: int,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run ablation study across all datasets."""
    # Get ablation configs
    if ablation_type == 'loss':
        configs = AblationSuite.loss_ablations()
    elif ablation_type == 'architecture':
        configs = AblationSuite.architecture_ablations()
    elif ablation_type == 'training':
        configs = AblationSuite.training_ablations()
    elif ablation_type == 'homo_sweep':
        configs = AblationSuite.homo_weight_sweep()
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {ablation_type} ablation study")
        print(f"Configs: {[c.name for c in configs]}")
        print(f"{'='*60}")

    all_results = {}

    for dataset_name, dataset in datasets.items():
        if verbose:
            print(f"\nDataset: {dataset_name}")

        results = run_dataset_experiments(
            dataset=dataset,
            configs=configs,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )

        all_results[dataset_name] = results

    return all_results


def run_oracle_comparison(
    datasets: Dict[str, SyntheticDataset],
    n_epochs: int,
    batch_size: int,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run experiments with detailed oracle comparison."""
    if verbose:
        print(f"\n{'='*60}")
        print("Running Oracle Comparison Study")
        print(f"{'='*60}")

    # Use default config
    config = ExperimentConfig(name='default')

    results = {}

    for dataset_name, dataset in datasets.items():
        if verbose:
            print(f"\nDataset: {dataset_name}")

        result = run_single_experiment(
            dataset=dataset,
            exp_config=config,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )

        # Add detailed oracle analysis
        if dataset.oracle_transform is not None:
            oracle_z = dataset.oracle_transform(dataset.x)
            n_features = dataset.x.shape[1]

            # Per-feature analysis
            per_feature_corr = []
            per_feature_var_ratio = []

            for j in range(n_features):
                corr = np.corrcoef(result['z_learned'][:, j], oracle_z[:, j])[0, 1]
                var_ratio = np.var(result['z_learned'][:, j]) / (np.var(oracle_z[:, j]) + 1e-8)
                per_feature_corr.append(float(corr))
                per_feature_var_ratio.append(float(var_ratio))

            result['metrics']['per_feature_oracle_corr'] = per_feature_corr
            result['metrics']['per_feature_variance_ratio'] = per_feature_var_ratio
            result['oracle_z'] = oracle_z

        result['dataset'] = dataset
        results[dataset_name] = result

    return results


def save_ablation_results(
    ablation_results: Dict[str, Dict[str, Dict[str, Any]]],
    ablation_type: str,
    output_dir: Path,
):
    """Save ablation study results."""
    ablation_dir = output_dir / 'ablations' / ablation_type
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for comparison
    comparison_data = {}

    for dataset_name, dataset_results in ablation_results.items():
        comparison_data[dataset_name] = {}

        for config_name, result in dataset_results.items():
            if 'error' in result:
                continue

            metrics = result['metrics']
            comparison_data[dataset_name][config_name] = {
                'variance_flatness_cv': metrics.get('variance_flatness_cv', np.nan),
                'train_time': metrics.get('train_time', np.nan),
            }

            if 'oracle_correlation' in metrics:
                comparison_data[dataset_name][config_name]['oracle_correlation'] = \
                    metrics.get('oracle_correlation', np.nan)

    # Save comparison data
    with open(ablation_dir / 'comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)

    # Save detailed results per config
    for dataset_name, dataset_results in ablation_results.items():
        dataset_dir = ablation_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        for config_name, result in dataset_results.items():
            if 'error' in result:
                continue

            config_dir = dataset_dir / config_name
            config_dir.mkdir(exist_ok=True)

            # Save metrics
            metrics_to_save = {}
            for key, value in result['metrics'].items():
                if isinstance(value, (int, float, str, bool)):
                    metrics_to_save[key] = value
                elif isinstance(value, np.ndarray):
                    metrics_to_save[key] = value.tolist()
                elif isinstance(value, dict):
                    metrics_to_save[key] = {
                        k: v if isinstance(v, (int, float, str, bool, list)) else str(v)
                        for k, v in value.items()
                    }

            with open(config_dir / 'metrics.json', 'w') as f:
                json.dump(metrics_to_save, f, indent=2)

    print(f"  Ablation results saved to {ablation_dir}")


def create_ablation_visualizations(
    ablation_results: Dict[str, Dict[str, Dict[str, Any]]],
    ablation_type: str,
    output_dir: Path,
):
    """Create visualizations for ablation study."""
    import matplotlib.pyplot as plt

    figures_dir = output_dir / 'ablations' / ablation_type / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for visualization
    config_names = set()
    dataset_names = list(ablation_results.keys())

    for dataset_results in ablation_results.values():
        config_names.update(dataset_results.keys())

    config_names = sorted(list(config_names))

    # Extract variance flatness scores
    vf_scores = {}
    for dataset_name in dataset_names:
        vf_scores[dataset_name] = {}
        for config_name in config_names:
            if config_name in ablation_results[dataset_name]:
                result = ablation_results[dataset_name][config_name]
                if 'error' not in result:
                    vf_scores[dataset_name][config_name] = \
                        result['metrics'].get('variance_flatness_cv', np.nan)

    # Plot comparison
    fig = plot_ablation_results(
        ablation_results={
            'variance_flatness': vf_scores,
        },
        metric_name='variance_flatness',
        title=f'{ablation_type.title()} Ablation: Variance Flatness',
    )
    fig.savefig(figures_dir / 'variance_flatness_comparison.png', dpi=150, bbox_inches='tight')

    # Extract oracle correlations if available
    oracle_corrs = {}
    for dataset_name in dataset_names:
        oracle_corrs[dataset_name] = {}
        for config_name in config_names:
            if config_name in ablation_results[dataset_name]:
                result = ablation_results[dataset_name][config_name]
                if 'error' not in result and 'oracle_correlation' in result['metrics']:
                    oracle_corrs[dataset_name][config_name] = \
                        result['metrics'].get('oracle_correlation', np.nan)

    if any(oracle_corrs[d] for d in dataset_names):
        fig = plot_ablation_results(
            ablation_results={'oracle_correlation': oracle_corrs},
            metric_name='oracle_correlation',
            title=f'{ablation_type.title()} Ablation: Oracle Correlation',
        )
        fig.savefig(figures_dir / 'oracle_correlation_comparison.png', dpi=150, bbox_inches='tight')

    plt.close('all')
    print(f"  Ablation figures saved to {figures_dir}")


def create_oracle_visualizations(
    oracle_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
):
    """Create visualizations for oracle comparison."""
    import matplotlib.pyplot as plt

    figures_dir = output_dir / 'oracle_comparison' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, result in oracle_results.items():
        dataset_dir = figures_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        dataset = result['dataset']
        z_learned = result['z_learned']
        z_hat = result['z_hat']

        # Transform comparison
        if 'oracle_z' in result:
            fig = plot_transform_comparison(
                learned_transform=result['models']['rqs'],
                oracle_transform=dataset.oracle_transform,
                x_range=(dataset.x.min(), dataset.x.max()),
                title=f'Transform Comparison: {dataset.noise_type}',
            )
            fig.savefig(dataset_dir / 'transform_comparison.png', dpi=150, bbox_inches='tight')

        # Variance flatness
        residuals = z_learned - z_hat
        fig = plot_variance_flatness(z_hat, residuals, title=f'{dataset_name}: Variance Flatness')
        fig.savefig(dataset_dir / 'variance_flatness.png', dpi=150, bbox_inches='tight')

        # Before/after VST
        fig = plot_before_after_vst(dataset.x, z_learned)
        fig.savefig(dataset_dir / 'before_after_vst.png', dpi=150, bbox_inches='tight')

        # Training curves
        fig = plot_training_curves(result['history'])
        fig.savefig(dataset_dir / 'training_curves.png', dpi=150, bbox_inches='tight')

        # Residual diagnostics
        residuals = z_learned - z_hat
        fig = plot_residual_diagnostics(residuals)
        fig.savefig(dataset_dir / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')

    plt.close('all')
    print(f"  Oracle comparison figures saved to {figures_dir}")


def create_summary_report(
    oracle_results: Dict[str, Dict[str, Any]],
    ablation_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
):
    """Create a summary report of all experiments."""
    report_path = output_dir / 'summary_report.md'

    with open(report_path, 'w') as f:
        f.write("# Learnable VST Synthetic Experiments - Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Oracle comparison section
        f.write("## Oracle Comparison Results\n\n")
        f.write("| Dataset | Variance Flatness | Oracle Correlation | Train Time (s) |\n")
        f.write("|---------|------------------|-------------------|----------------|\n")

        for dataset_name, result in oracle_results.items():
            metrics = result['metrics']
            vf = metrics.get('variance_flatness_cv', np.nan)
            oc = metrics.get('oracle_correlation', np.nan)
            tt = metrics.get('train_time', np.nan)
            f.write(f"| {dataset_name} | {vf:.4f} | {oc:.4f} | {tt:.2f} |\n")

        f.write("\n")

        # Ablation sections
        for ablation_type, abl_results in ablation_results.items():
            f.write(f"## {ablation_type.title()} Ablation Study\n\n")

            # Get all config names
            config_names = set()
            for dataset_results in abl_results.values():
                config_names.update(dataset_results.keys())
            config_names = sorted(list(config_names))

            # Build table
            f.write("| Dataset | " + " | ".join(config_names) + " |\n")
            f.write("|---------|" + "|".join(["---"] * len(config_names)) + "|\n")

            for dataset_name, dataset_results in abl_results.items():
                row = [dataset_name]
                for config_name in config_names:
                    if config_name in dataset_results and 'error' not in dataset_results[config_name]:
                        vf = dataset_results[config_name]['metrics'].get('variance_flatness_cv', np.nan)
                        row.append(f"{vf:.4f}")
                    else:
                        row.append("N/A")
                f.write("| " + " | ".join(row) + " |\n")

            f.write("\n")

        # Best configurations
        f.write("## Best Configurations per Dataset\n\n")

        for dataset_name, result in oracle_results.items():
            f.write(f"### {dataset_name}\n\n")

            # Find best across ablations
            best_configs = []

            for ablation_type, abl_results in ablation_results.items():
                if dataset_name in abl_results:
                    best_vf = np.inf
                    best_config = None

                    for config_name, config_result in abl_results[dataset_name].items():
                        if 'error' not in config_result:
                            vf = config_result['metrics'].get('variance_flatness_cv', np.inf)
                            if vf < best_vf:  # Lower CV is better
                                best_vf = vf
                                best_config = config_name

                    if best_config:
                        best_configs.append((ablation_type, best_config, best_vf))

            for abl_type, config, vf in best_configs:
                f.write(f"- **{abl_type}**: {config} (VF: {vf:.4f})\n")

            f.write("\n")

    print(f"  Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run full VST experiment suite with ablations'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['poisson_like', 'multiplicative', 'affine', 'homoscedastic', 'mixed'],
        help='Datasets to use'
    )
    parser.add_argument(
        '--ablations',
        nargs='+',
        default=['loss', 'architecture', 'training'],
        choices=['loss', 'architecture', 'training', 'homo_sweep'],
        help='Ablation studies to run'
    )
    parser.add_argument(
        '--skip_ablations',
        action='store_true',
        help='Skip ablation studies'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5000,
        help='Number of samples'
    )
    parser.add_argument(
        '--n_features',
        type=int,
        default=10,
        help='Number of features'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Training batch size'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/full',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no_viz',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'

    verbose = not args.quiet
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Full VST Experiment Suite")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Datasets: {args.datasets}")
        print(f"Ablations: {args.ablations if not args.skip_ablations else 'skipped'}")
        print(f"Samples: {args.n_samples}, Features: {args.n_features}")
        print(f"Epochs: {args.epochs}")

    # Generate datasets
    if verbose:
        print("\nGenerating datasets...")

    generators = {
        'poisson_like': generate_poisson_like,
        'multiplicative': generate_multiplicative,
        'affine': generate_affine_variance,
        'homoscedastic': generate_homoscedastic,
        'mixed': generate_mixed,
        'challenging': generate_challenging,
    }

    datasets = {}
    for name in args.datasets:
        if name in generators:
            datasets[name] = generators[name](
                n_samples=args.n_samples,
                n_features=args.n_features,
                seed=args.seed,
            )
        else:
            print(f"Warning: Unknown dataset {name}, skipping")

    if not datasets:
        print("No valid datasets specified!")
        return

    # Run oracle comparison
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 1: Oracle Comparison")
        print("=" * 60)

    oracle_results = run_oracle_comparison(
        datasets=datasets,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        verbose=verbose,
    )

    # Save oracle results
    oracle_dir = output_dir / 'oracle_comparison'
    oracle_dir.mkdir(exist_ok=True)

    for dataset_name, result in oracle_results.items():
        dataset_dir = oracle_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        # Save metrics
        metrics_to_save = {}
        for key, value in result['metrics'].items():
            if isinstance(value, (int, float, str, bool)):
                metrics_to_save[key] = value
            elif isinstance(value, np.ndarray):
                metrics_to_save[key] = value.tolist()
            elif isinstance(value, list):
                metrics_to_save[key] = value
            elif isinstance(value, dict):
                metrics_to_save[key] = {
                    k: v if isinstance(v, (int, float, str, bool, list)) else str(v)
                    for k, v in value.items()
                }

        with open(dataset_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

    if not args.no_viz:
        if verbose:
            print("\nCreating oracle comparison visualizations...")
        create_oracle_visualizations(oracle_results, output_dir)

    # Run ablation studies
    ablation_results = {}

    if not args.skip_ablations:
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 2: Ablation Studies")
            print("=" * 60)

        for ablation_type in args.ablations:
            results = run_ablation_study(
                datasets=datasets,
                ablation_type=ablation_type,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                verbose=verbose,
            )

            ablation_results[ablation_type] = results

            # Save ablation results
            save_ablation_results(results, ablation_type, output_dir)

            # Create ablation visualizations
            if not args.no_viz:
                if verbose:
                    print(f"\nCreating {ablation_type} ablation visualizations...")
                create_ablation_visualizations(results, ablation_type, output_dir)

    # Create summary report
    if verbose:
        print("\n" + "=" * 60)
        print("Creating Summary Report")
        print("=" * 60)

    create_summary_report(oracle_results, ablation_results, output_dir)

    # Final summary
    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)

        print("\nOracle Comparison Results:")
        for dataset_name, result in oracle_results.items():
            metrics = result['metrics']
            vf = metrics.get('variance_flatness_cv', np.nan)
            oc = metrics.get('oracle_correlation', np.nan)
            print(f"  {dataset_name}: VF CV={vf:.4f}, Oracle Corr={oc:.4f}")

        if ablation_results:
            print("\nBest ablation configurations (by variance flatness CV, lower is better):")
            for ablation_type, abl_results in ablation_results.items():
                print(f"\n  {ablation_type}:")
                for dataset_name in datasets.keys():
                    if dataset_name in abl_results:
                        best_vf = np.inf
                        best_config = None
                        for config_name, result in abl_results[dataset_name].items():
                            if 'error' not in result:
                                vf = result['metrics'].get('variance_flatness_cv', np.inf)
                                if vf < best_vf:  # Lower CV is better
                                    best_vf = vf
                                    best_config = config_name
                        if best_config:
                            print(f"    {dataset_name}: {best_config} (VF CV={best_vf:.4f})")

        print(f"\nAll results saved to: {output_dir}")
        print("=" * 60)

    return {
        'oracle_results': oracle_results,
        'ablation_results': ablation_results,
    }


if __name__ == '__main__':
    main()
