#!/usr/bin/env python
"""
Minimal synthetic experiment for learnable VST.

This script demonstrates the core functionality with a single dataset
and basic evaluation. For comprehensive experiments with ablations,
use run_full.py instead.

Usage:
    python run_minimal.py --dataset poisson_like --output_dir results/minimal
    python run_minimal.py --dataset all --epochs 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.transforms import MonotoneFeatureTransform
from src.denoisers import LightweightTabularDenoiser
from src.training.losses import CombinedTransformLoss, DenoiserLoss
from src.training.alternating_trainer import AlternatingTrainer, TrainerConfig
from src.noise_model import NoiseModelSampler
from src.utils.metrics import (
    compute_all_metrics,
    variance_flatness_score,
    compare_with_oracle,
)
from src.utils.visualization import (
    plot_transform_comparison,
    plot_variance_flatness,
    plot_training_curves,
    plot_residual_diagnostics,
    create_experiment_dashboard,
)
from experiments.synthetic.generate_data import (
    generate_poisson_like,
    generate_multiplicative,
    generate_affine_variance,
    generate_homoscedastic,
    generate_mixed,
    get_benchmark_datasets,
    SyntheticDataset,
)


def create_model(
    n_features: int,
    n_bins: int = 8,
    hidden_dim: int = 64,
    device: str = 'cpu',
) -> Dict[str, nn.Module]:
    """Create transform and denoiser models."""
    # MonotoneFeatureTransform includes RQS with gauge fixing (running stats)
    transform = MonotoneFeatureTransform(
        num_features=n_features,
        num_bins=n_bins,
        track_running_stats=True,
    )

    # Blind-spot denoiser
    denoiser = LightweightTabularDenoiser(
        num_features=n_features,
        hidden_dim=hidden_dim,
    )

    # Move to device
    transform = transform.to(device)
    denoiser = denoiser.to(device)

    return {
        'transform': transform,
        'denoiser': denoiser,
    }


def run_experiment(
    dataset: SyntheticDataset,
    config: TrainerConfig,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single experiment on a dataset.

    Returns:
        Dictionary with trained models, metrics, and history.
    """
    n_samples, n_features = dataset.x.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment on: {dataset.noise_type}")
        print(f"  Samples: {n_samples}, Features: {n_features}")
        print(f"  Has oracle: {dataset.oracle_transform is not None}")
        print(f"{'='*60}")

    # Create models
    models = create_model(
        n_features=n_features,
        n_bins=config.rqs_bins if hasattr(config, 'rqs_bins') else 8,
        device=device,
    )

    # Prepare data
    from torch.utils.data import TensorDataset, DataLoader

    y_train = torch.tensor(dataset.x, dtype=torch.float32)
    train_dataset = TensorDataset(y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.transform_batch_size,
        shuffle=True,
    )

    # Create trainer
    trainer = AlternatingTrainer(
        transform=models['transform'],
        denoiser=models['denoiser'],
        config=config,
    )

    # Train
    if verbose:
        print("\nTraining...")

    # Initialize transform normalization
    models['transform'].set_input_normalization(y_train.to(device))

    train_result = trainer.train(train_loader)
    history = train_result["history"]

    # Evaluate
    if verbose:
        print("\nEvaluating...")

    models['transform'].eval()
    with torch.no_grad():
        # Apply learned transform (includes gauge fixing via running stats)
        y_eval = y_train.to(device)
        z_learned = models['transform'](y_eval, update_stats=False)
        z_learned_np = z_learned.cpu().numpy()

        # Get denoiser predictions
        z_hat = models['denoiser'](z_learned)
        z_hat_np = z_hat.cpu().numpy()

    # Compute metrics
    residuals = z_learned_np - z_hat_np
    metrics = {
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_type': dataset.noise_type,
    }

    # Variance flatness
    vf_result = variance_flatness_score(z_hat_np, residuals)
    metrics['variance_flatness_cv'] = vf_result.cv
    metrics['variance_flatness_is_flat'] = vf_result.is_flat

    # Oracle comparison if available
    if dataset.oracle_transform is not None:
        def learned_transform_fn(x):
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32).to(device)
                return models['transform'](x_t, update_stats=False).cpu().numpy()

        oracle_result = compare_with_oracle(
            learned_transform=learned_transform_fn,
            oracle_transform=dataset.oracle_transform,
            x_test=dataset.x[:1000],  # Subsample for efficiency
        )
        metrics['oracle_correlation'] = oracle_result.correlation
        metrics['oracle_mse'] = oracle_result.mse

    if verbose:
        print(f"\nResults for {dataset.noise_type}:")
        print(f"  Variance Flatness CV: {vf_result.cv:.4f}")
        print(f"  Is Flat: {vf_result.is_flat}")
        if 'oracle_correlation' in metrics:
            print(f"  Oracle Correlation: {metrics['oracle_correlation']:.4f}")

    return {
        'models': models,
        'history': history,
        'metrics': metrics,
        'dataset': dataset,
        'z_learned': z_learned_np,
        'z_hat': z_hat_np,
    }


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_models: bool = True,
):
    """Save experiment results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_to_save = {}
    for key, value in results['metrics'].items():
        if isinstance(value, (int, float, str, bool)):
            metrics_to_save[key] = value
        elif isinstance(value, np.ndarray):
            metrics_to_save[key] = value.tolist()
        elif isinstance(value, dict):
            metrics_to_save[key] = {
                k: v if isinstance(v, (int, float, str, bool)) else str(v)
                for k, v in value.items()
            }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    # Save training history
    history_to_save = {}
    for key, value in results['history'].items():
        if isinstance(value, list):
            history_to_save[key] = value
        elif isinstance(value, np.ndarray):
            history_to_save[key] = value.tolist()

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history_to_save, f, indent=2)

    # Save models
    if save_models:
        model_dir = output_dir / 'models'
        model_dir.mkdir(exist_ok=True)

        torch.save(
            results['models']['transform'].state_dict(),
            model_dir / 'transform.pt'
        )
        torch.save(
            results['models']['denoiser'].state_dict(),
            model_dir / 'denoiser.pt'
        )

    # Save arrays
    np.savez(
        output_dir / 'arrays.npz',
        z_learned=results['z_learned'],
        z_hat=results['z_hat'],
        x=results['dataset'].x,
        mu=results['dataset'].mu,
    )

    print(f"Results saved to {output_dir}")


def create_visualizations(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Create and save visualizations."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = results['dataset']
    z_learned = results['z_learned']
    z_hat = results['z_hat']
    history = results['history']

    # 1. Transform comparison
    if dataset.oracle_transform is not None:
        fig = plot_transform_comparison(
            learned_transform=results['models']['transform'],
            oracle_transform=dataset.oracle_transform,
            x_range=(dataset.x.min(), dataset.x.max()),
            title=f'Transform Comparison: {dataset.noise_type}',
        )
        fig.savefig(figures_dir / 'transform_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved transform_comparison.png")

    # 2. Variance flatness
    residuals = z_learned - z_hat
    fig = plot_variance_flatness(
        z_hat=z_hat,
        residuals=residuals,
        title=f'Variance Flatness: {dataset.noise_type}',
    )
    fig.savefig(figures_dir / 'variance_flatness.png', dpi=150, bbox_inches='tight')
    print(f"  Saved variance_flatness.png")

    # 3. Training curves
    fig = plot_training_curves(history)
    fig.savefig(figures_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"  Saved training_curves.png")

    # 4. Residual diagnostics
    residuals = z_learned - z_hat
    fig = plot_residual_diagnostics(residuals)
    fig.savefig(figures_dir / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved residual_diagnostics.png")

    # Close figures to free memory
    import matplotlib.pyplot as plt
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Run minimal VST experiment on synthetic data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='poisson_like',
        choices=['poisson_like', 'multiplicative', 'affine', 'homoscedastic', 'mixed', 'all'],
        help='Dataset type to use'
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
        default='results/minimal',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for training'
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

    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("Minimal VST Experiment")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {args.n_samples}, Features: {args.n_features}")
        print(f"Epochs: {args.epochs}")

    # Create training config
    config = TrainerConfig(
        num_outer_iters=args.epochs,
        transform_batch_size=args.batch_size,
        denoiser_batch_size=args.batch_size,
        transform_lr=1e-3,
        denoiser_lr=1e-3,
        lambda_homo=1.0,
        device=device,
    )

    # Generate datasets
    if args.dataset == 'all':
        datasets = get_benchmark_datasets(
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
        )
    else:
        generators = {
            'poisson_like': generate_poisson_like,
            'multiplicative': generate_multiplicative,
            'affine': generate_affine_variance,
            'homoscedastic': generate_homoscedastic,
            'mixed': generate_mixed,
        }
        datasets = {
            args.dataset: generators[args.dataset](
                n_samples=args.n_samples,
                n_features=args.n_features,
                seed=args.seed,
            )
        }

    # Run experiments
    all_results = {}

    for name, dataset in datasets.items():
        output_dir = Path(args.output_dir) / name

        # Run experiment
        results = run_experiment(
            dataset=dataset,
            config=config,
            device=device,
            verbose=verbose,
        )

        # Save results
        save_results(results, output_dir)

        # Create visualizations
        if not args.no_viz:
            if verbose:
                print("\nCreating visualizations...")
            create_visualizations(results, output_dir)

        all_results[name] = results

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        for name, results in all_results.items():
            metrics = results['metrics']
            print(f"\n{name}:")
            print(f"  Variance Flatness CV: {metrics.get('variance_flatness_cv', 'N/A'):.4f}")
            if 'oracle_correlation' in metrics:
                print(f"  Oracle Correlation: {metrics['oracle_correlation']:.4f}")

        print(f"\nResults saved to: {args.output_dir}")
        print("=" * 60)

    return all_results


if __name__ == '__main__':
    main()
