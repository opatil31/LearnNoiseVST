#!/usr/bin/env python
"""
Staged training experiment for learnable VST.

This script uses the StagedTrainer which implements the key insight from
Noise2VST: freezing the denoiser during VST training provides clearer
gradient signals for variance stabilization.

Training Stages:
    Stage 1 (Warmup): Train denoiser with current transform
    Stage 2 (VST Focus): Freeze denoiser, train VST (key improvement)
    Stage 3 (Refinement): Joint fine-tuning with low learning rates

Usage:
    python run_staged.py --dataset multiplicative --output_dir results/staged
    python run_staged.py --dataset all --warmup_epochs 20 --vst_epochs 50
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
from src.training.alternating_trainer import StagedTrainer, StagedTrainerConfig
from src.utils.metrics import (
    variance_flatness_score,
    compare_with_oracle,
)
from src.utils.visualization import (
    plot_transform_comparison,
    plot_variance_flatness,
    plot_training_curves,
    plot_residual_diagnostics,
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
    transform = MonotoneFeatureTransform(
        num_features=n_features,
        num_bins=n_bins,
        track_running_stats=True,
    )

    denoiser = LightweightTabularDenoiser(
        num_features=n_features,
        hidden_dim=hidden_dim,
    )

    transform = transform.to(device)
    denoiser = denoiser.to(device)

    return {
        'transform': transform,
        'denoiser': denoiser,
    }


def run_staged_experiment(
    dataset: SyntheticDataset,
    config: StagedTrainerConfig,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a staged training experiment on a dataset.

    Returns:
        Dictionary with trained models, metrics, and history.
    """
    n_samples, n_features = dataset.x.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running STAGED experiment on: {dataset.noise_type}")
        print(f"  Samples: {n_samples}, Features: {n_features}")
        print(f"  Has oracle: {dataset.oracle_transform is not None}")
        print(f"  Stages: Warmup({config.warmup_epochs}) -> VST({config.vst_epochs}) -> Refine({config.refine_epochs})")
        print(f"{'='*60}")

    # Create models
    models = create_model(
        n_features=n_features,
        n_bins=8,
        device=device,
    )

    # Prepare data
    from torch.utils.data import TensorDataset, DataLoader

    y_train = torch.tensor(dataset.x, dtype=torch.float32)
    train_dataset = TensorDataset(y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Create staged trainer
    trainer = StagedTrainer(
        transform=models['transform'],
        denoiser=models['denoiser'],
        config=config,
    )

    # Initialize transform normalization
    models['transform'].set_input_normalization(y_train.to(device))

    # Train with staged approach
    if verbose:
        print("\nStarting staged training...")

    train_result = trainer.train(train_loader)
    history = train_result["history"]

    # Evaluate
    if verbose:
        print("\nEvaluating...")

    models['transform'].eval()
    with torch.no_grad():
        y_eval = y_train.to(device)
        z_learned = models['transform'](y_eval, update_stats=False)
        z_learned_np = z_learned.cpu().numpy()

        z_hat = models['denoiser'](z_learned)
        z_hat_np = z_hat.cpu().numpy()

    # Compute metrics
    residuals = z_learned_np - z_hat_np
    metrics = {
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_type': dataset.noise_type,
        'training_stages': {
            'warmup_epochs': config.warmup_epochs,
            'vst_epochs': config.vst_epochs,
            'refine_epochs': config.refine_epochs,
        },
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
            x_test=dataset.x[:1000],
        )
        metrics['oracle_correlation'] = oracle_result.correlation
        metrics['oracle_mse'] = oracle_result.mse

    # Extract variance function from VST
    x_grid = torch.linspace(
        float(dataset.x.min()),
        float(dataset.x.max()),
        100
    )
    x_vals, var_vals = trainer.extract_variance_function(x_grid)
    metrics['extracted_variance_function'] = {
        'x': x_vals.numpy().tolist(),
        'variance': var_vals.numpy().tolist(),
    }

    if verbose:
        print(f"\nResults for {dataset.noise_type}:")
        print(f"  Variance Flatness CV: {vf_result.cv:.4f}")
        print(f"  Is Flat: {vf_result.is_flat}")
        if 'oracle_correlation' in metrics:
            print(f"  Oracle Correlation: {metrics['oracle_correlation']:.4f}")
            print(f"  Oracle MSE: {metrics['oracle_mse']:.4f}")

    return {
        'models': models,
        'trainer': trainer,
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
            metrics_to_save[key] = value

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
        n_features = dataset.x.shape[1] if dataset.x.ndim > 1 else 1
        transform_model = results['models']['transform']

        def learned_transform_1d(x_1d):
            if isinstance(x_1d, np.ndarray):
                x_1d = torch.from_numpy(x_1d).float()
            if x_1d.dim() == 1:
                x_1d = x_1d.unsqueeze(1)
            x_full = x_1d.expand(-1, n_features)
            with torch.no_grad():
                transform_model.eval()
                z = transform_model(x_full, update_stats=False)
            return z[:, 0].cpu().numpy()

        def oracle_transform_1d(x_1d):
            if x_1d.ndim == 1:
                x_1d = x_1d.reshape(-1, 1)
            x_full = np.broadcast_to(x_1d, (len(x_1d), n_features))
            z = dataset.oracle_transform(x_full)
            return z[:, 0]

        fig = plot_transform_comparison(
            learned_transform=learned_transform_1d,
            oracle_transform=oracle_transform_1d,
            x_range=(dataset.x.min(), dataset.x.max()),
            title=f'Transform Comparison: {dataset.noise_type} (Staged Training)',
        )
        fig.savefig(figures_dir / 'transform_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved transform_comparison.png")

    # 2. Variance flatness
    residuals = z_learned - z_hat
    fig = plot_variance_flatness(
        z_hat=z_hat,
        residuals=residuals,
        title=f'Variance Flatness: {dataset.noise_type} (Staged)',
    )
    fig.savefig(figures_dir / 'variance_flatness.png', dpi=150, bbox_inches='tight')
    print(f"  Saved variance_flatness.png")

    # 3. Staged training curves
    fig = plot_staged_training_curves(history)
    fig.savefig(figures_dir / 'staged_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"  Saved staged_training_curves.png")

    # 4. Extracted variance function
    if 'extracted_variance_function' in results['metrics']:
        fig = plot_extracted_variance_function(
            results['metrics']['extracted_variance_function'],
            dataset,
        )
        fig.savefig(figures_dir / 'variance_function.png', dpi=150, bbox_inches='tight')
        print(f"  Saved variance_function.png")

    # 5. Residual diagnostics
    fig = plot_residual_diagnostics(residuals)
    fig.savefig(figures_dir / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved residual_diagnostics.png")

    import matplotlib.pyplot as plt
    plt.close('all')


def plot_staged_training_curves(history: Dict) -> 'matplotlib.figure.Figure':
    """Plot training curves with stage boundaries."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    stages = history.get('stage', [])
    epochs = list(range(len(stages)))

    # Find stage boundaries
    stage_changes = [0]
    for i in range(1, len(stages)):
        if stages[i] != stages[i-1]:
            stage_changes.append(i)
    stage_changes.append(len(stages))

    stage_colors = ['#e6f3ff', '#fff3e6', '#e6ffe6']
    stage_names = ['Stage 1: Warmup', 'Stage 2: VST Focus', 'Stage 3: Refinement']

    # Plot 1: Transform loss
    ax = axes[0, 0]
    t_loss = history.get('transform_loss', [])
    if t_loss:
        ax.plot(epochs, t_loss, 'b-', label='Transform Loss')
    for i in range(len(stage_changes) - 1):
        if i < len(stage_colors):
            ax.axvspan(stage_changes[i], stage_changes[i+1], alpha=0.3, color=stage_colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Transform Loss by Stage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Denoiser loss
    ax = axes[0, 1]
    d_loss = history.get('denoiser_loss', [])
    if d_loss:
        ax.plot(epochs, d_loss, 'r-', label='Denoiser Loss')
    for i in range(len(stage_changes) - 1):
        if i < len(stage_colors):
            ax.axvspan(stage_changes[i], stage_changes[i+1], alpha=0.3, color=stage_colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Denoiser Loss by Stage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Binned variance loss
    ax = axes[1, 0]
    binned = history.get('binned_loss', [])
    if binned:
        ax.plot(epochs, binned, 'g-', label='Binned Variance Loss')
    for i in range(len(stage_changes) - 1):
        if i < len(stage_colors):
            ax.axvspan(stage_changes[i], stage_changes[i+1], alpha=0.3, color=stage_colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Binned Variance Loss by Stage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Stage legend
    ax = axes[1, 1]
    ax.axis('off')
    for i, (name, color) in enumerate(zip(stage_names, stage_colors)):
        ax.add_patch(plt.Rectangle((0.1, 0.7 - i*0.25), 0.15, 0.15,
                                    facecolor=color, edgecolor='black'))
        ax.text(0.3, 0.75 - i*0.25, name, fontsize=12, va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Training Stages')

    plt.tight_layout()
    return fig


def plot_extracted_variance_function(
    var_func: Dict,
    dataset: SyntheticDataset,
) -> 'matplotlib.figure.Figure':
    """Plot extracted variance function vs ground truth."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array(var_func['x'])
    var = np.array(var_func['variance'])

    # Normalize variance for comparison
    var = var / var.mean()

    ax.plot(x, var, 'b-', linewidth=2, label='Extracted from VST')

    # Plot ground truth if available
    if dataset.noise_type == 'poisson_like':
        # σ²(μ) ∝ μ for Poisson
        x_sorted = np.sort(x)
        var_true = x_sorted / x_sorted.mean()
        ax.plot(x_sorted, var_true, 'r--', linewidth=2, label='Ground Truth (∝ μ)')
    elif dataset.noise_type == 'multiplicative':
        # σ²(μ) ∝ μ² for multiplicative
        x_sorted = np.sort(x)
        var_true = (x_sorted ** 2) / (x_sorted ** 2).mean()
        ax.plot(x_sorted, var_true, 'r--', linewidth=2, label='Ground Truth (∝ μ²)')

    ax.set_xlabel('Signal Level (μ)')
    ax.set_ylabel('Normalized Variance σ²(μ)')
    ax.set_title(f'Extracted Variance Function: {dataset.noise_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Run staged VST experiment on synthetic data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='multiplicative',
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
        '--warmup_epochs',
        type=int,
        default=20,
        help='Stage 1: Denoiser warmup epochs'
    )
    parser.add_argument(
        '--vst_epochs',
        type=int,
        default=50,
        help='Stage 2: VST training epochs (frozen denoiser)'
    )
    parser.add_argument(
        '--refine_epochs',
        type=int,
        default=30,
        help='Stage 3: Joint refinement epochs'
    )
    parser.add_argument(
        '--noise_std',
        type=float,
        default=1.0,
        help='Stage 1: Synthetic Gaussian noise std (key for Noise2VST-style training)'
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
        default='results/staged',
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
        print("STAGED VST Experiment (with Synthetic Gaussian Noise)")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {args.n_samples}, Features: {args.n_features}")
        print(f"Training Stages:")
        print(f"  Stage 1 (Warmup):  {args.warmup_epochs} epochs (synthetic noise σ={args.noise_std})")
        print(f"  Stage 2 (VST):     {args.vst_epochs} epochs (frozen denoiser)")
        print(f"  Stage 3 (Refine):  {args.refine_epochs} epochs")
        print(f"\nKey insight: Stage 1 uses SYNTHETIC Gaussian noise so denoiser")
        print(f"             expects homoscedastic input. This creates strong")
        print(f"             gradient signal for VST learning in Stage 2.")

    # Create staged training config
    config = StagedTrainerConfig(
        warmup_epochs=args.warmup_epochs,
        warmup_noise_std=args.noise_std,  # Key parameter!
        vst_epochs=args.vst_epochs,
        refine_epochs=args.refine_epochs,
        batch_size=args.batch_size,
        device=device,
        log_every=5,
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

        # Run staged experiment
        results = run_staged_experiment(
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
        print("STAGED EXPERIMENT SUMMARY")
        print("=" * 60)

        for name, results in all_results.items():
            metrics = results['metrics']
            print(f"\n{name}:")
            print(f"  Variance Flatness CV: {metrics.get('variance_flatness_cv', 'N/A'):.4f}")
            if 'oracle_correlation' in metrics:
                print(f"  Oracle Correlation: {metrics['oracle_correlation']:.4f}")
                print(f"  Oracle MSE: {metrics['oracle_mse']:.4f}")

        print(f"\nResults saved to: {args.output_dir}")
        print("=" * 60)

    return all_results


if __name__ == '__main__':
    main()
