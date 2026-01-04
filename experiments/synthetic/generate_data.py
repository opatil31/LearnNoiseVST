"""
Synthetic data generation with known ground-truth VSTs.

This module provides synthetic datasets where we know the true
variance-stabilizing transform, allowing us to validate that
the learned transform approximates the oracle.

Supported noise models:
1. Poisson-like: Var(X|μ) = μ, true VST = 2√x (Anscombe)
2. Multiplicative: X = μ(1+ε), true VST = log(x)
3. Affine: Var(X|μ) = a + bμ, true VST = generalized Anscombe
4. Mixed: Different features have different noise models

Each generator returns:
- x: Noisy observations
- mu: True clean signal
- sigma: True noise standard deviation at each point
- oracle_transform: Function implementing the true VST
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Callable, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class SyntheticDataset:
    """Container for synthetic dataset with ground truth."""
    x: np.ndarray  # Noisy observations [N, d]
    mu: np.ndarray  # True clean signal [N, d]
    sigma: np.ndarray  # True noise std [N, d]
    oracle_transform: Callable[[np.ndarray], np.ndarray]
    oracle_inverse: Callable[[np.ndarray], np.ndarray]
    oracle_derivative: Callable[[np.ndarray], np.ndarray]
    noise_type: str
    description: str

    def __len__(self):
        return len(self.x)

    def to_torch(self) -> TensorDataset:
        """Convert to PyTorch TensorDataset."""
        return TensorDataset(torch.from_numpy(self.x).float())

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Get PyTorch DataLoader."""
        return DataLoader(self.to_torch(), batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# Oracle Transforms
# ============================================================================

def anscombe_transform(x: np.ndarray) -> np.ndarray:
    """Anscombe VST for Poisson noise: T(x) = 2√(x + 3/8)."""
    return 2 * np.sqrt(np.maximum(x + 3/8, 0))


def anscombe_inverse(z: np.ndarray) -> np.ndarray:
    """Inverse Anscombe: T^{-1}(z) = (z/2)² - 3/8."""
    return (z / 2) ** 2 - 3/8


def anscombe_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of Anscombe: T'(x) = 1/√(x + 3/8)."""
    return 1 / np.sqrt(np.maximum(x + 3/8, 1e-8))


def log_transform(x: np.ndarray) -> np.ndarray:
    """Log VST for multiplicative noise: T(x) = log(x)."""
    return np.log(np.maximum(x, 1e-8))


def log_inverse(z: np.ndarray) -> np.ndarray:
    """Inverse log: T^{-1}(z) = exp(z)."""
    return np.exp(z)


def log_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of log: T'(x) = 1/x."""
    return 1 / np.maximum(x, 1e-8)


def generalized_anscombe_transform(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Generalized Anscombe for Var(X|μ) = a + bμ.

    T(x) = (2/b) * √(bx + a + b²/4)
    """
    return (2 / b) * np.sqrt(np.maximum(b * x + a + b**2 / 4, 0))


def generalized_anscombe_inverse(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Inverse generalized Anscombe."""
    return ((b * z / 2) ** 2 - a - b**2 / 4) / b


def generalized_anscombe_derivative(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Derivative of generalized Anscombe."""
    return 1 / np.sqrt(np.maximum(b * x + a + b**2 / 4, 1e-8))


def sqrt_transform(x: np.ndarray) -> np.ndarray:
    """Simple square root transform: T(x) = √x."""
    return np.sqrt(np.maximum(x, 0))


def sqrt_inverse(z: np.ndarray) -> np.ndarray:
    """Inverse sqrt: T^{-1}(z) = z²."""
    return z ** 2


def sqrt_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sqrt: T'(x) = 1/(2√x)."""
    return 0.5 / np.sqrt(np.maximum(x, 1e-8))


def identity_transform(x: np.ndarray) -> np.ndarray:
    """Identity transform (for homoscedastic noise)."""
    return x.copy()


def identity_inverse(z: np.ndarray) -> np.ndarray:
    """Inverse identity."""
    return z.copy()


def identity_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of identity = 1."""
    return np.ones_like(x)


# ============================================================================
# Data Generators
# ============================================================================

def generate_poisson_like(
    n_samples: int,
    n_features: int,
    signal_range: Tuple[float, float] = (1, 100),
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Generate data with Poisson-like heteroscedastic noise.

    Noise model: Var(X|μ) = μ (variance proportional to mean)
    True VST: T(x) = 2√(x + 3/8) (Anscombe transform)

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        signal_range: (min, max) for uniform signal generation.
        seed: Random seed.

    Returns:
        SyntheticDataset with ground truth.
    """
    if seed is not None:
        np.random.seed(seed)

    # Clean signal
    mu = np.random.uniform(signal_range[0], signal_range[1], size=(n_samples, n_features))

    # Poisson-like noise: σ = √μ
    sigma = np.sqrt(mu)
    noise = np.random.randn(n_samples, n_features) * sigma

    x = mu + noise
    x = np.maximum(x, 0.1)  # Ensure positive

    return SyntheticDataset(
        x=x,
        mu=mu,
        sigma=sigma,
        oracle_transform=anscombe_transform,
        oracle_inverse=anscombe_inverse,
        oracle_derivative=anscombe_derivative,
        noise_type="poisson_like",
        description=f"Poisson-like noise, Var(X|μ)=μ, signal range {signal_range}",
    )


def generate_multiplicative(
    n_samples: int,
    n_features: int,
    signal_range: Tuple[float, float] = (1, 10),
    noise_cv: float = 0.2,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Generate data with multiplicative noise.

    Noise model: X = μ * (1 + ε), where ε ~ N(0, cv²)
    True VST: T(x) = log(x)

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        signal_range: (min, max) for uniform signal generation.
        noise_cv: Coefficient of variation for noise.
        seed: Random seed.

    Returns:
        SyntheticDataset with ground truth.
    """
    if seed is not None:
        np.random.seed(seed)

    mu = np.random.uniform(signal_range[0], signal_range[1], size=(n_samples, n_features))

    # Multiplicative noise
    noise_mult = 1 + np.random.randn(n_samples, n_features) * noise_cv
    x = mu * noise_mult
    x = np.maximum(x, 0.1)  # Ensure positive

    sigma = mu * noise_cv  # σ = μ * cv

    return SyntheticDataset(
        x=x,
        mu=mu,
        sigma=sigma,
        oracle_transform=log_transform,
        oracle_inverse=log_inverse,
        oracle_derivative=log_derivative,
        noise_type="multiplicative",
        description=f"Multiplicative noise, cv={noise_cv}, signal range {signal_range}",
    )


def generate_affine_variance(
    n_samples: int,
    n_features: int,
    signal_range: Tuple[float, float] = (1, 50),
    a: float = 1.0,
    b: float = 0.5,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Generate data with affine variance model.

    Noise model: Var(X|μ) = a + b*μ
    True VST: Generalized Anscombe

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        signal_range: (min, max) for uniform signal generation.
        a: Constant variance component.
        b: Signal-dependent variance slope.
        seed: Random seed.

    Returns:
        SyntheticDataset with ground truth.
    """
    if seed is not None:
        np.random.seed(seed)

    mu = np.random.uniform(signal_range[0], signal_range[1], size=(n_samples, n_features))

    # Affine variance: Var = a + b*μ
    variance = a + b * mu
    sigma = np.sqrt(variance)
    noise = np.random.randn(n_samples, n_features) * sigma

    x = mu + noise
    x = np.maximum(x, 0.1)

    # Create bound oracle functions
    def oracle_transform(x):
        return generalized_anscombe_transform(x, a, b)

    def oracle_inverse(z):
        return generalized_anscombe_inverse(z, a, b)

    def oracle_derivative(x):
        return generalized_anscombe_derivative(x, a, b)

    return SyntheticDataset(
        x=x,
        mu=mu,
        sigma=sigma,
        oracle_transform=oracle_transform,
        oracle_inverse=oracle_inverse,
        oracle_derivative=oracle_derivative,
        noise_type="affine",
        description=f"Affine variance, Var(X|μ)={a}+{b}μ, signal range {signal_range}",
    )


def generate_homoscedastic(
    n_samples: int,
    n_features: int,
    signal_range: Tuple[float, float] = (-3, 3),
    noise_std: float = 0.5,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Generate data with constant (homoscedastic) noise.

    Noise model: X = μ + ε, where ε ~ N(0, σ²) (constant σ)
    True VST: Identity (no transform needed)

    This is the baseline case where no VST is required.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        signal_range: (min, max) for uniform signal generation.
        noise_std: Constant noise standard deviation.
        seed: Random seed.

    Returns:
        SyntheticDataset with ground truth.
    """
    if seed is not None:
        np.random.seed(seed)

    mu = np.random.uniform(signal_range[0], signal_range[1], size=(n_samples, n_features))

    sigma = np.full_like(mu, noise_std)
    noise = np.random.randn(n_samples, n_features) * noise_std

    x = mu + noise

    return SyntheticDataset(
        x=x,
        mu=mu,
        sigma=sigma,
        oracle_transform=identity_transform,
        oracle_inverse=identity_inverse,
        oracle_derivative=identity_derivative,
        noise_type="homoscedastic",
        description=f"Homoscedastic noise, σ={noise_std}, signal range {signal_range}",
    )


def generate_mixed(
    n_samples: int,
    n_features: int,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Generate mixed-type data with different noise models per feature.

    First half: Poisson-like
    Second half: Multiplicative

    Tests whether per-feature transforms are learned correctly.

    Args:
        n_samples: Number of samples.
        n_features: Number of features (should be even).
        seed: Random seed.

    Returns:
        SyntheticDataset with combined ground truth.
    """
    if seed is not None:
        np.random.seed(seed)

    n_poisson = n_features // 2
    n_mult = n_features - n_poisson

    # Generate each type
    ds_poisson = generate_poisson_like(n_samples, n_poisson, seed=seed)
    ds_mult = generate_multiplicative(
        n_samples, n_mult, seed=seed + 1 if seed else None
    )

    # Combine
    x = np.concatenate([ds_poisson.x, ds_mult.x], axis=1)
    mu = np.concatenate([ds_poisson.mu, ds_mult.mu], axis=1)
    sigma = np.concatenate([ds_poisson.sigma, ds_mult.sigma], axis=1)

    def mixed_transform(x):
        z = np.zeros_like(x)
        z[:, :n_poisson] = anscombe_transform(x[:, :n_poisson])
        z[:, n_poisson:] = log_transform(x[:, n_poisson:])
        return z

    def mixed_inverse(z):
        x = np.zeros_like(z)
        x[:, :n_poisson] = anscombe_inverse(z[:, :n_poisson])
        x[:, n_poisson:] = log_inverse(z[:, n_poisson:])
        return x

    def mixed_derivative(x):
        d = np.zeros_like(x)
        d[:, :n_poisson] = anscombe_derivative(x[:, :n_poisson])
        d[:, n_poisson:] = log_derivative(x[:, n_poisson:])
        return d

    return SyntheticDataset(
        x=x,
        mu=mu,
        sigma=sigma,
        oracle_transform=mixed_transform,
        oracle_inverse=mixed_inverse,
        oracle_derivative=mixed_derivative,
        noise_type="mixed",
        description=f"Mixed: {n_poisson} Poisson-like + {n_mult} multiplicative features",
    )


def generate_challenging(
    n_samples: int,
    n_features: int,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Generate challenging data with high noise and wide dynamic range.

    Poisson-like noise with:
    - Wide signal range (0.1 to 1000)
    - Strong heteroscedasticity

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        seed: Random seed.

    Returns:
        SyntheticDataset with ground truth.
    """
    if seed is not None:
        np.random.seed(seed)

    # Log-uniform signal distribution for wide dynamic range
    log_mu = np.random.uniform(np.log(0.5), np.log(500), size=(n_samples, n_features))
    mu = np.exp(log_mu)

    # Poisson-like noise
    sigma = np.sqrt(mu)
    noise = np.random.randn(n_samples, n_features) * sigma

    x = mu + noise
    x = np.maximum(x, 0.1)

    return SyntheticDataset(
        x=x,
        mu=mu,
        sigma=sigma,
        oracle_transform=anscombe_transform,
        oracle_inverse=anscombe_inverse,
        oracle_derivative=anscombe_derivative,
        noise_type="challenging",
        description="Challenging: Poisson-like with wide dynamic range (0.5-500)",
    )


def generate_correlated_latent(
    n_samples: int,
    n_features: int,
    latent_dim: int = 3,
    signal_range: Tuple[float, float] = (1, 100),
    trend_scale: float = 0.3,
    noise_model: str = "poisson_like",
    noise_cv: float = 0.2,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Generate correlated data via shared latent factors plus smooth trends.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        latent_dim: Latent dimensionality for shared correlations.
        signal_range: (min, max) for scaled signal generation.
        trend_scale: Amplitude of smooth per-feature trend.
        noise_model: "poisson_like" or "multiplicative".
        noise_cv: Coefficient of variation for multiplicative noise.
        seed: Random seed.

    Returns:
        SyntheticDataset with correlated clean signal.
    """
    if seed is not None:
        np.random.seed(seed)

    h = np.random.randn(n_samples, latent_dim)
    w = np.random.randn(latent_dim, n_features)
    mu_latent = h @ w

    mu_latent = (mu_latent - mu_latent.mean(axis=0, keepdims=True)) / (
        mu_latent.std(axis=0, keepdims=True) + 1e-8
    )

    t = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    freq = np.random.uniform(1.0, 3.0, size=(1, n_features))
    phase = np.random.uniform(0.0, 2 * np.pi, size=(1, n_features))
    slope = np.random.uniform(-trend_scale, trend_scale, size=(1, n_features))
    trend = trend_scale * np.sin(2 * np.pi * t * freq + phase) + slope * t

    mu = mu_latent + trend
    mu_min = mu.min()
    mu_max = mu.max()
    mu = (mu - mu_min) / (mu_max - mu_min + 1e-8)
    mu = signal_range[0] + mu * (signal_range[1] - signal_range[0])

    if noise_model == "multiplicative":
        noise_mult = 1 + np.random.randn(n_samples, n_features) * noise_cv
        x = mu * noise_mult
        x = np.maximum(x, 0.1)
        sigma = mu * noise_cv
        oracle_transform = log_transform
        oracle_inverse = log_inverse
        oracle_derivative = log_derivative
        noise_type = "correlated_multiplicative"
    else:
        sigma = np.sqrt(mu)
        noise = np.random.randn(n_samples, n_features) * sigma
        x = mu + noise
        x = np.maximum(x, 0.1)
        oracle_transform = anscombe_transform
        oracle_inverse = anscombe_inverse
        oracle_derivative = anscombe_derivative
        noise_type = "correlated_poisson_like"

    return SyntheticDataset(
        x=x,
        mu=mu,
        sigma=sigma,
        oracle_transform=oracle_transform,
        oracle_inverse=oracle_inverse,
        oracle_derivative=oracle_derivative,
        noise_type=noise_type,
        description=(
            f"Correlated latent ({latent_dim}D) + smooth trend, "
            f"{noise_model} noise, signal range {signal_range}"
        ),
    )


# ============================================================================
# Dataset Splitting
# ============================================================================

def split_dataset(
    dataset: SyntheticDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    """
    Split dataset into train/val/calibration splits.

    Args:
        dataset: Full dataset.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        (train, val, calibration) datasets.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(dataset)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    cal_idx = indices[n_train + n_val:]

    def subset(idx):
        return SyntheticDataset(
            x=dataset.x[idx],
            mu=dataset.mu[idx],
            sigma=dataset.sigma[idx],
            oracle_transform=dataset.oracle_transform,
            oracle_inverse=dataset.oracle_inverse,
            oracle_derivative=dataset.oracle_derivative,
            noise_type=dataset.noise_type,
            description=dataset.description,
        )

    return subset(train_idx), subset(val_idx), subset(cal_idx)


# ============================================================================
# Benchmark Suite
# ============================================================================

def get_benchmark_datasets(
    n_samples: int = 5000,
    n_features: int = 10,
    seed: int = 42,
) -> Dict[str, SyntheticDataset]:
    """
    Get a suite of benchmark datasets for comprehensive evaluation.

    Returns:
        Dict mapping dataset name to SyntheticDataset.
    """
    return {
        "poisson_easy": generate_poisson_like(
            n_samples, n_features, signal_range=(10, 100), seed=seed
        ),
        "poisson_hard": generate_poisson_like(
            n_samples, n_features, signal_range=(1, 100), seed=seed + 1
        ),
        "multiplicative": generate_multiplicative(
            n_samples, n_features, noise_cv=0.2, seed=seed + 2
        ),
        "affine": generate_affine_variance(
            n_samples, n_features, a=1.0, b=0.5, seed=seed + 3
        ),
        "homoscedastic": generate_homoscedastic(
            n_samples, n_features, noise_std=0.5, seed=seed + 4
        ),
        "mixed": generate_mixed(
            n_samples, n_features, seed=seed + 5
        ),
        "challenging": generate_challenging(
            n_samples, n_features, seed=seed + 6
        ),
        "correlated": generate_correlated_latent(
            n_samples, n_features, seed=seed + 7
        ),
    }


if __name__ == "__main__":
    # Quick test
    print("Testing synthetic data generators...")

    for name, ds in get_benchmark_datasets(n_samples=100, n_features=5).items():
        print(f"\n{name}:")
        print(f"  Shape: {ds.x.shape}")
        print(f"  X range: [{ds.x.min():.2f}, {ds.x.max():.2f}]")
        print(f"  μ range: [{ds.mu.min():.2f}, {ds.mu.max():.2f}]")
        print(f"  σ range: [{ds.sigma.min():.2f}, {ds.sigma.max():.2f}]")

        # Test oracle
        z = ds.oracle_transform(ds.x)
        x_rec = ds.oracle_inverse(z)
        error = np.abs(ds.x - x_rec).max()
        print(f"  Oracle invertibility error: {error:.2e}")
