"""Synthetic data experiments for learnable VST."""

from experiments.synthetic.generate_data import (
    SyntheticDataset,
    generate_poisson_like,
    generate_multiplicative,
    generate_affine_variance,
    generate_homoscedastic,
    generate_mixed,
    generate_challenging,
    get_benchmark_datasets,
)

__all__ = [
    'SyntheticDataset',
    'generate_poisson_like',
    'generate_multiplicative',
    'generate_affine_variance',
    'generate_homoscedastic',
    'generate_mixed',
    'generate_challenging',
    'get_benchmark_datasets',
]
