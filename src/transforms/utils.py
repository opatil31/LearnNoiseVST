"""
Utility functions for transforms.

Includes:
- Classical variance-stabilizing transforms (for initialization/comparison)
- Derivative bound checking
- Smoothness penalties
- Data range estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# =============================================================================
# Classical Variance-Stabilizing Transforms (for reference/initialization)
# =============================================================================

def anscombe_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Anscombe transform for Poisson-distributed data.

    T(x) = 2 * sqrt(x + 3/8)

    Stabilizes variance for Poisson noise where Var(X) ≈ E[X].
    """
    return 2 * torch.sqrt(x + 3 / 8)


def anscombe_inverse(y: torch.Tensor) -> torch.Tensor:
    """
    Inverse Anscombe transform.

    T^{-1}(y) = (y/2)^2 - 3/8
    """
    return (y / 2).pow(2) - 3 / 8


def generalized_anscombe_transform(x: torch.Tensor, gain: float = 1.0,
                                    sigma: float = 0.0, mu: float = 0.0) -> torch.Tensor:
    """
    Generalized Anscombe transform for Poisson-Gaussian mixture noise.

    For data: X = gain * P + N, where P ~ Poisson(λ), N ~ N(mu, sigma^2)

    T(x) = (2/gain) * sqrt(gain*x + (3/8)*gain^2 + sigma^2 - gain*mu)
    """
    return (2 / gain) * torch.sqrt(
        torch.clamp(gain * x + (3 / 8) * gain ** 2 + sigma ** 2 - gain * mu, min=1e-8)
    )


def freeman_tukey_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Freeman-Tukey transform for Poisson data.

    T(x) = sqrt(x) + sqrt(x + 1)
    """
    return torch.sqrt(x) + torch.sqrt(x + 1)


def log_transform(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Log transform for multiplicative/exponential noise.

    T(x) = log(x + epsilon)

    Stabilizes variance when Var(X) ∝ E[X]^2 (constant CV).
    """
    return torch.log(x + epsilon)


def box_cox_transform(x: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Box-Cox transform with parameter lambda.

    T(x) = (x^λ - 1) / λ  if λ ≠ 0
         = log(x)         if λ = 0

    Data must be positive.
    """
    if abs(lam) < 1e-8:
        return torch.log(x)
    else:
        return (x.pow(lam) - 1) / lam


# =============================================================================
# Data Range Estimation
# =============================================================================

def estimate_data_bounds(
    x: torch.Tensor,
    quantiles: Tuple[float, float] = (0.001, 0.999),
    margin: float = 0.1
) -> Tuple[float, float]:
    """
    Estimate robust data bounds using quantiles.

    Args:
        x: Data tensor (any shape, will be flattened).
        quantiles: (low_q, high_q) quantiles to use.
        margin: Fractional margin to add beyond quantiles.

    Returns:
        (lower_bound, upper_bound)
    """
    x_flat = x.flatten()
    q_low = torch.quantile(x_flat, quantiles[0]).item()
    q_high = torch.quantile(x_flat, quantiles[1]).item()

    range_width = q_high - q_low
    lower = q_low - margin * range_width
    upper = q_high + margin * range_width

    return lower, upper


def estimate_per_feature_bounds(
    x: torch.Tensor,
    quantiles: Tuple[float, float] = (0.001, 0.999),
    margin: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate bounds per feature.

    Args:
        x: Data tensor of shape [batch, num_features, ...].
        quantiles: (low_q, high_q) quantiles.
        margin: Fractional margin.

    Returns:
        (lower_bounds, upper_bounds) each of shape [num_features]
    """
    num_features = x.shape[1]
    device = x.device
    dtype = x.dtype

    # Reshape to [batch * ..., num_features] then transpose
    x_flat = x.transpose(1, -1).reshape(-1, num_features)  # [N, F]

    q_low = torch.quantile(x_flat, quantiles[0], dim=0)
    q_high = torch.quantile(x_flat, quantiles[1], dim=0)

    range_width = q_high - q_low
    lower = q_low - margin * range_width
    upper = q_high + margin * range_width

    return lower, upper


# =============================================================================
# Derivative and Smoothness Utilities
# =============================================================================

def check_derivative_bounds(
    transform: nn.Module,
    x_samples: torch.Tensor,
    min_deriv: float = 0.01,
    max_deriv: float = 100.0
) -> Tuple[bool, float, float]:
    """
    Check if transform derivatives are within bounds.

    Args:
        transform: Transform module with .derivative() method.
        x_samples: Sample points to check.
        min_deriv: Minimum allowed derivative.
        max_deriv: Maximum allowed derivative.

    Returns:
        (within_bounds, actual_min, actual_max)
    """
    with torch.no_grad():
        derivs = transform.derivative(x_samples)
        actual_min = derivs.min().item()
        actual_max = derivs.max().item()
        within_bounds = (actual_min >= min_deriv) and (actual_max <= max_deriv)

    return within_bounds, actual_min, actual_max


def derivative_bound_penalty(
    derivs: torch.Tensor,
    min_deriv: float = 0.1,
    max_deriv: float = 10.0
) -> torch.Tensor:
    """
    Soft penalty for derivatives outside [min_deriv, max_deriv].

    Returns squared penalty: sum of (violation)^2.
    """
    below_min = F.relu(min_deriv - derivs)
    above_max = F.relu(derivs - max_deriv)
    return (below_min.pow(2) + above_max.pow(2)).mean()


def smoothness_penalty(
    transform: nn.Module,
    x_samples: torch.Tensor,
    order: int = 2
) -> torch.Tensor:
    """
    Penalty on the smoothness of the transform.

    For order=2, penalizes curvature (second derivative).
    Approximated via finite differences of T'(x).

    Args:
        transform: Transform module with .derivative() method.
        x_samples: Sorted sample points [N] or [N, F].
        order: Derivative order to penalize (1=slope variation, 2=curvature).

    Returns:
        Penalty value.
    """
    derivs = transform.derivative(x_samples)

    # Finite difference approximation of higher derivatives
    for _ in range(order - 1):
        if derivs.dim() == 1:
            derivs = derivs[1:] - derivs[:-1]
        else:
            derivs = derivs[1:, :] - derivs[:-1, :]

    return derivs.pow(2).mean()


# =============================================================================
# Initialization Utilities
# =============================================================================

def initialize_rqs_from_classical(
    rqs_module: nn.Module,
    classical_transform: str,
    data_samples: torch.Tensor,
    num_grid_points: int = 100
):
    """
    Initialize RQS parameters to approximate a classical transform.

    This helps with the bootstrap problem by starting near a known good VST.

    Args:
        rqs_module: RQS module to initialize.
        classical_transform: One of 'anscombe', 'log', 'freeman_tukey', 'identity'.
        data_samples: Representative data samples.
        num_grid_points: Number of points for fitting.
    """
    # Get data range
    x_min, x_max = estimate_data_bounds(data_samples)

    # Create grid in data space
    x_grid = torch.linspace(x_min, x_max, num_grid_points)

    # Evaluate classical transform
    if classical_transform == 'anscombe':
        y_grid = anscombe_transform(x_grid)
    elif classical_transform == 'log':
        y_grid = log_transform(x_grid)
    elif classical_transform == 'freeman_tukey':
        y_grid = freeman_tukey_transform(x_grid)
    elif classical_transform == 'identity':
        y_grid = x_grid.clone()
    else:
        raise ValueError(f"Unknown classical transform: {classical_transform}")

    # TODO: Fit RQS parameters to approximate (x_grid, y_grid) mapping
    # This is non-trivial and would require solving a constrained optimization.
    # For now, we just note this as future work.
    # A simple approach: use the derivative at each knot location.

    print(f"Warning: initialize_rqs_from_classical not fully implemented. "
          f"Using identity initialization.")
    rqs_module.set_to_identity()


# =============================================================================
# Numerical Stability Utilities
# =============================================================================

def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log with numerical safety."""
    return torch.log(x.clamp(min=eps))


def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Square root with numerical safety."""
    return torch.sqrt(x.clamp(min=eps))


def soft_clamp(x: torch.Tensor, min_val: float, max_val: float,
               sharpness: float = 10.0) -> torch.Tensor:
    """
    Soft clamping using sigmoid.

    Approximately clamps to [min_val, max_val] but remains differentiable.
    """
    range_val = max_val - min_val
    normalized = (x - min_val) / range_val
    clamped = torch.sigmoid(sharpness * (normalized - 0.5))
    return min_val + range_val * clamped


# =============================================================================
# Jacobian Utilities (for leakage checking)
# =============================================================================

def compute_jacobian_diagonal(
    model: nn.Module,
    x: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    num_samples: int = 100
) -> torch.Tensor:
    """
    Compute diagonal elements of the Jacobian ∂y/∂x.

    For checking blind-spot property: diagonal should be ~0.

    Args:
        model: Model to evaluate.
        x: Input tensor.
        indices: Specific indices to check (if None, samples randomly).
        num_samples: Number of diagonal elements to check.

    Returns:
        Tensor of diagonal Jacobian elements.
    """
    x = x.requires_grad_(True)
    y = model(x)

    diag_elements = []

    if indices is None:
        # Random sampling
        flat_size = x.numel()
        indices = torch.randint(0, flat_size, (min(num_samples, flat_size),))

    x_flat = x.flatten()
    y_flat = y.flatten()

    for idx in indices:
        idx = idx.item()
        if idx < y_flat.shape[0]:
            grad = torch.autograd.grad(
                y_flat[idx], x, retain_graph=True, create_graph=False
            )[0]
            diag_elem = grad.flatten()[idx]
            diag_elements.append(diag_elem.detach())

    return torch.stack(diag_elements)
