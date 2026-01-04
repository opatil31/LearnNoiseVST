"""
Loss functions for learnable variance-stabilizing transforms.

This module implements the losses used in the alternating optimization:

1. L_homo (Homoscedasticity Loss):
   Penalizes correlation between predicted signal and residual variance.
   Makes Var(r | ẑ) approximately constant.

2. J[T] (Variance Flatness Functional):
   Directly measures variance of log conditional variances.
   The principled objective from classical VST theory.

3. L_shape (Shape Penalty):
   Encourages standardized residuals to be Gaussian-like.
   Uses robust skewness/kurtosis measures.

4. L_reg (Transform Regularization):
   Smoothness and derivative bound penalties on T.

5. L_prox (Proximity Penalty):
   Trust region to prevent T from changing too much per iteration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math


class HomoscedasticityLoss(nn.Module):
    """
    Homoscedasticity loss: penalizes signal-dependent variance.

    L_homo = Σ_g Σ_j Cov(φ_j(ẑ_g), u_g²)²

    where:
    - g indexes groups (channels for images, features for tabular)
    - φ_j are basis functions (ẑ, ẑ², etc.)
    - u_g = r_g / σ̂_g is the standardized residual

    This loss is zero when residual variance is independent of signal level.

    Args:
        basis_degree: Maximum polynomial degree for basis functions.
        use_spline_basis: If True, use spline basis instead of polynomial.
        num_spline_knots: Number of knots for spline basis.
    """

    def __init__(
        self,
        basis_degree: int = 2,
        use_spline_basis: bool = False,
        num_spline_knots: int = 5,
    ):
        super().__init__()
        self.basis_degree = basis_degree
        self.use_spline_basis = use_spline_basis
        self.num_spline_knots = num_spline_knots

    def forward(
        self,
        z_hat: torch.Tensor,
        residuals: torch.Tensor,
        groups: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute homoscedasticity loss.

        Args:
            z_hat: Predicted clean signal [B, ...].
            residuals: Residuals r = z - ẑ [B, ...].
            groups: Optional list of group indices. If None, treats all as one group.

        Returns:
            Scalar loss value.
        """
        if groups is None:
            # Single group: flatten everything
            return self._compute_group_loss(z_hat.flatten(), residuals.flatten())

        # Multiple groups (e.g., channels)
        loss = 0.0
        for g in groups:
            if z_hat.dim() == 4:  # Image: [B, C, H, W]
                z_hat_g = z_hat[:, g].flatten()
                r_g = residuals[:, g].flatten()
            elif z_hat.dim() == 2:  # Tabular: [B, F]
                z_hat_g = z_hat[:, g]
                r_g = residuals[:, g]
            else:
                raise ValueError(f"Unsupported tensor dimension: {z_hat.dim()}")

            loss = loss + self._compute_group_loss(z_hat_g, r_g)

        return loss / len(groups)

    def _compute_group_loss(
        self,
        z_hat: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a single group."""
        # Standardize residuals
        r_var = r.var() + 1e-8
        u_sq = (r ** 2) / r_var  # Standardized squared residual

        loss = torch.tensor(0.0, device=z_hat.device, dtype=z_hat.dtype)

        if self.use_spline_basis:
            # Use spline basis functions
            basis = self._compute_spline_basis(z_hat)
        else:
            # Polynomial basis: φ_j(x) = x^j for j = 1, ..., degree
            basis = [z_hat ** j for j in range(1, self.basis_degree + 1)]

        for phi in basis:
            # Compute covariance: Cov(φ, u²) = E[φ * u²] - E[φ] * E[u²]
            phi_centered = phi - phi.mean()
            u_sq_centered = u_sq - u_sq.mean()

            cov = (phi_centered * u_sq_centered).mean()
            loss = loss + cov ** 2

        return loss

    def _compute_spline_basis(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute B-spline basis functions."""
        # Simple piecewise linear spline basis
        x_min = x.min().item()
        x_max = x.max().item()
        knots = torch.linspace(x_min, x_max, self.num_spline_knots + 2, device=x.device)

        basis = []
        for i in range(1, len(knots) - 1):
            # Tent function centered at knot i
            left = (x - knots[i - 1]) / (knots[i] - knots[i - 1] + 1e-8)
            right = (knots[i + 1] - x) / (knots[i + 1] - knots[i] + 1e-8)
            tent = torch.clamp(torch.min(left, right), min=0)
            basis.append(tent)

        return basis


class VarianceFlatnessLoss(nn.Module):
    """
    Variance flatness functional J[T].

    J[T] = Var(log σ²(ẑ))

    where σ²(ẑ) is the conditional variance of residuals given ẑ.

    This is the principled loss from VST theory. Minimizing J[T] gives
    the classical VST condition T'(μ)²V(μ) = constant.

    Uses kernel-smoothed local variance estimation for differentiability.

    Args:
        bandwidth: Kernel bandwidth for local variance estimation.
            If 'auto', uses Silverman's rule of thumb.
        min_var: Minimum variance for numerical stability.
        subsample: If > 0, subsample to this many points for efficiency.
    """

    def __init__(
        self,
        bandwidth: Union[float, str] = 'auto',
        min_var: float = 1e-6,
        subsample: int = 1000,
    ):
        super().__init__()
        self.bandwidth_param = bandwidth
        self.min_var = min_var
        self.subsample = subsample

    def forward(
        self,
        z_hat: torch.Tensor,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute variance flatness loss.

        Args:
            z_hat: Predicted signal (flattened or will be flattened).
            residuals: Residuals r = z - ẑ.

        Returns:
            J[T] = Var(log local_variance).
        """
        z_hat_flat = z_hat.flatten()
        r_flat = residuals.flatten()

        n = z_hat_flat.shape[0]

        # Subsample for efficiency
        if self.subsample > 0 and n > self.subsample:
            idx = torch.randperm(n, device=z_hat.device)[:self.subsample]
            z_hat_flat = z_hat_flat[idx]
            r_flat = r_flat[idx]
            n = self.subsample

        # Compute bandwidth
        if self.bandwidth_param == 'auto':
            # Silverman's rule of thumb
            std = z_hat_flat.std()
            iqr = torch.quantile(z_hat_flat, 0.75) - torch.quantile(z_hat_flat, 0.25)
            bandwidth = 0.9 * torch.min(std, iqr / 1.34) * (n ** (-0.2))
            bandwidth = bandwidth.clamp(min=0.1)
        else:
            bandwidth = self.bandwidth_param

        # Compute kernel weights: K(z_i, z_j) = exp(-|z_i - z_j|² / (2h²))
        # This is O(n²) but with subsampling it's manageable
        diffs = z_hat_flat.unsqueeze(0) - z_hat_flat.unsqueeze(1)  # [n, n]
        weights = torch.exp(-diffs ** 2 / (2 * bandwidth ** 2))

        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Local variance estimate at each point
        r_sq = r_flat ** 2
        local_var = (weights * r_sq.unsqueeze(0)).sum(dim=1)  # [n]
        local_var = local_var.clamp(min=self.min_var)

        # Variance of log local variance
        log_var = torch.log(local_var)
        J_T = log_var.var()

        return J_T


class ShapePenalty(nn.Module):
    """
    Shape penalty for residual distribution.

    L_shape = Σ_g (skew(u_g)² + α(kurt(u_g) - 3)²)

    Encourages standardized residuals to be Gaussian-like.
    Uses robust estimators (median-based) to handle outliers.

    Args:
        kurt_weight: Weight for kurtosis term relative to skewness.
        use_robust: If True, use median-based robust estimators.
        target_skew: Target skewness (default 0 for symmetric).
        target_kurt: Target kurtosis (default 3 for Gaussian).
    """

    def __init__(
        self,
        kurt_weight: float = 0.1,
        use_robust: bool = True,
        target_skew: float = 0.0,
        target_kurt: float = 3.0,
    ):
        super().__init__()
        self.kurt_weight = kurt_weight
        self.use_robust = use_robust
        self.target_skew = target_skew
        self.target_kurt = target_kurt

    def forward(
        self,
        standardized_residuals: torch.Tensor,
        groups: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute shape penalty.

        Args:
            standardized_residuals: u = r / σ̂, should have mean≈0, var≈1.
            groups: Optional group indices.

        Returns:
            Shape penalty value.
        """
        if groups is None:
            return self._compute_group_penalty(standardized_residuals.flatten())

        loss = 0.0
        for g in groups:
            if standardized_residuals.dim() == 4:
                u_g = standardized_residuals[:, g].flatten()
            else:
                u_g = standardized_residuals[:, g]
            loss = loss + self._compute_group_penalty(u_g)

        return loss / len(groups)

    def _compute_group_penalty(self, u: torch.Tensor) -> torch.Tensor:
        """Compute penalty for a single group."""
        if self.use_robust:
            skew = self._robust_skewness(u)
            kurt = self._robust_kurtosis(u)
        else:
            skew = self._standard_skewness(u)
            kurt = self._standard_kurtosis(u)

        penalty = (skew - self.target_skew) ** 2
        penalty = penalty + self.kurt_weight * (kurt - self.target_kurt) ** 2

        return penalty

    def _standard_skewness(self, x: torch.Tensor) -> torch.Tensor:
        """Standard moment-based skewness."""
        mean = x.mean()
        std = x.std() + 1e-8
        return ((x - mean) / std).pow(3).mean()

    def _standard_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """Standard moment-based kurtosis."""
        mean = x.mean()
        std = x.std() + 1e-8
        return ((x - mean) / std).pow(4).mean()

    def _robust_skewness(self, x: torch.Tensor) -> torch.Tensor:
        """Median-based robust skewness (Bowley skewness)."""
        q1 = torch.quantile(x, 0.25)
        q2 = torch.quantile(x, 0.50)  # median
        q3 = torch.quantile(x, 0.75)

        iqr = q3 - q1 + 1e-8
        return (q3 + q1 - 2 * q2) / iqr

    def _robust_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Robust kurtosis approximation.

        Uses the ratio of interquartile range to interdecile range.
        For Gaussian, this ratio is approximately 0.263.
        """
        q1 = torch.quantile(x, 0.25)
        q3 = torch.quantile(x, 0.75)
        d1 = torch.quantile(x, 0.10)
        d9 = torch.quantile(x, 0.90)

        iqr = q3 - q1 + 1e-8
        idr = d9 - d1 + 1e-8

        # Convert to kurtosis-like scale
        # For Gaussian: ratio ≈ 0.263, kurtosis = 3
        # Map ratio to kurtosis estimate
        ratio = iqr / idr
        # Approximate mapping (empirically calibrated)
        kurt_estimate = 3.0 / (ratio / 0.263 + 1e-8)

        return kurt_estimate.clamp(1, 10)


class TransformRegularization(nn.Module):
    """
    Regularization penalties for the transform.

    Includes:
    - Smoothness: Penalize large second derivatives (curvature)
    - Derivative bounds: Soft penalty for T' outside [min, max]
    - Proximity: Trust region to limit change per iteration

    Args:
        smoothness_weight: Weight for smoothness penalty.
        deriv_min: Minimum allowed derivative.
        deriv_max: Maximum allowed derivative.
        deriv_bound_weight: Weight for derivative bound penalty.
        proximity_weight: Weight for proximity penalty.
    """

    def __init__(
        self,
        smoothness_weight: float = 0.01,
        deriv_min: float = 0.1,
        deriv_max: float = 10.0,
        deriv_bound_weight: float = 1.0,
        proximity_weight: float = 0.1,
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.deriv_min = deriv_min
        self.deriv_max = deriv_max
        self.deriv_bound_weight = deriv_bound_weight
        self.proximity_weight = proximity_weight

        # Store previous parameters for proximity penalty
        self.prev_params: Optional[torch.Tensor] = None

    def forward(
        self,
        transform: nn.Module,
        x_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute regularization loss.

        Args:
            transform: The transform module (must have .derivative() method).
            x_samples: Sample points for evaluating derivatives.

        Returns:
            Total regularization loss.
        """
        loss = torch.tensor(0.0, device=next(transform.parameters()).device)

        if x_samples is not None:
            # Smoothness penalty
            if self.smoothness_weight > 0:
                loss = loss + self.smoothness_weight * self._smoothness_penalty(
                    transform, x_samples
                )

            # Derivative bound penalty
            if self.deriv_bound_weight > 0:
                loss = loss + self.deriv_bound_weight * self._derivative_bound_penalty(
                    transform, x_samples
                )

        # Proximity penalty
        if self.proximity_weight > 0 and self.prev_params is not None:
            loss = loss + self.proximity_weight * self._proximity_penalty(transform)

        return loss

    def _smoothness_penalty(
        self,
        transform: nn.Module,
        x_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize curvature (second derivative) of the transform."""
        # Compute derivative at sample points
        derivs = transform.derivative(x_samples)

        # Approximate second derivative via finite differences
        # Sort x to get proper ordering
        if x_samples.dim() == 1:
            sorted_idx = torch.argsort(x_samples)
            x_sorted = x_samples[sorted_idx]
            derivs_sorted = derivs[sorted_idx]

            # Finite difference of derivatives
            dx = x_sorted[1:] - x_sorted[:-1] + 1e-8
            d2 = (derivs_sorted[1:] - derivs_sorted[:-1]) / dx

            return d2.pow(2).mean()
        else:
            # For multi-feature, average over features
            penalty = 0.0
            for f in range(x_samples.shape[1]):
                x_f = x_samples[:, f]
                d_f = derivs[:, f]

                sorted_idx = torch.argsort(x_f)
                x_sorted = x_f[sorted_idx]
                d_sorted = d_f[sorted_idx]

                dx = x_sorted[1:] - x_sorted[:-1] + 1e-8
                d2 = (d_sorted[1:] - d_sorted[:-1]) / dx
                penalty = penalty + d2.pow(2).mean()

            return penalty / x_samples.shape[1]

    def _derivative_bound_penalty(
        self,
        transform: nn.Module,
        x_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Soft penalty for derivatives outside bounds."""
        derivs = transform.derivative(x_samples)

        below_min = F.relu(self.deriv_min - derivs)
        above_max = F.relu(derivs - self.deriv_max)

        return (below_min.pow(2) + above_max.pow(2)).mean()

    def _proximity_penalty(self, transform: nn.Module) -> torch.Tensor:
        """Penalize distance from previous parameters."""
        curr_params = torch.cat([p.flatten() for p in transform.parameters()])

        if self.prev_params.shape != curr_params.shape:
            return torch.tensor(0.0, device=curr_params.device)

        return (curr_params - self.prev_params).pow(2).sum()

    def update_prev_params(self, transform: nn.Module):
        """Store current parameters for next iteration's proximity penalty."""
        self.prev_params = torch.cat([
            p.detach().clone().flatten() for p in transform.parameters()
        ])

    def reset_prev_params(self):
        """Clear stored parameters."""
        self.prev_params = None


class CombinedTransformLoss(nn.Module):
    """
    Combined loss for transform optimization.

    L_T = λ_homo * L_homo + λ_vf * J[T] + λ_shape * L_shape + λ_reg * L_reg

    Args:
        num_features: Number of features (for initialization).
        lambda_homo: Weight for homoscedasticity loss.
        lambda_vf: Weight for variance flatness.
        lambda_shape: Weight for shape penalty.
        lambda_reg: Weight for regularization.
        basis_degree: Polynomial degree for L_homo.
        vf_bandwidth: Bandwidth for J[T] kernel.
        kurt_weight: Kurtosis weight in shape penalty.
    """

    def __init__(
        self,
        num_features: int = 1,
        lambda_homo: float = 1.0,
        lambda_vf: float = 0.5,
        lambda_shape: float = 0.1,
        lambda_reg: float = 0.01,
        basis_degree: int = 2,
        vf_bandwidth: Union[float, str] = 'auto',
        kurt_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        deriv_min: float = 0.1,
        deriv_max: float = 10.0,
        proximity_weight: float = 0.1,
    ):
        super().__init__()

        self.num_features = num_features
        self.lambda_homo = lambda_homo
        self.lambda_vf = lambda_vf
        self.lambda_shape = lambda_shape
        self.lambda_reg = lambda_reg

        self.homo_loss = HomoscedasticityLoss(basis_degree=basis_degree)
        self.vf_loss = VarianceFlatnessLoss(bandwidth=vf_bandwidth)
        self.shape_loss = ShapePenalty(kurt_weight=kurt_weight)
        self.reg_loss = TransformRegularization(
            smoothness_weight=smoothness_weight,
            deriv_min=deriv_min,
            deriv_max=deriv_max,
            proximity_weight=proximity_weight,
        )

    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        z_hat: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        transform: Optional[nn.Module] = None,
        x_samples: Optional[torch.Tensor] = None,
        log_deriv: Optional[torch.Tensor] = None,
        groups: Optional[List[int]] = None,
    ) -> dict:
        """
        Compute combined transform loss.

        Args:
            z: Transformed data (optional, used to compute residuals if not provided).
            z_hat: Predicted clean signal.
            residuals: Residuals r = z - ẑ (computed if not provided).
            transform: Transform module (for regularization).
            x_samples: Sample points (for regularization).
            log_deriv: Log derivatives from transform (for diagnostics).
            groups: Group indices.

        Returns:
            Dictionary with 'total' and individual loss components.
        """
        # Compute residuals if not provided
        if residuals is None and z is not None and z_hat is not None:
            residuals = z - z_hat

        if z_hat is None or residuals is None:
            raise ValueError("Need either (z, z_hat) or (z_hat, residuals)")

        losses = {}

        # Homoscedasticity loss
        if self.lambda_homo > 0:
            losses['homo'] = self.homo_loss(z_hat, residuals, groups).item()
            homo_tensor = self.homo_loss(z_hat, residuals, groups)
        else:
            losses['homo'] = 0.0
            homo_tensor = torch.tensor(0.0, device=z_hat.device)

        # Variance flatness
        if self.lambda_vf > 0:
            losses['vf'] = self.vf_loss(z_hat, residuals).item()
            vf_tensor = self.vf_loss(z_hat, residuals)
        else:
            losses['vf'] = 0.0
            vf_tensor = torch.tensor(0.0, device=z_hat.device)

        # Shape penalty
        if self.lambda_shape > 0:
            # Standardize residuals
            sigma_hat = (residuals ** 2).mean().sqrt() + 1e-8
            u = residuals / sigma_hat
            losses['shape'] = self.shape_loss(u, groups).item()
            shape_tensor = self.shape_loss(u, groups)
        else:
            losses['shape'] = 0.0
            shape_tensor = torch.tensor(0.0, device=z_hat.device)

        # Regularization
        if self.lambda_reg > 0 and transform is not None:
            losses['reg'] = self.reg_loss(transform, x_samples).item()
            reg_tensor = self.reg_loss(transform, x_samples)
        else:
            losses['reg'] = 0.0
            reg_tensor = torch.tensor(0.0, device=z_hat.device)

        # Total (as tensor for backprop)
        total = (
            self.lambda_homo * homo_tensor +
            self.lambda_vf * vf_tensor +
            self.lambda_shape * shape_tensor +
            self.lambda_reg * reg_tensor
        )

        losses['total'] = total

        return losses

    def update_prev_params(self, transform: nn.Module):
        """Update stored parameters for proximity penalty."""
        self.reg_loss.update_prev_params(transform)


class DenoiserLoss(nn.Module):
    """
    Loss for training the blind-spot denoiser.

    Standard MSE loss (valid because denoiser has blind-spot property).
    Optionally uses Huber/Charbonnier for robustness to outliers.

    Args:
        loss_type: One of 'mse', 'huber', 'charbonnier'.
        huber_delta: Delta parameter for Huber loss.
        charbonnier_eps: Epsilon for Charbonnier loss.
        reduction: Reduction method ('mean' or 'sum').
        robust: If True, use Huber loss instead of MSE.
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        huber_delta: float = 1.0,
        charbonnier_eps: float = 1e-3,
        reduction: str = 'mean',
        robust: bool = False,
    ):
        super().__init__()
        if robust:
            loss_type = 'huber'
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.charbonnier_eps = charbonnier_eps
        self.reduction = reduction

    def forward(
        self,
        z: torch.Tensor,
        z_hat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute denoiser loss.

        Args:
            z: Target (noisy transformed data).
            z_hat: Denoiser prediction.
            mask: Optional mask for valid positions.

        Returns:
            (loss, components_dict) tuple.
        """
        diff = z_hat - z

        if mask is not None:
            diff = diff * mask

        if self.loss_type == 'mse':
            loss_per_elem = diff.pow(2)
        elif self.loss_type == 'huber':
            loss_per_elem = F.huber_loss(z_hat, z, reduction='none', delta=self.huber_delta)
        elif self.loss_type == 'charbonnier':
            loss_per_elem = torch.sqrt(diff.pow(2) + self.charbonnier_eps ** 2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if mask is not None:
            loss = loss_per_elem.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_elem.mean()

        # Compute additional metrics
        with torch.no_grad():
            mse = diff.pow(2).mean().item()
            mae = diff.abs().mean().item()

        components = {
            'mse': mse,
            'mae': mae,
            'loss': loss.item(),
        }

        return loss, components
