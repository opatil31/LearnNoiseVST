"""
Rational Quadratic Spline (RQS) implementation.

Based on Neural Spline Flows (Durkan et al., 2019).
Provides monotonic, invertible, differentiable transforms with analytic inverse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class RationalQuadraticSpline(nn.Module):
    """
    Monotonic Rational Quadratic Spline transform.

    The spline is defined over [-bound, bound] with K bins.
    Outside this region, linear tails are used.

    Parameters:
        num_bins: Number of bins (K). More bins = more flexibility.
        bound: The spline is defined on [-bound, bound].
        min_derivative: Minimum derivative value (for numerical stability).
        min_bin_width: Minimum bin width (fraction of total width).
        min_bin_height: Minimum bin height (fraction of total height).

    Learnable parameters:
        - unnorm_widths: [K] unnormalized bin widths
        - unnorm_heights: [K] unnormalized bin heights
        - unnorm_derivatives: [K+1] unnormalized derivatives at knots
    """

    def __init__(
        self,
        num_bins: int = 16,
        bound: float = 5.0,
        min_derivative: float = 1e-3,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.bound = bound
        self.min_derivative = min_derivative
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        # Learnable parameters (initialized near identity)
        # Widths and heights: softmax will normalize to sum to 2*bound
        self.unnorm_widths = nn.Parameter(torch.zeros(num_bins))
        self.unnorm_heights = nn.Parameter(torch.zeros(num_bins))

        # Derivatives at knots: softplus ensures positivity
        # Initialize to give derivative ≈ 1 (identity-like)
        self.unnorm_derivatives = nn.Parameter(torch.zeros(num_bins + 1))

    def _compute_spline_params(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute constrained spline parameters from unconstrained learnable params.

        Returns:
            widths: [K] bin widths (sum to 2*bound)
            heights: [K] bin heights (sum to 2*bound)
            derivatives: [K+1] positive derivatives at knots
            knot_x: [K+1] x-coordinates of knots
            knot_y: [K+1] y-coordinates of knots
        """
        # Widths: softmax ensures positive and sums to 1, then scale to 2*bound
        # Add min_bin_width to ensure no bin is too small
        widths = F.softmax(self.unnorm_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths
        widths = widths * 2 * self.bound

        # Heights: same treatment
        heights = F.softmax(self.unnorm_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        heights = heights * 2 * self.bound

        # Derivatives: softplus ensures positive, add min_derivative
        derivatives = F.softplus(self.unnorm_derivatives) + self.min_derivative

        # Knot positions: cumulative sum starting from -bound
        knot_x = torch.cat([
            torch.tensor([-self.bound], device=widths.device, dtype=widths.dtype),
            -self.bound + torch.cumsum(widths, dim=-1)
        ])
        knot_y = torch.cat([
            torch.tensor([-self.bound], device=heights.device, dtype=heights.dtype),
            -self.bound + torch.cumsum(heights, dim=-1)
        ])

        return widths, heights, derivatives, knot_x, knot_y

    def forward(
        self, x: torch.Tensor, return_log_deriv: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the spline transform.

        Args:
            x: Input tensor of any shape.
            return_log_deriv: If True, also return log|T'(x)|.

        Returns:
            y: Transformed values, same shape as x.
            log_deriv: Log derivative (if return_log_deriv=True), else None.
        """
        widths, heights, derivatives, knot_x, knot_y = self._compute_spline_params()

        # Flatten for processing
        x_flat = x.flatten()
        y_flat = torch.zeros_like(x_flat)
        log_deriv_flat = torch.zeros_like(x_flat) if return_log_deriv else None

        # Handle three regions: left tail, spline, right tail
        left_mask = x_flat < -self.bound
        right_mask = x_flat > self.bound
        interior_mask = ~(left_mask | right_mask)

        # Left linear tail: y = d_0 * (x - x_0) + y_0
        if left_mask.any():
            d_left = derivatives[0]
            y_flat[left_mask] = knot_y[0] + d_left * (x_flat[left_mask] - knot_x[0])
            if return_log_deriv:
                log_deriv_flat[left_mask] = torch.log(d_left)

        # Right linear tail: y = d_K * (x - x_K) + y_K
        if right_mask.any():
            d_right = derivatives[-1]
            y_flat[right_mask] = knot_y[-1] + d_right * (x_flat[right_mask] - knot_x[-1])
            if return_log_deriv:
                log_deriv_flat[right_mask] = torch.log(d_right)

        # Interior: rational quadratic spline
        if interior_mask.any():
            x_interior = x_flat[interior_mask]
            y_interior, ld_interior = self._forward_spline(
                x_interior, widths, heights, derivatives, knot_x, knot_y,
                compute_log_deriv=return_log_deriv
            )
            y_flat[interior_mask] = y_interior
            if return_log_deriv:
                log_deriv_flat[interior_mask] = ld_interior

        y = y_flat.view_as(x)
        log_deriv = log_deriv_flat.view_as(x) if return_log_deriv else None

        return y, log_deriv

    def _forward_spline(
        self,
        x: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivatives: torch.Tensor,
        knot_x: torch.Tensor,
        knot_y: torch.Tensor,
        compute_log_deriv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate the spline in the interior region.

        Uses the rational quadratic formula from Durkan et al. (2019).
        """
        # Find which bin each x belongs to
        # bin_idx[i] = k means x[i] is in bin k (between knot_x[k] and knot_x[k+1])
        bin_idx = torch.searchsorted(knot_x[1:], x).clamp(0, self.num_bins - 1)

        # Gather bin parameters
        w_k = widths[bin_idx]  # width of bin
        h_k = heights[bin_idx]  # height of bin
        d_k = derivatives[bin_idx]  # derivative at left knot
        d_k1 = derivatives[bin_idx + 1]  # derivative at right knot
        x_k = knot_x[bin_idx]  # x-coord of left knot
        y_k = knot_y[bin_idx]  # y-coord of left knot

        # Normalized position within bin: xi in [0, 1]
        xi = (x - x_k) / w_k

        # Slope of the bin (secant)
        s_k = h_k / w_k

        # Rational quadratic coefficients
        # y(xi) = y_k + h_k * [s_k * xi^2 + d_k * xi * (1 - xi)] / [s_k + (d_k + d_k1 - 2*s_k) * xi * (1 - xi)]
        xi_1mxi = xi * (1 - xi)
        numerator = s_k * xi.pow(2) + d_k * xi_1mxi
        denominator = s_k + (d_k + d_k1 - 2 * s_k) * xi_1mxi

        y = y_k + h_k * numerator / denominator

        if compute_log_deriv:
            # Derivative formula (from the paper):
            # dy/dx = (s_k^2 * (d_k1 * xi^2 + 2*s_k*xi*(1-xi) + d_k*(1-xi)^2)) / denominator^2
            deriv_numerator = s_k.pow(2) * (
                d_k1 * xi.pow(2) + 2 * s_k * xi_1mxi + d_k * (1 - xi).pow(2)
            )
            deriv = deriv_numerator / denominator.pow(2)
            log_deriv = torch.log(deriv.clamp(min=1e-8))
        else:
            log_deriv = None

        return y, log_deriv

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse spline transform.

        Args:
            y: Transformed values.

        Returns:
            x: Original values.
        """
        widths, heights, derivatives, knot_x, knot_y = self._compute_spline_params()

        y_flat = y.flatten()
        x_flat = torch.zeros_like(y_flat)

        # Handle three regions
        left_mask = y_flat < -self.bound
        right_mask = y_flat > self.bound
        interior_mask = ~(left_mask | right_mask)

        # Left linear tail inverse
        if left_mask.any():
            d_left = derivatives[0]
            x_flat[left_mask] = knot_x[0] + (y_flat[left_mask] - knot_y[0]) / d_left

        # Right linear tail inverse
        if right_mask.any():
            d_right = derivatives[-1]
            x_flat[right_mask] = knot_x[-1] + (y_flat[right_mask] - knot_y[-1]) / d_right

        # Interior: inverse rational quadratic
        if interior_mask.any():
            y_interior = y_flat[interior_mask]
            x_interior = self._inverse_spline(
                y_interior, widths, heights, derivatives, knot_x, knot_y
            )
            x_flat[interior_mask] = x_interior

        return x_flat.view_as(y)

    def _inverse_spline(
        self,
        y: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivatives: torch.Tensor,
        knot_x: torch.Tensor,
        knot_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inverse of the rational quadratic spline in the interior.

        Solves a quadratic equation to find xi, then recovers x.
        """
        # Find which bin each y belongs to
        bin_idx = torch.searchsorted(knot_y[1:], y).clamp(0, self.num_bins - 1)

        # Gather bin parameters
        w_k = widths[bin_idx]
        h_k = heights[bin_idx]
        d_k = derivatives[bin_idx]
        d_k1 = derivatives[bin_idx + 1]
        x_k = knot_x[bin_idx]
        y_k = knot_y[bin_idx]

        # Slope
        s_k = h_k / w_k

        # Normalized y within bin
        y_norm = (y - y_k) / h_k

        # Coefficients for quadratic in xi:
        # From the forward formula, rearranging:
        # y_norm = [s_k * xi^2 + d_k * xi * (1-xi)] / [s_k + (d_k + d_k1 - 2*s_k) * xi * (1-xi)]
        # Let delta = d_k + d_k1 - 2*s_k
        # y_norm * (s_k + delta * xi * (1-xi)) = s_k * xi^2 + d_k * xi - d_k * xi^2
        # y_norm * s_k + y_norm * delta * xi - y_norm * delta * xi^2 = s_k * xi^2 + d_k * xi - d_k * xi^2
        # Rearranging to: a * xi^2 + b * xi + c = 0
        delta = d_k + d_k1 - 2 * s_k

        a = h_k * (s_k - y_norm * delta + y_norm * d_k - d_k) + 1e-8  # Add eps for stability
        # Simplify: a = h_k * (s_k - d_k + y_norm * (d_k - delta))
        #            = h_k * (s_k - d_k) + h_k * y_norm * (d_k - d_k - d_k1 + 2*s_k)
        #            = h_k * (s_k - d_k) + h_k * y_norm * (2*s_k - d_k1 - d_k + d_k)
        # This gets complex; let's use the standard form from the paper.

        # Using the formula from Durkan et al.:
        # a = h_k * (s_k - d_k) + (y - y_k) * (d_k + d_k1 - 2*s_k)
        # b = h_k * d_k - (y - y_k) * (d_k + d_k1 - 2*s_k)
        # c = -s_k * (y - y_k)

        y_diff = y - y_k
        a = h_k * (s_k - d_k) + y_diff * delta
        b = h_k * d_k - y_diff * delta
        c = -s_k * y_diff

        # Solve quadratic: xi = (-b + sqrt(b^2 - 4ac)) / (2a)
        # Use numerically stable form
        discriminant = b.pow(2) - 4 * a * c
        discriminant = discriminant.clamp(min=0)  # Numerical safety

        # Standard quadratic formula (taking the + root for monotonicity)
        xi = (-b + torch.sqrt(discriminant)) / (2 * a + 1e-8)

        # Clamp xi to [0, 1] for safety
        xi = xi.clamp(0, 1)

        # Recover x
        x = x_k + w_k * xi

        return x

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute T'(x), the derivative of the transform.

        Args:
            x: Input tensor.

        Returns:
            deriv: T'(x), same shape as x.
        """
        _, log_deriv = self.forward(x, return_log_deriv=True)
        return torch.exp(log_deriv)

    def log_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log|T'(x)|.

        Args:
            x: Input tensor.

        Returns:
            log_deriv: log|T'(x)|, same shape as x.
        """
        _, log_deriv = self.forward(x, return_log_deriv=True)
        return log_deriv

    def set_to_identity(self):
        """Reset parameters to approximate identity transform."""
        nn.init.zeros_(self.unnorm_widths)
        nn.init.zeros_(self.unnorm_heights)
        nn.init.zeros_(self.unnorm_derivatives)

    def get_knots(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current knot positions for visualization."""
        _, _, _, knot_x, knot_y = self._compute_spline_params()
        return knot_x.detach(), knot_y.detach()


class RQSBatch(nn.Module):
    """
    Batch of independent RQS transforms, one per feature.

    More efficient than a list of RQS modules when num_features is large.
    All features share the same num_bins and bound, but have independent parameters.

    Parameters:
        num_features: Number of features (d).
        num_bins: Number of bins per feature.
        bound: Spline domain [-bound, bound].
        min_derivative: Minimum derivative value.
    """

    def __init__(
        self,
        num_features: int,
        num_bins: int = 16,
        bound: float = 5.0,
        min_derivative: float = 1e-3,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.bound = bound
        self.min_derivative = min_derivative
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        # Learnable parameters: [num_features, num_bins] or [num_features, num_bins+1]
        self.unnorm_widths = nn.Parameter(torch.zeros(num_features, num_bins))
        self.unnorm_heights = nn.Parameter(torch.zeros(num_features, num_bins))
        self.unnorm_derivatives = nn.Parameter(torch.zeros(num_features, num_bins + 1))

    def _compute_spline_params(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute constrained parameters for all features.

        Returns tensors of shape [num_features, ...].
        """
        # Widths: [num_features, num_bins]
        widths = F.softmax(self.unnorm_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths
        widths = widths * 2 * self.bound

        # Heights: [num_features, num_bins]
        heights = F.softmax(self.unnorm_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        heights = heights * 2 * self.bound

        # Derivatives: [num_features, num_bins+1]
        derivatives = F.softplus(self.unnorm_derivatives) + self.min_derivative

        # Knot positions: [num_features, num_bins+1]
        device = widths.device
        dtype = widths.dtype

        # Cumsum for knot positions
        cum_widths = torch.cumsum(widths, dim=-1)
        knot_x = torch.cat([
            torch.full((self.num_features, 1), -self.bound, device=device, dtype=dtype),
            -self.bound + cum_widths
        ], dim=-1)

        cum_heights = torch.cumsum(heights, dim=-1)
        knot_y = torch.cat([
            torch.full((self.num_features, 1), -self.bound, device=device, dtype=dtype),
            -self.bound + cum_heights
        ], dim=-1)

        return widths, heights, derivatives, knot_x, knot_y

    def forward(
        self, x: torch.Tensor, return_log_deriv: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply per-feature spline transforms.

        Args:
            x: Input tensor of shape [batch, num_features] or [batch, num_features, ...].
            return_log_deriv: If True, also return log|T'(x)|.

        Returns:
            y: Transformed values.
            log_deriv: Log derivative (if requested).
        """
        # For now, assume x is [batch, num_features]
        # TODO: Support higher-dimensional inputs (images)
        assert x.dim() >= 2 and x.shape[1] == self.num_features, \
            f"Expected x.shape[1] == {self.num_features}, got {x.shape}"

        batch_size = x.shape[0]
        extra_dims = x.shape[2:] if x.dim() > 2 else ()

        # Flatten extra dimensions if present
        if extra_dims:
            x_flat = x.view(batch_size, self.num_features, -1)  # [B, F, ...]
            num_extra = x_flat.shape[2]
        else:
            x_flat = x  # [B, F]
            num_extra = 1

        widths, heights, derivatives, knot_x, knot_y = self._compute_spline_params()

        y = torch.zeros_like(x_flat)
        log_deriv = torch.zeros_like(x_flat) if return_log_deriv else None

        # Process each feature
        for f in range(self.num_features):
            x_f = x_flat[:, f]  # [B] or [B, num_extra]
            y_f, ld_f = self._forward_single_feature(
                x_f,
                widths[f], heights[f], derivatives[f],
                knot_x[f], knot_y[f],
                compute_log_deriv=return_log_deriv
            )
            y[:, f] = y_f
            if return_log_deriv:
                log_deriv[:, f] = ld_f

        # Reshape back
        if extra_dims:
            y = y.view(batch_size, self.num_features, *extra_dims)
            if return_log_deriv:
                log_deriv = log_deriv.view(batch_size, self.num_features, *extra_dims)

        return y, log_deriv

    def _forward_single_feature(
        self,
        x: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivatives: torch.Tensor,
        knot_x: torch.Tensor,
        knot_y: torch.Tensor,
        compute_log_deriv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for a single feature (vectorized over batch)."""
        x_flat = x.flatten()
        y_flat = torch.zeros_like(x_flat)
        log_deriv_flat = torch.zeros_like(x_flat) if compute_log_deriv else None

        # Masks
        left_mask = x_flat < -self.bound
        right_mask = x_flat > self.bound
        interior_mask = ~(left_mask | right_mask)

        # Left tail
        if left_mask.any():
            d_left = derivatives[0]
            y_flat[left_mask] = knot_y[0] + d_left * (x_flat[left_mask] - knot_x[0])
            if compute_log_deriv:
                log_deriv_flat[left_mask] = torch.log(d_left)

        # Right tail
        if right_mask.any():
            d_right = derivatives[-1]
            y_flat[right_mask] = knot_y[-1] + d_right * (x_flat[right_mask] - knot_x[-1])
            if compute_log_deriv:
                log_deriv_flat[right_mask] = torch.log(d_right)

        # Interior
        if interior_mask.any():
            x_int = x_flat[interior_mask]

            # Find bins
            bin_idx = torch.searchsorted(knot_x[1:], x_int).clamp(0, self.num_bins - 1)

            w_k = widths[bin_idx]
            h_k = heights[bin_idx]
            d_k = derivatives[bin_idx]
            d_k1 = derivatives[bin_idx + 1]
            x_k = knot_x[bin_idx]
            y_k = knot_y[bin_idx]

            xi = (x_int - x_k) / w_k
            s_k = h_k / w_k

            xi_1mxi = xi * (1 - xi)
            numerator = s_k * xi.pow(2) + d_k * xi_1mxi
            denominator = s_k + (d_k + d_k1 - 2 * s_k) * xi_1mxi

            y_flat[interior_mask] = y_k + h_k * numerator / denominator

            if compute_log_deriv:
                deriv_num = s_k.pow(2) * (
                    d_k1 * xi.pow(2) + 2 * s_k * xi_1mxi + d_k * (1 - xi).pow(2)
                )
                deriv = deriv_num / denominator.pow(2)
                log_deriv_flat[interior_mask] = torch.log(deriv.clamp(min=1e-8))

        y = y_flat.view_as(x)
        log_deriv = log_deriv_flat.view_as(x) if compute_log_deriv else None

        return y, log_deriv

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transform for all features."""
        assert y.dim() >= 2 and y.shape[1] == self.num_features

        batch_size = y.shape[0]
        extra_dims = y.shape[2:] if y.dim() > 2 else ()

        if extra_dims:
            y_flat = y.view(batch_size, self.num_features, -1)
        else:
            y_flat = y

        widths, heights, derivatives, knot_x, knot_y = self._compute_spline_params()

        x = torch.zeros_like(y_flat)

        for f in range(self.num_features):
            y_f = y_flat[:, f]
            x_f = self._inverse_single_feature(
                y_f,
                widths[f], heights[f], derivatives[f],
                knot_x[f], knot_y[f]
            )
            x[:, f] = x_f

        if extra_dims:
            x = x.view(batch_size, self.num_features, *extra_dims)

        return x

    def _inverse_single_feature(
        self,
        y: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivatives: torch.Tensor,
        knot_x: torch.Tensor,
        knot_y: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse for a single feature."""
        y_flat = y.flatten()
        x_flat = torch.zeros_like(y_flat)

        left_mask = y_flat < -self.bound
        right_mask = y_flat > self.bound
        interior_mask = ~(left_mask | right_mask)

        # Left tail
        if left_mask.any():
            d_left = derivatives[0]
            x_flat[left_mask] = knot_x[0] + (y_flat[left_mask] - knot_y[0]) / d_left

        # Right tail
        if right_mask.any():
            d_right = derivatives[-1]
            x_flat[right_mask] = knot_x[-1] + (y_flat[right_mask] - knot_y[-1]) / d_right

        # Interior
        if interior_mask.any():
            y_int = y_flat[interior_mask]

            bin_idx = torch.searchsorted(knot_y[1:], y_int).clamp(0, self.num_bins - 1)

            w_k = widths[bin_idx]
            h_k = heights[bin_idx]
            d_k = derivatives[bin_idx]
            d_k1 = derivatives[bin_idx + 1]
            x_k = knot_x[bin_idx]
            y_k = knot_y[bin_idx]

            s_k = h_k / w_k
            delta = d_k + d_k1 - 2 * s_k
            y_diff = y_int - y_k

            a = h_k * (s_k - d_k) + y_diff * delta
            b = h_k * d_k - y_diff * delta
            c = -s_k * y_diff

            discriminant = (b.pow(2) - 4 * a * c).clamp(min=0)
            xi = (-b + torch.sqrt(discriminant)) / (2 * a + 1e-8)
            xi = xi.clamp(0, 1)

            x_flat[interior_mask] = x_k + w_k * xi

        return x_flat.view_as(y)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute T'(x) for all features."""
        _, log_deriv = self.forward(x, return_log_deriv=True)
        return torch.exp(log_deriv)

    def set_to_identity(self):
        """Reset all features to identity-like transforms."""
        nn.init.zeros_(self.unnorm_widths)
        nn.init.zeros_(self.unnorm_heights)
        nn.init.zeros_(self.unnorm_derivatives)
