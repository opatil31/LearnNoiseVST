"""
Monotone Feature Transform with Gauge-Fixing.

The main transform module that applies per-feature RQS transforms
followed by standardization (gauge-fixing) to ensure:
- E[z_f] = 0
- Var(z_f) = 1

This prevents the transform from "cheating" by shrinking the output
to reduce apparent noise variance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Literal
from .rqs import RationalQuadraticSpline, RQSBatch
from .utils import estimate_per_feature_bounds


class MonotoneFeatureTransform(nn.Module):
    """
    Per-feature monotone transform with gauge-fixing.

    For each feature f:
        z_f = (RQS_f(x_f) - μ_f) / σ_f

    where μ_f and σ_f are running statistics that ensure standardization.

    Parameters:
        num_features: Number of features (d).
        num_bins: Number of RQS bins per feature.
        bound: RQS spline domain [-bound, bound].
        tail_quantiles: Quantiles for determining linear tail region.
        momentum: Momentum for running statistics updates.
        track_running_stats: Whether to track running mean/var.
        affine: Whether to include learnable affine params after standardization.
        min_derivative: Minimum derivative (numerical stability).
        use_batch_rqs: If True, use efficient batched RQS (recommended for large d).
    """

    def __init__(
        self,
        num_features: int,
        num_bins: int = 16,
        bound: float = 5.0,
        tail_quantiles: Tuple[float, float] = (0.001, 0.999),
        momentum: float = 0.1,
        track_running_stats: bool = True,
        affine: bool = False,
        min_derivative: float = 1e-3,
        use_batch_rqs: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_bins = num_bins
        self.bound = bound
        self.tail_quantiles = tail_quantiles
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.affine = affine
        self.use_batch_rqs = use_batch_rqs

        # Per-feature RQS transforms
        if use_batch_rqs:
            self.rqs = RQSBatch(
                num_features=num_features,
                num_bins=num_bins,
                bound=bound,
                min_derivative=min_derivative,
            )
        else:
            self.rqs = nn.ModuleList([
                RationalQuadraticSpline(
                    num_bins=num_bins,
                    bound=bound,
                    min_derivative=min_derivative,
                )
                for _ in range(num_features)
            ])

        # Running statistics for gauge-fixing
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        # Optional learnable affine parameters (post-standardization)
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Per-feature input normalization (for mapping data to spline domain)
        self.register_buffer('input_shift', torch.zeros(num_features))
        self.register_buffer('input_scale', torch.ones(num_features))
        self._input_normalization_set = False

    def set_input_normalization(self, data: torch.Tensor):
        """
        Set input normalization to map data to spline domain [-bound, bound].

        Should be called once on representative training data before training.

        Args:
            data: Representative data of shape [N, num_features, ...].
        """
        with torch.no_grad():
            lower, upper = estimate_per_feature_bounds(
                data, quantiles=self.tail_quantiles, margin=0.1
            )

            # Map [lower, upper] -> [-bound, bound]
            data_range = upper - lower
            data_range = torch.clamp(data_range, min=1e-8)

            self.input_shift = (lower + upper) / 2
            self.input_scale = data_range / (2 * self.bound)

            self._input_normalization_set = True

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to spline domain."""
        # x: [B, F, ...], shift/scale: [F]
        # Need to broadcast correctly
        shape = [1, self.num_features] + [1] * (x.dim() - 2)
        shift = self.input_shift.view(*shape)
        scale = self.input_scale.view(*shape)
        return (x - shift) / scale

    def _denormalize_input(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of _normalize_input."""
        shape = [1, self.num_features] + [1] * (x_norm.dim() - 2)
        shift = self.input_shift.view(*shape)
        scale = self.input_scale.view(*shape)
        return x_norm * scale + shift

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        return_prenorm: bool = False,
        return_log_deriv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the transform: x -> z.

        Args:
            x: Input tensor of shape [batch, num_features] or [batch, num_features, ...].
            update_stats: Whether to update running mean/var (during training).
            return_prenorm: If True, also return pre-normalized spline output.
            return_log_deriv: If True, also return log|T'(x)|.

        Returns:
            z: Transformed and standardized output.
            s: (optional) Pre-normalized spline output.
            log_deriv: (optional) Log derivative.
        """
        # Normalize input to spline domain
        x_norm = self._normalize_input(x)

        # Apply RQS
        if self.use_batch_rqs:
            s, log_deriv_rqs = self.rqs(x_norm, return_log_deriv=return_log_deriv)
        else:
            # Apply per-feature splines
            s_list = []
            log_deriv_list = [] if return_log_deriv else None

            for f in range(self.num_features):
                if x.dim() == 2:
                    x_f = x_norm[:, f]
                else:
                    x_f = x_norm[:, f, ...]

                s_f, ld_f = self.rqs[f](x_f, return_log_deriv=return_log_deriv)
                s_list.append(s_f)
                if return_log_deriv:
                    log_deriv_list.append(ld_f)

            s = torch.stack(s_list, dim=1)
            log_deriv_rqs = torch.stack(log_deriv_list, dim=1) if return_log_deriv else None

        # Gauge-fixing: standardize to mean=0, var=1
        z, log_deriv_gauge = self._gauge_fix(s, update_stats=update_stats)

        # Optional affine
        if self.affine:
            shape = [1, self.num_features] + [1] * (z.dim() - 2)
            z = z * self.weight.view(*shape) + self.bias.view(*shape)

        # Combine log derivatives
        if return_log_deriv:
            # Total log deriv = log(d_input_norm) + log_deriv_rqs + log_deriv_gauge
            # d_input_norm = 1/input_scale (per feature)
            shape = [1, self.num_features] + [1] * (x.dim() - 2)
            log_deriv_input = -torch.log(self.input_scale).view(*shape)
            log_deriv = log_deriv_input + log_deriv_rqs + log_deriv_gauge
            if self.affine:
                log_deriv = log_deriv + torch.log(self.weight.abs()).view(*shape)
        else:
            log_deriv = None

        # Build return tuple
        result = [z]
        if return_prenorm:
            result.append(s)
        if return_log_deriv:
            result.append(log_deriv)

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def _gauge_fix(
        self,
        s: torch.Tensor,
        update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gauge-fixing (standardization).

        Args:
            s: Pre-normalized spline output [B, F, ...].
            update_stats: Whether to update running statistics.

        Returns:
            z: Standardized output.
            log_deriv_gauge: Log derivative from standardization (= -0.5 * log(var)).
        """
        if not self.track_running_stats:
            # Use batch statistics
            mean, var = self._compute_batch_stats(s)
        else:
            if self.training and update_stats:
                # Compute batch statistics
                batch_mean, batch_var = self._compute_batch_stats(s)

                # Update running statistics with momentum
                with torch.no_grad():
                    self.running_mean = (
                        (1 - self.momentum) * self.running_mean +
                        self.momentum * batch_mean
                    )
                    self.running_var = (
                        (1 - self.momentum) * self.running_var +
                        self.momentum * batch_var
                    )
                    self.num_batches_tracked += 1

            # Use running statistics for standardization
            mean = self.running_mean
            var = self.running_var

        # Standardize
        shape = [1, self.num_features] + [1] * (s.dim() - 2)
        mean = mean.view(*shape)
        var = var.view(*shape)

        std = torch.sqrt(var + 1e-8)
        z = (s - mean) / std

        # Log derivative from standardization: d/ds[(s-μ)/σ] = 1/σ
        log_deriv_gauge = -0.5 * torch.log(var + 1e-8)  # = -log(σ)
        log_deriv_gauge = log_deriv_gauge.expand_as(z)

        return z, log_deriv_gauge

    def _compute_batch_stats(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance per feature from batch."""
        # s: [B, F, ...]
        # Compute over all dimensions except feature dimension
        dims_to_reduce = [0] + list(range(2, s.dim()))

        mean = s.mean(dim=dims_to_reduce)
        var = s.var(dim=dims_to_reduce, unbiased=False)

        return mean, var

    def forward_prenorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only pre-normalized spline output.

        Useful for computing gauge-fixing statistics.
        """
        x_norm = self._normalize_input(x)

        if self.use_batch_rqs:
            s, _ = self.rqs(x_norm, return_log_deriv=False)
        else:
            s_list = []
            for f in range(self.num_features):
                if x.dim() == 2:
                    x_f = x_norm[:, f]
                else:
                    x_f = x_norm[:, f, ...]
                s_f, _ = self.rqs[f](x_f, return_log_deriv=False)
                s_list.append(s_f)
            s = torch.stack(s_list, dim=1)

        return s

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse transform: z -> x.

        Args:
            z: Transformed values.

        Returns:
            x: Original values.
        """
        # Undo affine
        if self.affine:
            shape = [1, self.num_features] + [1] * (z.dim() - 2)
            z = (z - self.bias.view(*shape)) / self.weight.view(*shape)

        # Undo gauge-fixing (de-standardize)
        shape = [1, self.num_features] + [1] * (z.dim() - 2)

        if self.track_running_stats:
            mean = self.running_mean.view(*shape)
            var = self.running_var.view(*shape)
        else:
            raise RuntimeError(
                "Cannot invert without running stats. "
                "Set track_running_stats=True or call refresh_stats() first."
            )

        std = torch.sqrt(var + 1e-8)
        s = z * std + mean

        # Invert RQS
        if self.use_batch_rqs:
            x_norm = self.rqs.inverse(s)
        else:
            x_norm_list = []
            for f in range(self.num_features):
                if s.dim() == 2:
                    s_f = s[:, f]
                else:
                    s_f = s[:, f, ...]
                x_norm_f = self.rqs[f].inverse(s_f)
                x_norm_list.append(x_norm_f)
            x_norm = torch.stack(x_norm_list, dim=1)

        # Denormalize input
        x = self._denormalize_input(x_norm)

        return x

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute T'(x), the derivative of the transform.

        Args:
            x: Input values.

        Returns:
            deriv: T'(x).
        """
        _, log_deriv = self.forward(x, update_stats=False, return_log_deriv=True)
        return torch.exp(log_deriv)

    def log_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log|T'(x)|."""
        _, log_deriv = self.forward(x, update_stats=False, return_log_deriv=True)
        return log_deriv

    def refresh_stats(self, dataloader, device: Optional[torch.device] = None):
        """
        Refresh running statistics using full pass over data.

        Should be called periodically during training or after training.

        Args:
            dataloader: DataLoader yielding batches of shape [B, F, ...].
            device: Device to use.
        """
        self.eval()

        all_s = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch

                if device is not None:
                    x = x.to(device)

                s = self.forward_prenorm(x)
                all_s.append(s.cpu())

        all_s = torch.cat(all_s, dim=0)

        # Compute statistics
        dims_to_reduce = [0] + list(range(2, all_s.dim()))
        self.running_mean = all_s.mean(dim=dims_to_reduce).to(self.running_mean.device)
        self.running_var = all_s.var(dim=dims_to_reduce, unbiased=False).to(self.running_var.device)

        self.train()

    def set_to_identity(self):
        """Reset transform to approximate identity."""
        if self.use_batch_rqs:
            self.rqs.set_to_identity()
        else:
            for rqs in self.rqs:
                rqs.set_to_identity()

        # Reset affine
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        # Reset stats
        if self.track_running_stats:
            nn.init.zeros_(self.running_mean)
            nn.init.ones_(self.running_var)
            self.num_batches_tracked.zero_()

    def get_num_parameters(self) -> int:
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"num_bins={self.num_bins}, "
            f"bound={self.bound}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )


class ImageMonotoneTransform(MonotoneFeatureTransform):
    """
    MonotoneFeatureTransform specialized for images.

    Treats each channel as a separate feature.
    Input shape: [B, C, H, W]
    """

    def __init__(
        self,
        num_channels: int,
        num_bins: int = 16,
        bound: float = 5.0,
        **kwargs
    ):
        super().__init__(
            num_features=num_channels,
            num_bins=num_bins,
            bound=bound,
            **kwargs
        )
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for images.

        Args:
            x: Image tensor [B, C, H, W].

        Returns:
            z: Transformed image [B, C, H, W].
        """
        assert x.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {x.dim()}D"
        assert x.shape[1] == self.num_channels, \
            f"Expected {self.num_channels} channels, got {x.shape[1]}"

        return super().forward(x, **kwargs)
