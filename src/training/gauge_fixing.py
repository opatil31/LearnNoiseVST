"""
Gauge-fixing utilities for the transform.

Gauge-fixing ensures that the transform output is standardized:
    E[z_f] = 0, Var(z_f) = 1

This prevents the transform from "cheating" by shrinking the output
to artificially reduce noise variance.

This module provides utilities for:
1. Computing running statistics (EMA or batch-based)
2. Periodic full-pass recomputation
3. Proper handling during training vs inference
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Iterator
from torch.utils.data import DataLoader
import math


class RunningStats(nn.Module):
    """
    Running statistics tracker with exponential moving average.

    Tracks mean and variance using EMA updates during training,
    with option for periodic full recomputation.

    Args:
        num_features: Number of features to track.
        momentum: EMA momentum (1 - decay). Higher = faster adaptation.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Running statistics (not parameters, but saved with model)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # For Welford's online algorithm (more stable)
        self.register_buffer('_welford_mean', torch.zeros(num_features))
        self.register_buffer('_welford_m2', torch.zeros(num_features))
        self.register_buffer('_welford_count', torch.tensor(0, dtype=torch.long))

    def reset(self):
        """Reset running statistics to defaults."""
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_batches_tracked.zero_()
        self._welford_mean.zero_()
        self._welford_m2.zero_()
        self._welford_count.zero_()

    def update_ema(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
    ):
        """
        Update running statistics using EMA.

        Args:
            batch_mean: Mean computed from current batch [num_features].
            batch_var: Variance computed from current batch [num_features].
        """
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

    def update_welford(self, x: torch.Tensor):
        """
        Update using Welford's online algorithm (numerically stable).

        Args:
            x: Batch of data [B, num_features, ...].
        """
        with torch.no_grad():
            # Flatten spatial dimensions if present
            if x.dim() > 2:
                x = x.transpose(1, -1).reshape(-1, self.num_features)
            else:
                x = x.view(-1, self.num_features)

            n = x.shape[0]

            for i in range(n):
                self._welford_count += 1
                delta = x[i] - self._welford_mean
                self._welford_mean = self._welford_mean + delta / self._welford_count
                delta2 = x[i] - self._welford_mean
                self._welford_m2 = self._welford_m2 + delta * delta2

    def finalize_welford(self):
        """Convert Welford accumulators to mean and variance."""
        with torch.no_grad():
            if self._welford_count > 1:
                self.running_mean = self._welford_mean.clone()
                self.running_var = self._welford_m2 / (self._welford_count - 1)
                self.running_var = self.running_var.clamp(min=self.eps)

            # Reset Welford accumulators
            self._welford_mean.zero_()
            self._welford_m2.zero_()
            self._welford_count.zero_()

    def get_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current running mean and standard deviation."""
        std = torch.sqrt(self.running_var + self.eps)
        return self.running_mean, std

    def standardize(
        self,
        x: torch.Tensor,
        return_log_deriv: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standardize input using running statistics.

        Args:
            x: Input tensor [B, num_features, ...].
            return_log_deriv: If True, return log derivative of standardization.

        Returns:
            z: Standardized output.
            log_deriv: Log derivative (if requested).
        """
        mean, std = self.get_stats()

        # Broadcast to match input shape
        shape = [1, self.num_features] + [1] * (x.dim() - 2)
        mean = mean.view(*shape)
        std = std.view(*shape)

        z = (x - mean) / std

        if return_log_deriv:
            log_deriv = -torch.log(std).expand_as(x)
            return z, log_deriv
        else:
            return z, None

    def destandardize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse of standardize.

        Args:
            z: Standardized tensor.

        Returns:
            x: De-standardized tensor.
        """
        mean, std = self.get_stats()

        shape = [1, self.num_features] + [1] * (z.dim() - 2)
        mean = mean.view(*shape)
        std = std.view(*shape)

        return z * std + mean


class GaugeFixingManager:
    """
    Manager for gauge-fixing during training.

    Handles:
    - EMA updates during minibatch training
    - Periodic full-pass recomputation
    - Proper freeze/unfreeze behavior

    Args:
        transform: The transform module.
        momentum: EMA momentum for running stats.
        full_refresh_every: Recompute full stats every N outer iterations.
        warmup_batches: Number of batches before using EMA (use batch stats initially).
    """

    def __init__(
        self,
        transform: nn.Module,
        momentum: float = 0.1,
        full_refresh_every: int = 5,
        warmup_batches: int = 10,
    ):
        self.transform = transform
        self.momentum = momentum
        self.full_refresh_every = full_refresh_every
        self.warmup_batches = warmup_batches

        self.batch_count = 0
        self.outer_iter = 0

    def update_batch(self, x: torch.Tensor):
        """
        Update gauge-fixing stats from a batch.

        Called during training after each batch.

        Args:
            x: Input batch [B, F, ...].
        """
        self.batch_count += 1

        with torch.no_grad():
            # Compute pre-normalized spline output
            s = self.transform.forward_prenorm(x)

            # Compute batch statistics
            dims = [0] + list(range(2, s.dim()))
            batch_mean = s.mean(dim=dims)
            batch_var = s.var(dim=dims, unbiased=False)

            # During warmup, use higher momentum for faster adaptation
            if self.batch_count < self.warmup_batches:
                momentum = 0.5  # Faster adaptation during warmup
            else:
                momentum = self.momentum

            # Update running stats
            self.transform.running_mean = (
                (1 - momentum) * self.transform.running_mean +
                momentum * batch_mean
            )
            self.transform.running_var = (
                (1 - momentum) * self.transform.running_var +
                momentum * batch_var.clamp(min=1e-8)
            )

    def full_refresh(self, dataloader: DataLoader, device: torch.device):
        """
        Full pass over data to recompute statistics.

        Args:
            dataloader: DataLoader for training data.
            device: Device to use.
        """
        self.transform.eval()

        all_s = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(device)
                s = self.transform.forward_prenorm(x)
                all_s.append(s.cpu())

        all_s = torch.cat(all_s, dim=0)

        # Compute statistics
        dims = [0] + list(range(2, all_s.dim()))
        self.transform.running_mean = all_s.mean(dim=dims).to(device)
        self.transform.running_var = all_s.var(dim=dims, unbiased=False).clamp(min=1e-8).to(device)

        self.transform.train()

    def should_refresh(self) -> bool:
        """Check if full refresh is due."""
        return (
            self.full_refresh_every > 0 and
            self.outer_iter > 0 and
            self.outer_iter % self.full_refresh_every == 0
        )

    def step_outer(self):
        """Called at the end of each outer iteration."""
        self.outer_iter += 1
        self.batch_count = 0


def compute_standardization_stats(
    transform: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute standardization statistics for a transform.

    Args:
        transform: Transform module with forward_prenorm method.
        dataloader: DataLoader yielding batches.
        device: Device to use.
        max_batches: Maximum batches to process (None = all).

    Returns:
        (mean, var) tensors of shape [num_features].
    """
    transform.eval()

    running_stats = RunningStats(transform.num_features)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device)
            s = transform.forward_prenorm(x)

            # Use Welford's algorithm for stability
            running_stats.update_welford(s.cpu())

    running_stats.finalize_welford()

    transform.train()

    return running_stats.running_mean.to(device), running_stats.running_var.to(device)


def check_gauge_quality(
    z: torch.Tensor,
    tolerance: float = 0.1,
) -> Tuple[bool, dict]:
    """
    Check if transformed data is properly gauge-fixed.

    Args:
        z: Transformed data [B, F, ...].
        tolerance: Allowed deviation from target (0 for mean, 1 for var).

    Returns:
        (passed, stats_dict) where stats_dict contains actual mean/var.
    """
    # Compute per-feature statistics
    dims = [0] + list(range(2, z.dim()))
    mean = z.mean(dim=dims)
    var = z.var(dim=dims, unbiased=False)

    # Check tolerance
    mean_ok = (mean.abs() < tolerance).all()
    var_ok = ((var - 1).abs() < tolerance).all()

    passed = mean_ok and var_ok

    stats = {
        'mean': mean,
        'var': var,
        'mean_max_dev': mean.abs().max().item(),
        'var_max_dev': (var - 1).abs().max().item(),
    }

    return passed, stats
