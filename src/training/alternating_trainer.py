"""
Alternating optimization trainer for VST learning.

The training procedure alternates between:
1. Transform step: Fix denoiser D, update transform T to improve homoscedasticity
2. Denoiser step: Fix transform T, update denoiser D to minimize reconstruction error

Each step respects trust regions to prevent oscillation:
- Transform: δ_T controls how much T can change per outer iteration
- Denoiser: Re-trained from scratch or fine-tuned each outer iteration

Gauge-fixing is applied after each transform update to maintain E[z]=0, Var[z]=1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable, Union, List
from dataclasses import dataclass
import logging
import time

from .losses import (
    CombinedTransformLoss,
    DenoiserLoss,
    HomoscedasticityLoss,
    VarianceFlatnessLoss,
)
from .gauge_fixing import GaugeFixingManager, compute_standardization_stats
from .diagnostics import DiagnosticSuite


logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for alternating trainer."""

    # Outer loop
    num_outer_iters: int = 100
    patience: int = 20

    # Transform optimization
    transform_lr: float = 1e-3
    transform_inner_iters: int = 100
    transform_batch_size: int = 256

    # Denoiser optimization
    denoiser_lr: float = 1e-3
    denoiser_inner_iters: int = 500
    denoiser_batch_size: int = 64
    denoiser_retrain_from_scratch: bool = False

    # Loss weights
    lambda_homo: float = 1.0
    lambda_vf: float = 0.1
    lambda_shape: float = 0.1
    lambda_reg: float = 0.01
    lambda_prox: float = 0.1

    # Trust region
    prox_weight_schedule: str = "constant"  # "constant", "decay", "increase"

    # Gauge-fixing
    gauge_momentum: float = 0.1
    gauge_refresh_every: int = 5

    # Diagnostics
    log_every: int = 10
    diagnose_every: int = 50
    leakage_check_every: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AlternatingTrainer:
    """
    Trainer for alternating optimization of VST.

    This implements the bi-level optimization:
        min_T L_transform(T; D*(T))
        where D*(T) = argmin_D L_denoiser(D; T)

    In practice, we alternate:
    1. Fix D, take gradient steps on T
    2. Fix T, (re-)train D

    Args:
        transform: The monotone transform module.
        denoiser: The blind-spot denoiser module.
        config: Trainer configuration.
    """

    def __init__(
        self,
        transform: nn.Module,
        denoiser: nn.Module,
        config: Optional[TrainerConfig] = None,
    ):
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)

        # Move models to device
        self.transform = transform.to(self.device)
        self.denoiser = denoiser.to(self.device)

        # Store reference transform for proximity regularization
        self.transform_ref = None
        self._save_transform_reference()

        # Optimizers (created fresh each outer iteration for transform)
        self.transform_optimizer = None
        self.denoiser_optimizer = optim.Adam(
            self.denoiser.parameters(),
            lr=self.config.denoiser_lr,
        )

        # Loss functions
        self._setup_losses()

        # Gauge-fixing
        self.gauge_manager = GaugeFixingManager(
            transform=self.transform,
            momentum=self.config.gauge_momentum,
            full_refresh_every=self.config.gauge_refresh_every,
        )

        # Diagnostics
        num_features = getattr(self.transform, 'num_features', 1)
        self.diagnostics = DiagnosticSuite(num_features)

        # Training state
        self.outer_iter = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

        # History
        self.history: Dict[str, List[float]] = {
            "transform_loss": [],
            "denoiser_loss": [],
            "homo_loss": [],
            "vf_loss": [],
            "mse_loss": [],
        }

    def _setup_losses(self):
        """Initialize loss functions."""
        num_features = getattr(self.transform, 'num_features', 1)

        self.transform_loss_fn = CombinedTransformLoss(
            num_features=num_features,
            lambda_homo=self.config.lambda_homo,
            lambda_vf=self.config.lambda_vf,
            lambda_shape=self.config.lambda_shape,
            lambda_reg=self.config.lambda_reg,
        )

        self.denoiser_loss_fn = DenoiserLoss(
            reduction="mean",
            robust=False,
        )

    def _save_transform_reference(self):
        """Save copy of transform parameters for proximity regularization."""
        self.transform_ref = {
            name: param.clone().detach()
            for name, param in self.transform.named_parameters()
        }

    def _compute_proximity_loss(self) -> torch.Tensor:
        """Compute proximity loss to reference transform."""
        if self.transform_ref is None:
            return torch.tensor(0.0, device=self.device)

        prox_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.transform.named_parameters():
            if name in self.transform_ref:
                prox_loss = prox_loss + (param - self.transform_ref[name]).pow(2).sum()

        return prox_loss

    def _get_prox_weight(self) -> float:
        """Get proximity weight based on schedule."""
        base = self.config.lambda_prox

        if self.config.prox_weight_schedule == "constant":
            return base
        elif self.config.prox_weight_schedule == "decay":
            # Decay as training progresses
            decay = 0.95 ** self.outer_iter
            return base * decay
        elif self.config.prox_weight_schedule == "increase":
            # Increase to prevent oscillation
            growth = min(2.0, 1.0 + 0.01 * self.outer_iter)
            return base * growth
        else:
            return base

    def train_transform_step(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        One inner loop of transform optimization.

        Fixes the denoiser and updates transform to improve homoscedasticity.

        Args:
            dataloader: DataLoader for training data.

        Returns:
            Dictionary of loss values.
        """
        self.transform.train()
        self.denoiser.eval()

        # Fresh optimizer for transform
        self.transform_optimizer = optim.Adam(
            self.transform.parameters(),
            lr=self.config.transform_lr,
        )

        total_loss = 0.0
        total_homo = 0.0
        total_vf = 0.0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            if i >= self.config.transform_inner_iters:
                break

            # Get data
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)

            # Forward through transform
            z = self.transform(x)

            # Get denoiser predictions (no grad for denoiser)
            with torch.no_grad():
                z_hat = self.denoiser(z)

            # Compute residuals
            residuals = z - z_hat

            # Get log derivatives for variance computation
            if hasattr(self.transform, 'log_derivative'):
                log_deriv = self.transform.log_derivative(x)
            else:
                log_deriv = None

            # Compute transform loss
            losses = self.transform_loss_fn(
                z=z,
                z_hat=z_hat,
                residuals=residuals,
                log_deriv=log_deriv,
            )

            # Add proximity regularization
            prox_weight = self._get_prox_weight()
            prox_loss = self._compute_proximity_loss()
            total = losses["total"] + prox_weight * prox_loss

            # Backward
            self.transform_optimizer.zero_grad()
            total.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.transform.parameters(), 1.0)

            self.transform_optimizer.step()

            # Update gauge-fixing stats
            self.gauge_manager.update_batch(x)

            # Accumulate
            total_loss += total.item()
            total_homo += losses.get("homo", 0.0)
            total_vf += losses.get("vf", 0.0)
            num_batches += 1

        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_homo = total_homo / max(num_batches, 1)
        avg_vf = total_vf / max(num_batches, 1)

        # Log
        self.diagnostics.convergence.log_losses(avg_loss, 0.0, {
            "homo": avg_homo,
            "vf": avg_vf,
        })
        self.diagnostics.convergence.log_gradients(self.transform, self.denoiser)

        return {
            "loss": avg_loss,
            "homo": avg_homo,
            "vf": avg_vf,
        }

    def train_denoiser_step(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        One inner loop of denoiser optimization.

        Fixes the transform and updates denoiser to minimize reconstruction.

        Args:
            dataloader: DataLoader for training data.

        Returns:
            Dictionary of loss values.
        """
        self.transform.eval()
        self.denoiser.train()

        # Optionally reset denoiser
        if self.config.denoiser_retrain_from_scratch:
            self._reset_denoiser()

        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            if i >= self.config.denoiser_inner_iters:
                break

            # Get data
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)

            # Forward through transform (no grad)
            with torch.no_grad():
                z = self.transform(x)

            # Denoiser forward
            z_hat = self.denoiser(z)

            # Compute loss
            loss, components = self.denoiser_loss_fn(z, z_hat)

            # Backward
            self.denoiser_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)

            self.denoiser_optimizer.step()

            # Update diagnostics
            self.diagnostics.gauge.update(z)
            self.diagnostics.residuals.update(z, z_hat)

            # Accumulate
            total_loss += loss.item()
            total_mse += components.get("mse", loss.item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "mse": avg_mse,
        }

    def _reset_denoiser(self):
        """Reset denoiser parameters."""
        for module in self.denoiser.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        # Reset optimizer
        self.denoiser_optimizer = optim.Adam(
            self.denoiser.parameters(),
            lr=self.config.denoiser_lr,
        )

    def train(
        self,
        dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Main training loop with alternating optimization.

        Args:
            dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
            callback: Optional callback function called each outer iteration.

        Returns:
            Training history dictionary.
        """
        logger.info("Starting alternating optimization training")
        logger.info(f"Config: {self.config}")

        start_time = time.time()

        for outer_iter in range(self.config.num_outer_iters):
            self.outer_iter = outer_iter
            iter_start = time.time()

            # === Transform Step ===
            transform_results = self.train_transform_step(dataloader)
            self.history["transform_loss"].append(transform_results["loss"])
            self.history["homo_loss"].append(transform_results["homo"])
            self.history["vf_loss"].append(transform_results["vf"])

            # Save reference for next iteration's proximity
            self._save_transform_reference()

            # === Gauge Refresh ===
            if self.gauge_manager.should_refresh():
                logger.info(f"Outer {outer_iter}: Full gauge refresh")
                self.gauge_manager.full_refresh(dataloader, self.device)

            self.gauge_manager.step_outer()

            # === Denoiser Step ===
            denoiser_results = self.train_denoiser_step(dataloader)
            self.history["denoiser_loss"].append(denoiser_results["loss"])
            self.history["mse_loss"].append(denoiser_results["mse"])

            # === Logging ===
            if outer_iter % self.config.log_every == 0:
                elapsed = time.time() - iter_start
                logger.info(
                    f"Outer {outer_iter}/{self.config.num_outer_iters} "
                    f"T_loss={transform_results['loss']:.4f} "
                    f"D_loss={denoiser_results['loss']:.4f} "
                    f"({elapsed:.1f}s)"
                )

            # === Diagnostics ===
            if outer_iter % self.config.diagnose_every == 0:
                self._run_diagnostics(dataloader)

            # === Validation ===
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                self._check_improvement(val_loss)
            else:
                combined = transform_results["loss"] + denoiser_results["loss"]
                self._check_improvement(combined)

            # === Early Stopping ===
            if self.epochs_without_improvement >= self.config.patience:
                logger.info(f"Early stopping at outer iteration {outer_iter}")
                break

            # === Callback ===
            if callback is not None:
                callback(self, outer_iter, transform_results, denoiser_results)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")

        return {
            "history": self.history,
            "best_loss": self.best_loss,
            "final_outer_iter": self.outer_iter,
        }

    def _validate(self, dataloader: DataLoader) -> float:
        """Compute validation loss."""
        self.transform.eval()
        self.denoiser.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)

                z = self.transform(x)
                z_hat = self.denoiser(z)

                loss, _ = self.denoiser_loss_fn(z, z_hat)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _check_improvement(self, loss: float):
        """Check if loss improved and update early stopping state."""
        if loss < self.best_loss - 1e-4:
            self.best_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def _run_diagnostics(self, dataloader: DataLoader):
        """Run diagnostic checks."""
        # Get a batch for diagnostics
        batch = next(iter(dataloader))
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        x = x.to(self.device)

        with torch.no_grad():
            z = self.transform(x)
            z_hat = self.denoiser(z)

        # Run checks
        results = self.diagnostics.run_all_checks(
            transform_model=self.transform,
            denoiser_model=self.denoiser,
            z=z,
            z_hat=z_hat,
        )

        # Log issues
        for name, result in results.items():
            if not result.passed:
                logger.warning(f"Diagnostic {name} FAILED: {result.message}")

    def get_trained_models(self) -> Tuple[nn.Module, nn.Module]:
        """Get trained transform and denoiser."""
        return self.transform, self.denoiser

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "transform_state": self.transform.state_dict(),
            "denoiser_state": self.denoiser.state_dict(),
            "config": self.config,
            "history": self.history,
            "outer_iter": self.outer_iter,
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        # Use weights_only=False since checkpoint contains config objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.transform.load_state_dict(checkpoint["transform_state"])
        self.denoiser.load_state_dict(checkpoint["denoiser_state"])
        self.history = checkpoint.get("history", self.history)
        self.outer_iter = checkpoint.get("outer_iter", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        logger.info(f"Loaded checkpoint from {path}")


class LightweightTrainer:
    """
    Simplified trainer for quick experiments.

    Less configurable but easier to use for debugging
    and initial experiments.
    """

    def __init__(
        self,
        transform: nn.Module,
        denoiser: nn.Module,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.transform = transform.to(self.device)
        self.denoiser = denoiser.to(self.device)

        self.transform_opt = optim.Adam(transform.parameters(), lr=lr)
        self.denoiser_opt = optim.Adam(denoiser.parameters(), lr=lr)

        self.homo_loss = HomoscedasticityLoss()
        self.mse_loss = nn.MSELoss()

    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """Single training step updating both models."""
        x = x.to(self.device)

        # Transform step
        self.transform.train()
        self.denoiser.eval()

        z = self.transform(x)
        with torch.no_grad():
            z_hat = self.denoiser(z)

        residuals = z - z_hat
        t_loss = self.homo_loss(z_hat, residuals)

        self.transform_opt.zero_grad()
        t_loss.backward()
        self.transform_opt.step()

        # Denoiser step
        self.transform.eval()
        self.denoiser.train()

        with torch.no_grad():
            z = self.transform(x)

        z_hat = self.denoiser(z)
        d_loss = self.mse_loss(z_hat, z)

        self.denoiser_opt.zero_grad()
        d_loss.backward()
        self.denoiser_opt.step()

        return {
            "transform_loss": t_loss.item(),
            "denoiser_loss": d_loss.item(),
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        total_t = 0.0
        total_d = 0.0
        n = 0

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            losses = self.train_step(x)
            total_t += losses["transform_loss"]
            total_d += losses["denoiser_loss"]
            n += 1

        return {
            "transform_loss": total_t / n,
            "denoiser_loss": total_d / n,
        }
