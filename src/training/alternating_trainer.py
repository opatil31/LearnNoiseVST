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
    lambda_binned: float = 10.0  # Strong weight for binned variance loss
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
            "binned_loss": [],
            "mse_loss": [],
        }

    def _setup_losses(self):
        """Initialize loss functions."""
        num_features = getattr(self.transform, 'num_features', 1)

        self.transform_loss_fn = CombinedTransformLoss(
            num_features=num_features,
            lambda_homo=self.config.lambda_homo,
            lambda_vf=self.config.lambda_vf,
            lambda_binned=self.config.lambda_binned,
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
        total_binned = 0.0
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
            # Pass x_samples for binned variance loss (bins by input signal level)
            losses = self.transform_loss_fn(
                z=z,
                z_hat=z_hat,
                residuals=residuals,
                log_deriv=log_deriv,
                x_samples=x,
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
            total_binned += losses.get("binned", 0.0)
            num_batches += 1

        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_homo = total_homo / max(num_batches, 1)
        avg_vf = total_vf / max(num_batches, 1)
        avg_binned = total_binned / max(num_batches, 1)

        # Log
        self.diagnostics.convergence.log_losses(avg_loss, 0.0, {
            "homo": avg_homo,
            "vf": avg_vf,
            "binned": avg_binned,
        })
        self.diagnostics.convergence.log_gradients(self.transform, self.denoiser)

        return {
            "loss": avg_loss,
            "homo": avg_homo,
            "vf": avg_vf,
            "binned": avg_binned,
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
            self.history["binned_loss"].append(transform_results["binned"])

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


@dataclass
class StagedTrainerConfig:
    """
    Configuration for staged training approach.

    Staged training improves VST learning by providing clearer gradient signals:
    - Stage 1 (Warmup): Train denoiser only, transform frozen
    - Stage 2 (VST Focus): Train VST only, denoiser frozen (key insight from Noise2VST)
    - Stage 3 (Refinement): Joint fine-tuning with low learning rates
    """

    # Stage 1: Denoiser warmup (transform frozen)
    warmup_epochs: int = 20
    warmup_denoiser_lr: float = 1e-3
    warmup_inner_iters: int = 500

    # Stage 2: VST focus (denoiser frozen) - key stage for learning good VST
    vst_epochs: int = 50
    vst_lr: float = 1e-3
    vst_inner_iters: int = 100

    # Stage 3: Joint refinement (both unfrozen, low LR)
    refine_epochs: int = 30
    refine_transform_lr: float = 1e-4
    refine_denoiser_lr: float = 1e-5  # Even slower to preserve VST
    refine_inner_iters: int = 50

    # Batch sizes
    batch_size: int = 256

    # Loss weights for VST stage
    lambda_homo: float = 1.0
    lambda_vf: float = 0.1
    lambda_binned: float = 10.0
    lambda_shape: float = 0.1
    lambda_reg: float = 0.01

    # Gauge-fixing
    gauge_momentum: float = 0.1
    gauge_refresh_every: int = 5

    # Logging
    log_every: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class StagedTrainer:
    """
    Staged training for VST learning.

    This trainer implements the key insight from Noise2VST: freezing the denoiser
    during VST training provides clearer gradient signals for variance stabilization.

    Unlike full Noise2VST (which uses a pre-trained Gaussian denoiser), this approach:
    1. Trains the denoiser on our actual data first (domain-agnostic)
    2. Freezes it to train the VST (clear gradients)
    3. Optionally fine-tunes both together

    This preserves noise characterization capability while improving VST learning.

    Training Stages:
        Stage 1 (Warmup): Train denoiser with identity transform
        Stage 2 (VST Focus): Freeze denoiser, train VST (key improvement)
        Stage 3 (Refinement): Joint fine-tuning with low learning rates

    Args:
        transform: The monotone transform module.
        denoiser: The blind-spot denoiser module.
        config: Staged trainer configuration.
    """

    def __init__(
        self,
        transform: nn.Module,
        denoiser: nn.Module,
        config: Optional[StagedTrainerConfig] = None,
    ):
        self.config = config or StagedTrainerConfig()
        self.device = torch.device(self.config.device)

        # Move models to device
        self.transform = transform.to(self.device)
        self.denoiser = denoiser.to(self.device)

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

        # History tracking
        self.history: Dict[str, List[float]] = {
            "stage": [],
            "epoch": [],
            "transform_loss": [],
            "denoiser_loss": [],
            "homo_loss": [],
            "vf_loss": [],
            "binned_loss": [],
            "mse_loss": [],
        }

        # Current stage
        self.current_stage = 0

    def _setup_losses(self):
        """Initialize loss functions."""
        num_features = getattr(self.transform, 'num_features', 1)

        self.transform_loss_fn = CombinedTransformLoss(
            num_features=num_features,
            lambda_homo=self.config.lambda_homo,
            lambda_vf=self.config.lambda_vf,
            lambda_binned=self.config.lambda_binned,
            lambda_shape=self.config.lambda_shape,
            lambda_reg=self.config.lambda_reg,
        )

        self.denoiser_loss_fn = DenoiserLoss(
            reduction="mean",
            robust=False,
        )

    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in a module."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module: nn.Module):
        """Unfreeze all parameters in a module."""
        module.train()
        for param in module.parameters():
            param.requires_grad = True

    def _train_denoiser_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        max_iters: int,
    ) -> Dict[str, float]:
        """Train denoiser for one epoch with frozen transform."""
        self._freeze_module(self.transform)
        self._unfreeze_module(self.denoiser)

        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            if i >= max_iters:
                break

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)

            # Forward through frozen transform
            with torch.no_grad():
                z = self.transform(x)

            # Denoiser forward
            z_hat = self.denoiser(z)

            # Compute loss
            loss, components = self.denoiser_loss_fn(z, z_hat)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse += components.get("mse", loss.item())
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "mse": total_mse / max(num_batches, 1),
        }

    def _train_transform_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        max_iters: int,
    ) -> Dict[str, float]:
        """Train transform for one epoch with frozen denoiser."""
        self._unfreeze_module(self.transform)
        self._freeze_module(self.denoiser)

        total_loss = 0.0
        total_homo = 0.0
        total_vf = 0.0
        total_binned = 0.0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            if i >= max_iters:
                break

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)

            # Forward through transform
            z = self.transform(x)

            # Get denoiser predictions (no grad)
            with torch.no_grad():
                z_hat = self.denoiser(z)

            # Compute residuals
            residuals = z - z_hat

            # Get log derivatives
            if hasattr(self.transform, 'log_derivative'):
                log_deriv = self.transform.log_derivative(x)
            else:
                log_deriv = None

            # Compute transform loss (pass x for binning)
            losses = self.transform_loss_fn(
                z=z,
                z_hat=z_hat,
                residuals=residuals,
                log_deriv=log_deriv,
                x_samples=x,
            )

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.transform.parameters(), 1.0)
            optimizer.step()

            # Update gauge-fixing stats
            self.gauge_manager.update_batch(x)

            total_loss += losses["total"].item()
            total_homo += losses.get("homo", 0.0)
            total_vf += losses.get("vf", 0.0)
            total_binned += losses.get("binned", 0.0)
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "homo": total_homo / max(num_batches, 1),
            "vf": total_vf / max(num_batches, 1),
            "binned": total_binned / max(num_batches, 1),
        }

    def _train_joint_epoch(
        self,
        dataloader: DataLoader,
        transform_optimizer: optim.Optimizer,
        denoiser_optimizer: optim.Optimizer,
        max_iters: int,
    ) -> Dict[str, float]:
        """Train both models jointly for one epoch."""
        self._unfreeze_module(self.transform)
        self._unfreeze_module(self.denoiser)

        total_t_loss = 0.0
        total_d_loss = 0.0
        total_homo = 0.0
        total_binned = 0.0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            if i >= max_iters:
                break

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)

            # === Transform step ===
            self.denoiser.eval()
            z = self.transform(x)

            with torch.no_grad():
                z_hat = self.denoiser(z)

            residuals = z - z_hat

            if hasattr(self.transform, 'log_derivative'):
                log_deriv = self.transform.log_derivative(x)
            else:
                log_deriv = None

            t_losses = self.transform_loss_fn(
                z=z, z_hat=z_hat, residuals=residuals,
                log_deriv=log_deriv, x_samples=x,
            )

            transform_optimizer.zero_grad()
            t_losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.transform.parameters(), 1.0)
            transform_optimizer.step()

            # === Denoiser step ===
            self.transform.eval()
            self.denoiser.train()

            with torch.no_grad():
                z = self.transform(x)

            z_hat = self.denoiser(z)
            d_loss, d_components = self.denoiser_loss_fn(z, z_hat)

            denoiser_optimizer.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
            denoiser_optimizer.step()

            # Update gauge-fixing stats
            self.gauge_manager.update_batch(x)

            total_t_loss += t_losses["total"].item()
            total_d_loss += d_loss.item()
            total_homo += t_losses.get("homo", 0.0)
            total_binned += t_losses.get("binned", 0.0)
            num_batches += 1

        return {
            "transform_loss": total_t_loss / max(num_batches, 1),
            "denoiser_loss": total_d_loss / max(num_batches, 1),
            "homo": total_homo / max(num_batches, 1),
            "binned": total_binned / max(num_batches, 1),
        }

    def _log_history(
        self,
        stage: int,
        epoch: int,
        transform_loss: float = 0.0,
        denoiser_loss: float = 0.0,
        homo_loss: float = 0.0,
        vf_loss: float = 0.0,
        binned_loss: float = 0.0,
        mse_loss: float = 0.0,
    ):
        """Log metrics to history."""
        self.history["stage"].append(stage)
        self.history["epoch"].append(epoch)
        self.history["transform_loss"].append(transform_loss)
        self.history["denoiser_loss"].append(denoiser_loss)
        self.history["homo_loss"].append(homo_loss)
        self.history["vf_loss"].append(vf_loss)
        self.history["binned_loss"].append(binned_loss)
        self.history["mse_loss"].append(mse_loss)

    def train(
        self,
        dataloader: DataLoader,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Main staged training loop.

        Args:
            dataloader: Training data loader.
            callback: Optional callback function called each epoch.

        Returns:
            Training result dictionary with history.
        """
        logger.info("Starting staged training")
        logger.info(f"Stage 1: {self.config.warmup_epochs} epochs denoiser warmup")
        logger.info(f"Stage 2: {self.config.vst_epochs} epochs VST training (frozen denoiser)")
        logger.info(f"Stage 3: {self.config.refine_epochs} epochs joint refinement")

        start_time = time.time()
        global_epoch = 0

        # ===== STAGE 1: Denoiser Warmup =====
        self.current_stage = 1
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Denoiser Warmup (transform frozen)")
        logger.info("=" * 60)

        denoiser_optimizer = optim.Adam(
            self.denoiser.parameters(),
            lr=self.config.warmup_denoiser_lr,
        )

        for epoch in range(self.config.warmup_epochs):
            results = self._train_denoiser_epoch(
                dataloader, denoiser_optimizer,
                max_iters=self.config.warmup_inner_iters,
            )

            self._log_history(
                stage=1, epoch=global_epoch,
                denoiser_loss=results["loss"],
                mse_loss=results["mse"],
            )

            if epoch % self.config.log_every == 0:
                logger.info(
                    f"Stage 1 Epoch {epoch}/{self.config.warmup_epochs} "
                    f"D_loss={results['loss']:.4f} MSE={results['mse']:.4f}"
                )

            if callback:
                callback(self, global_epoch, {"stage": 1, **results})

            global_epoch += 1

        # ===== STAGE 2: VST Training (Frozen Denoiser) =====
        self.current_stage = 2
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: VST Training (denoiser frozen)")
        logger.info("This is the KEY stage - VST gets 100% of gradient signal")
        logger.info("=" * 60)

        transform_optimizer = optim.Adam(
            self.transform.parameters(),
            lr=self.config.vst_lr,
        )

        for epoch in range(self.config.vst_epochs):
            results = self._train_transform_epoch(
                dataloader, transform_optimizer,
                max_iters=self.config.vst_inner_iters,
            )

            # Gauge refresh
            if epoch % self.config.gauge_refresh_every == 0:
                self.gauge_manager.full_refresh(dataloader, self.device)

            self._log_history(
                stage=2, epoch=global_epoch,
                transform_loss=results["loss"],
                homo_loss=results["homo"],
                vf_loss=results["vf"],
                binned_loss=results["binned"],
            )

            if epoch % self.config.log_every == 0:
                logger.info(
                    f"Stage 2 Epoch {epoch}/{self.config.vst_epochs} "
                    f"T_loss={results['loss']:.4f} "
                    f"binned={results['binned']:.4f} "
                    f"homo={results['homo']:.6f}"
                )

            if callback:
                callback(self, global_epoch, {"stage": 2, **results})

            global_epoch += 1

        # ===== STAGE 3: Joint Refinement =====
        if self.config.refine_epochs > 0:
            self.current_stage = 3
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 3: Joint Refinement (low learning rates)")
            logger.info("=" * 60)

            transform_optimizer = optim.Adam(
                self.transform.parameters(),
                lr=self.config.refine_transform_lr,
            )
            denoiser_optimizer = optim.Adam(
                self.denoiser.parameters(),
                lr=self.config.refine_denoiser_lr,
            )

            for epoch in range(self.config.refine_epochs):
                results = self._train_joint_epoch(
                    dataloader, transform_optimizer, denoiser_optimizer,
                    max_iters=self.config.refine_inner_iters,
                )

                # Gauge refresh
                if epoch % self.config.gauge_refresh_every == 0:
                    self.gauge_manager.full_refresh(dataloader, self.device)

                self._log_history(
                    stage=3, epoch=global_epoch,
                    transform_loss=results["transform_loss"],
                    denoiser_loss=results["denoiser_loss"],
                    homo_loss=results["homo"],
                    binned_loss=results["binned"],
                )

                if epoch % self.config.log_every == 0:
                    logger.info(
                        f"Stage 3 Epoch {epoch}/{self.config.refine_epochs} "
                        f"T_loss={results['transform_loss']:.4f} "
                        f"D_loss={results['denoiser_loss']:.4f} "
                        f"binned={results['binned']:.4f}"
                    )

                if callback:
                    callback(self, global_epoch, {"stage": 3, **results})

                global_epoch += 1

        total_time = time.time() - start_time
        logger.info(f"\nStaged training completed in {total_time:.1f}s")

        return {
            "history": self.history,
            "total_epochs": global_epoch,
        }

    def extract_variance_function(
        self,
        x_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract variance function σ²(μ) from learned VST.

        For optimal VST: T'(μ) ∝ 1/σ(μ)
        Therefore: σ(μ) ∝ 1/T'(μ)

        This gives us the noise variance as a function of signal level,
        which can be used for noise-aligned data augmentation.

        Args:
            x_grid: Grid of signal values [N] or [N, 1] to evaluate at.

        Returns:
            Tuple of (x_grid, variance_values) where variance is normalized
            to have mean 1.
        """
        self.transform.eval()

        if x_grid.dim() == 1:
            x_grid = x_grid.unsqueeze(1)

        # Expand to match transform's expected input shape
        num_features = getattr(self.transform, 'num_features', 1)
        if x_grid.shape[1] == 1 and num_features > 1:
            x_grid = x_grid.expand(-1, num_features)

        x_grid = x_grid.to(self.device)

        with torch.no_grad():
            # Get log derivative: log T'(x)
            if hasattr(self.transform, 'log_derivative'):
                log_deriv = self.transform.log_derivative(x_grid)
            else:
                # Numerical approximation
                eps = 1e-4
                z_plus = self.transform(x_grid + eps)
                z_minus = self.transform(x_grid - eps)
                deriv = (z_plus - z_minus) / (2 * eps)
                log_deriv = torch.log(deriv.abs().clamp(min=1e-8))

            # σ(μ) ∝ 1/T'(μ) => log σ(μ) = -log T'(μ) + const
            # σ²(μ) ∝ 1/T'(μ)² => log σ²(μ) = -2 * log T'(μ) + const
            log_var = -2 * log_deriv

            # Take first feature if multi-feature
            if log_var.dim() > 1:
                log_var = log_var[:, 0]
                x_out = x_grid[:, 0]
            else:
                x_out = x_grid.squeeze()

            # Normalize so mean variance = 1
            log_var = log_var - log_var.mean()
            variance = torch.exp(log_var)

        return x_out.cpu(), variance.cpu()

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
            "current_stage": self.current_stage,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.transform.load_state_dict(checkpoint["transform_state"])
        self.denoiser.load_state_dict(checkpoint["denoiser_state"])
        self.history = checkpoint.get("history", self.history)
        self.current_stage = checkpoint.get("current_stage", 0)
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
