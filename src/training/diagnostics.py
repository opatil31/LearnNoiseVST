"""
Convergence diagnostics for training monitoring.

This module provides tools for monitoring:
1. Training convergence (loss curves, gradients)
2. Gauge-fixing quality (mean/variance tracking)
3. Blind-spot property verification (leakage detection)
4. Transform quality (monotonicity, smoothness)
5. Noise model quality (residual statistics)

These diagnostics help detect issues early and guide hyperparameter tuning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import warnings


@dataclass
class DiagnosticResult:
    """Container for diagnostic check results."""
    name: str
    passed: bool
    value: float
    threshold: float
    message: str
    details: Dict = field(default_factory=dict)


class ConvergenceDiagnostics:
    """
    Monitors training convergence.

    Tracks:
    - Loss trajectory and smoothed trends
    - Gradient norms for each parameter group
    - Learning rate schedules
    - Early stopping criteria

    Args:
        window_size: Window for moving average smoothing.
        patience: Number of epochs without improvement for early stopping.
        min_delta: Minimum improvement to count as progress.
    """

    def __init__(
        self,
        window_size: int = 50,
        patience: int = 20,
        min_delta: float = 1e-4,
    ):
        self.window_size = window_size
        self.patience = patience
        self.min_delta = min_delta

        # Loss history
        self.transform_losses: List[float] = []
        self.denoiser_losses: List[float] = []
        self.total_losses: List[float] = []

        # Component losses
        self.loss_components: Dict[str, List[float]] = {}

        # Gradient norms
        self.transform_grad_norms: List[float] = []
        self.denoiser_grad_norms: List[float] = []

        # Early stopping state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

    def log_losses(
        self,
        transform_loss: float,
        denoiser_loss: float,
        components: Optional[Dict[str, float]] = None,
    ):
        """Log losses for current iteration."""
        self.transform_losses.append(transform_loss)
        self.denoiser_losses.append(denoiser_loss)
        self.total_losses.append(transform_loss + denoiser_loss)

        if components:
            for name, value in components.items():
                if name not in self.loss_components:
                    self.loss_components[name] = []
                self.loss_components[name].append(value)

    def log_gradients(
        self,
        transform_model: nn.Module,
        denoiser_model: nn.Module,
    ):
        """Log gradient norms for models."""
        transform_norm = self._compute_grad_norm(transform_model)
        denoiser_norm = self._compute_grad_norm(denoiser_model)

        self.transform_grad_norms.append(transform_norm)
        self.denoiser_grad_norms.append(denoiser_norm)

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm for a model."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def get_smoothed_loss(self, which: str = 'total') -> float:
        """Get smoothed loss value."""
        if which == 'transform':
            losses = self.transform_losses
        elif which == 'denoiser':
            losses = self.denoiser_losses
        else:
            losses = self.total_losses

        if len(losses) < self.window_size:
            return np.mean(losses) if losses else float('inf')
        return np.mean(losses[-self.window_size:])

    def check_early_stopping(self) -> Tuple[bool, str]:
        """Check if training should stop early."""
        current_loss = self.get_smoothed_loss('total')

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            return False, "Improving"
        else:
            self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                return True, f"No improvement for {self.patience} epochs"
            return False, f"No improvement for {self.epochs_without_improvement} epochs"

    def check_gradient_health(
        self,
        max_norm: float = 100.0,
        min_norm: float = 1e-8,
    ) -> DiagnosticResult:
        """Check for vanishing/exploding gradients."""
        if not self.transform_grad_norms or not self.denoiser_grad_norms:
            return DiagnosticResult(
                name="gradient_health",
                passed=True,
                value=0.0,
                threshold=max_norm,
                message="No gradients logged yet"
            )

        t_norm = self.transform_grad_norms[-1]
        d_norm = self.denoiser_grad_norms[-1]
        max_observed = max(t_norm, d_norm)
        min_observed = min(t_norm, d_norm)

        issues = []
        if max_observed > max_norm:
            issues.append(f"Exploding gradients: {max_observed:.2e}")
        if min_observed < min_norm:
            issues.append(f"Vanishing gradients: {min_observed:.2e}")

        passed = len(issues) == 0
        message = "Gradients healthy" if passed else "; ".join(issues)

        return DiagnosticResult(
            name="gradient_health",
            passed=passed,
            value=max_observed,
            threshold=max_norm,
            message=message,
            details={"transform_norm": t_norm, "denoiser_norm": d_norm}
        )

    def check_loss_plateau(
        self,
        window: int = 100,
        threshold: float = 1e-5,
    ) -> DiagnosticResult:
        """Check if loss has plateaued."""
        if len(self.total_losses) < window:
            return DiagnosticResult(
                name="loss_plateau",
                passed=True,
                value=0.0,
                threshold=threshold,
                message="Not enough data"
            )

        recent = self.total_losses[-window:]
        std = np.std(recent)

        passed = std > threshold
        message = "Loss converging" if not passed else "Loss still varying"

        return DiagnosticResult(
            name="loss_plateau",
            passed=passed,
            value=std,
            threshold=threshold,
            message=message
        )

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {
            "num_iterations": len(self.total_losses),
            "current_loss": self.total_losses[-1] if self.total_losses else None,
            "smoothed_loss": self.get_smoothed_loss(),
            "best_loss": self.best_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

        # Add component breakdown
        for name, values in self.loss_components.items():
            if values:
                summary[f"current_{name}"] = values[-1]

        return summary


class GaugeQualityMonitor:
    """
    Monitors gauge-fixing quality during training.

    Tracks per-feature mean and variance to ensure they stay
    close to 0 and 1 respectively.

    Args:
        num_features: Number of features to track.
        tolerance: Allowed deviation from target.
        history_size: Number of checks to keep in history.
    """

    def __init__(
        self,
        num_features: int,
        tolerance: float = 0.1,
        history_size: int = 100,
    ):
        self.num_features = num_features
        self.tolerance = tolerance
        self.history_size = history_size

        self.mean_history: deque = deque(maxlen=history_size)
        self.var_history: deque = deque(maxlen=history_size)

    def update(self, z: torch.Tensor):
        """
        Update with transformed data.

        Args:
            z: Transformed data [B, num_features, ...].
        """
        with torch.no_grad():
            # Compute per-feature statistics
            dims = [0] + list(range(2, z.dim()))
            mean = z.mean(dim=dims).cpu().numpy()
            var = z.var(dim=dims, unbiased=False).cpu().numpy()

            self.mean_history.append(mean)
            self.var_history.append(var)

    def check_quality(self) -> DiagnosticResult:
        """Check if gauge-fixing is working properly."""
        if not self.mean_history:
            return DiagnosticResult(
                name="gauge_quality",
                passed=True,
                value=0.0,
                threshold=self.tolerance,
                message="No data yet"
            )

        # Use most recent values
        mean = self.mean_history[-1]
        var = self.var_history[-1]

        mean_dev = np.abs(mean).max()
        var_dev = np.abs(var - 1).max()

        passed = (mean_dev < self.tolerance) and (var_dev < self.tolerance)

        message = f"Mean dev: {mean_dev:.4f}, Var dev: {var_dev:.4f}"

        return DiagnosticResult(
            name="gauge_quality",
            passed=passed,
            value=max(mean_dev, var_dev),
            threshold=self.tolerance,
            message=message,
            details={
                "mean_max_dev": mean_dev,
                "var_max_dev": var_dev,
                "worst_mean_feature": int(np.argmax(np.abs(mean))),
                "worst_var_feature": int(np.argmax(np.abs(var - 1))),
            }
        )

    def get_trend(self) -> Dict:
        """Get trend of gauge quality over history."""
        if len(self.mean_history) < 2:
            return {"trend": "insufficient_data"}

        mean_devs = [np.abs(m).max() for m in self.mean_history]
        var_devs = [np.abs(v - 1).max() for v in self.var_history]

        # Simple trend: compare first half to second half
        mid = len(mean_devs) // 2
        mean_improving = np.mean(mean_devs[mid:]) < np.mean(mean_devs[:mid])
        var_improving = np.mean(var_devs[mid:]) < np.mean(var_devs[:mid])

        return {
            "mean_improving": mean_improving,
            "var_improving": var_improving,
            "mean_trend": mean_devs,
            "var_trend": var_devs,
        }


class BlindSpotLeakageDetector:
    """
    Detects information leakage in blind-spot denoising.

    The blind-spot property requires that the prediction for feature f
    does not depend on the input at feature f:
        d(z_hat_f) / d(z_f) = 0

    This class checks this property via:
    1. Gradient-based: compute Jacobian and check diagonal
    2. Perturbation-based: check sensitivity to input changes

    Args:
        num_checks: Number of random samples to check.
        tolerance: Maximum allowed leakage.
    """

    def __init__(
        self,
        num_checks: int = 10,
        tolerance: float = 1e-5,
    ):
        self.num_checks = num_checks
        self.tolerance = tolerance
        self.leakage_history: List[float] = []

    def check_gradient_leakage(
        self,
        denoiser: nn.Module,
        z: torch.Tensor,
    ) -> DiagnosticResult:
        """
        Check for leakage via gradient computation.

        Computes d(z_hat_f)/d(z_f) for each feature f and checks
        that it's approximately zero.

        Args:
            denoiser: The blind-spot denoiser.
            z: Input tensor [B, d] or [B, C, H, W].
        """
        z = z.clone().detach().requires_grad_(True)

        # Forward pass
        z_hat = denoiser(z)

        # Compute diagonal of Jacobian
        B = z.shape[0]

        if z.dim() == 2:
            # Tabular: [B, d]
            d = z.shape[1]
            leakages = []

            for f in range(min(d, self.num_checks)):
                # Gradient of z_hat[:, f] w.r.t z
                grad = torch.autograd.grad(
                    z_hat[:, f].sum(),
                    z,
                    create_graph=False,
                    retain_graph=True,
                )[0]

                # Leakage is the diagonal element
                leakage = grad[:, f].abs().mean().item()
                leakages.append(leakage)
        else:
            # Image: [B, C, H, W] - check random pixels
            C, H, W = z.shape[1:]
            leakages = []

            for _ in range(self.num_checks):
                c = np.random.randint(C)
                h = np.random.randint(H)
                w = np.random.randint(W)

                grad = torch.autograd.grad(
                    z_hat[:, c, h, w].sum(),
                    z,
                    create_graph=False,
                    retain_graph=True,
                )[0]

                leakage = grad[:, c, h, w].abs().mean().item()
                leakages.append(leakage)

        max_leakage = max(leakages)
        mean_leakage = np.mean(leakages)

        self.leakage_history.append(max_leakage)

        passed = max_leakage < self.tolerance
        message = f"Max leakage: {max_leakage:.2e}, Mean: {mean_leakage:.2e}"

        return DiagnosticResult(
            name="blind_spot_leakage",
            passed=passed,
            value=max_leakage,
            threshold=self.tolerance,
            message=message,
            details={"leakages": leakages, "mean_leakage": mean_leakage}
        )

    def check_perturbation_leakage(
        self,
        denoiser: nn.Module,
        z: torch.Tensor,
        perturbation: float = 0.1,
    ) -> DiagnosticResult:
        """
        Check for leakage via input perturbation.

        Perturbs each input feature and checks how much the
        corresponding output changes (should be zero for blind-spot).

        Args:
            denoiser: The blind-spot denoiser.
            z: Input tensor [B, d] or [B, C, H, W].
            perturbation: Size of perturbation.
        """
        with torch.no_grad():
            z_hat_orig = denoiser(z)

            leakages = []

            if z.dim() == 2:
                # Tabular
                d = z.shape[1]
                for f in range(min(d, self.num_checks)):
                    z_pert = z.clone()
                    z_pert[:, f] += perturbation
                    z_hat_pert = denoiser(z_pert)

                    # Change in output at same feature
                    change = (z_hat_pert[:, f] - z_hat_orig[:, f]).abs().mean().item()
                    leakages.append(change / perturbation)
            else:
                # Image
                C, H, W = z.shape[1:]
                for _ in range(self.num_checks):
                    c = np.random.randint(C)
                    h = np.random.randint(H)
                    w = np.random.randint(W)

                    z_pert = z.clone()
                    z_pert[:, c, h, w] += perturbation
                    z_hat_pert = denoiser(z_pert)

                    change = (z_hat_pert[:, c, h, w] - z_hat_orig[:, c, h, w]).abs().mean().item()
                    leakages.append(change / perturbation)

        max_leakage = max(leakages)

        passed = max_leakage < self.tolerance
        message = f"Max perturbation sensitivity: {max_leakage:.2e}"

        return DiagnosticResult(
            name="perturbation_leakage",
            passed=passed,
            value=max_leakage,
            threshold=self.tolerance,
            message=message,
            details={"leakages": leakages}
        )


class TransformQualityMonitor:
    """
    Monitors quality of the learned transform.

    Checks:
    - Monotonicity (derivative should be positive)
    - Smoothness (second derivative bounded)
    - No extreme compression/expansion

    Args:
        num_features: Number of features.
        min_derivative: Minimum allowed derivative (ensures monotonicity).
        max_derivative: Maximum allowed derivative (prevents explosion).
    """

    def __init__(
        self,
        num_features: int,
        min_derivative: float = 0.01,
        max_derivative: float = 100.0,
    ):
        self.num_features = num_features
        self.min_derivative = min_derivative
        self.max_derivative = max_derivative

        self.derivative_history: List[Dict] = []

    def check_monotonicity(
        self,
        transform: nn.Module,
        x_range: Tuple[float, float] = (-5, 5),
        num_points: int = 100,
    ) -> DiagnosticResult:
        """
        Check that transform is monotonically increasing.

        Args:
            transform: The transform module.
            x_range: Range to test.
            num_points: Number of test points.
        """
        device = next(transform.parameters()).device

        # Create test points
        x = torch.linspace(x_range[0], x_range[1], num_points, device=device)
        x = x.unsqueeze(0).expand(1, self.num_features, num_points)
        x = x.transpose(1, 2)  # [1, num_points, num_features]
        x = x.reshape(-1, self.num_features)  # [num_points, num_features]

        with torch.no_grad():
            # Get derivatives
            if hasattr(transform, 'derivative'):
                deriv = transform.derivative(x)
            else:
                # Numerical derivative
                eps = 1e-4
                x_plus = x + eps
                x_minus = x - eps
                y_plus = transform(x_plus)
                y_minus = transform(x_minus)
                deriv = (y_plus - y_minus) / (2 * eps)

        min_deriv = deriv.min().item()
        max_deriv = deriv.max().item()

        self.derivative_history.append({
            "min": min_deriv,
            "max": max_deriv,
        })

        passed = (min_deriv > self.min_derivative) and (max_deriv < self.max_derivative)

        issues = []
        if min_deriv <= self.min_derivative:
            issues.append(f"Non-monotonic (min deriv: {min_deriv:.4f})")
        if max_deriv >= self.max_derivative:
            issues.append(f"Extreme expansion (max deriv: {max_deriv:.2f})")

        message = "Transform healthy" if passed else "; ".join(issues)

        return DiagnosticResult(
            name="transform_monotonicity",
            passed=passed,
            value=min_deriv,
            threshold=self.min_derivative,
            message=message,
            details={"min_derivative": min_deriv, "max_derivative": max_deriv}
        )

    def check_smoothness(
        self,
        transform: nn.Module,
        x: torch.Tensor,
        eps: float = 1e-3,
    ) -> DiagnosticResult:
        """
        Check transform smoothness via second derivative.

        Args:
            transform: The transform module.
            x: Input data.
            eps: Step size for numerical differentiation.
        """
        with torch.no_grad():
            # Numerical second derivative
            y_center = transform(x)
            y_plus = transform(x + eps)
            y_minus = transform(x - eps)

            second_deriv = (y_plus - 2 * y_center + y_minus) / (eps ** 2)

            max_curvature = second_deriv.abs().max().item()
            mean_curvature = second_deriv.abs().mean().item()

        # Heuristic threshold
        threshold = 1000.0
        passed = max_curvature < threshold

        message = f"Max curvature: {max_curvature:.2f}, Mean: {mean_curvature:.2f}"

        return DiagnosticResult(
            name="transform_smoothness",
            passed=passed,
            value=max_curvature,
            threshold=threshold,
            message=message
        )


class ResidualStatisticsMonitor:
    """
    Monitors statistics of denoising residuals.

    After training, residuals u = z - z_hat should be approximately
    Gaussian if the transform is working properly.

    This monitors:
    - Normality (skewness, kurtosis)
    - Homoscedasticity (variance should be constant across input range)
    - Independence (autocorrelation for sequential data)
    """

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.residual_history: List[torch.Tensor] = []

    def update(self, z: torch.Tensor, z_hat: torch.Tensor):
        """
        Store residuals for analysis.

        Args:
            z: Original transformed data.
            z_hat: Denoised predictions.
        """
        residual = (z - z_hat).detach().cpu()
        self.residual_history.append(residual)

    def check_normality(
        self,
        skew_threshold: float = 0.5,
        kurtosis_threshold: float = 1.0,
    ) -> DiagnosticResult:
        """Check if residuals are approximately Gaussian."""
        if not self.residual_history:
            return DiagnosticResult(
                name="residual_normality",
                passed=True,
                value=0.0,
                threshold=skew_threshold,
                message="No residuals collected"
            )

        # Combine recent residuals
        residuals = torch.cat(self.residual_history[-10:], dim=0)

        # Flatten if needed
        if residuals.dim() > 2:
            residuals = residuals.view(residuals.shape[0], -1)

        # Compute per-feature statistics
        mean = residuals.mean(dim=0)
        std = residuals.std(dim=0)
        centered = (residuals - mean) / (std + 1e-8)

        skewness = (centered ** 3).mean(dim=0)
        kurtosis = (centered ** 4).mean(dim=0) - 3  # Excess kurtosis

        max_skew = skewness.abs().max().item()
        max_kurt = kurtosis.abs().max().item()

        passed = (max_skew < skew_threshold) and (max_kurt < kurtosis_threshold)

        message = f"Max |skewness|: {max_skew:.3f}, Max |excess kurtosis|: {max_kurt:.3f}"

        return DiagnosticResult(
            name="residual_normality",
            passed=passed,
            value=max_skew,
            threshold=skew_threshold,
            message=message,
            details={"max_skewness": max_skew, "max_kurtosis": max_kurt}
        )

    def check_homoscedasticity(
        self,
        z_hat: torch.Tensor,
        num_bins: int = 10,
        threshold: float = 0.5,
    ) -> DiagnosticResult:
        """Check if residual variance is constant across predictions."""
        if not self.residual_history:
            return DiagnosticResult(
                name="residual_homoscedasticity",
                passed=True,
                value=0.0,
                threshold=threshold,
                message="No residuals collected"
            )

        residuals = torch.cat(self.residual_history[-10:], dim=0)

        # Flatten
        if residuals.dim() > 2:
            residuals = residuals.view(residuals.shape[0], -1)

        # For simplicity, check overall
        z_flat = z_hat.view(-1).cpu()
        r_flat = residuals.view(-1)

        if len(z_flat) != len(r_flat):
            # Mismatch in sizes, skip check
            return DiagnosticResult(
                name="residual_homoscedasticity",
                passed=True,
                value=0.0,
                threshold=threshold,
                message="Size mismatch, skipping"
            )

        # Bin by prediction value
        bins = torch.linspace(z_flat.min(), z_flat.max(), num_bins + 1)
        variances = []

        for i in range(num_bins):
            mask = (z_flat >= bins[i]) & (z_flat < bins[i + 1])
            if mask.sum() > 10:
                var = r_flat[mask].var().item()
                variances.append(var)

        if len(variances) < 2:
            return DiagnosticResult(
                name="residual_homoscedasticity",
                passed=True,
                value=0.0,
                threshold=threshold,
                message="Insufficient data for bins"
            )

        # Coefficient of variation of variances
        var_array = np.array(variances)
        cv = var_array.std() / (var_array.mean() + 1e-8)

        passed = cv < threshold
        message = f"Variance CV across bins: {cv:.3f}"

        return DiagnosticResult(
            name="residual_homoscedasticity",
            passed=passed,
            value=cv,
            threshold=threshold,
            message=message,
            details={"bin_variances": variances}
        )


class DiagnosticSuite:
    """
    Complete diagnostic suite combining all monitors.

    Provides a unified interface for running all diagnostics
    and generating reports.

    Args:
        num_features: Number of features.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        num_features: int,
        config: Optional[Dict] = None,
    ):
        config = config or {}

        self.convergence = ConvergenceDiagnostics(
            window_size=config.get("window_size", 50),
            patience=config.get("patience", 20),
        )

        self.gauge = GaugeQualityMonitor(
            num_features=num_features,
            tolerance=config.get("gauge_tolerance", 0.1),
        )

        self.leakage = BlindSpotLeakageDetector(
            num_checks=config.get("leakage_checks", 10),
            tolerance=config.get("leakage_tolerance", 1e-5),
        )

        self.transform = TransformQualityMonitor(
            num_features=num_features,
            min_derivative=config.get("min_derivative", 0.01),
            max_derivative=config.get("max_derivative", 100.0),
        )

        self.residuals = ResidualStatisticsMonitor(num_features)

    def run_all_checks(
        self,
        transform_model: Optional[nn.Module] = None,
        denoiser_model: Optional[nn.Module] = None,
        z: Optional[torch.Tensor] = None,
        z_hat: Optional[torch.Tensor] = None,
    ) -> Dict[str, DiagnosticResult]:
        """Run all applicable diagnostic checks."""
        results = {}

        # Convergence checks
        results["gradient_health"] = self.convergence.check_gradient_health()
        results["loss_plateau"] = self.convergence.check_loss_plateau()

        # Gauge quality
        results["gauge_quality"] = self.gauge.check_quality()

        # Leakage (if we have denoiser and data)
        if denoiser_model is not None and z is not None:
            try:
                results["blind_spot_leakage"] = self.leakage.check_gradient_leakage(
                    denoiser_model, z[:min(4, len(z))]  # Limit batch size for speed
                )
            except Exception as e:
                warnings.warn(f"Leakage check failed: {e}")

        # Transform quality
        if transform_model is not None:
            results["transform_monotonicity"] = self.transform.check_monotonicity(
                transform_model
            )

        # Residual statistics
        if z is not None and z_hat is not None:
            results["residual_normality"] = self.residuals.check_normality()

        return results

    def generate_report(self) -> str:
        """Generate a text report of current diagnostic state."""
        lines = ["=" * 60, "DIAGNOSTIC REPORT", "=" * 60, ""]

        # Convergence summary
        summary = self.convergence.get_summary()
        lines.append("CONVERGENCE:")
        lines.append(f"  Iterations: {summary['num_iterations']}")
        lines.append(f"  Current loss: {summary['current_loss']:.6f}" if summary['current_loss'] else "  Current loss: N/A")
        lines.append(f"  Best loss: {summary['best_loss']:.6f}")
        lines.append(f"  Epochs without improvement: {summary['epochs_without_improvement']}")
        lines.append("")

        # Gauge quality
        gauge_check = self.gauge.check_quality()
        lines.append("GAUGE QUALITY:")
        lines.append(f"  Status: {'PASS' if gauge_check.passed else 'FAIL'}")
        lines.append(f"  {gauge_check.message}")
        lines.append("")

        # Recent gradient health
        grad_check = self.convergence.check_gradient_health()
        lines.append("GRADIENT HEALTH:")
        lines.append(f"  Status: {'PASS' if grad_check.passed else 'FAIL'}")
        lines.append(f"  {grad_check.message}")
        lines.append("")

        # Residual normality
        residual_check = self.residuals.check_normality()
        lines.append("RESIDUAL NORMALITY:")
        lines.append(f"  Status: {'PASS' if residual_check.passed else 'FAIL'}")
        lines.append(f"  {residual_check.message}")
        lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def should_stop_early(self) -> Tuple[bool, str]:
        """Check if training should stop based on all criteria."""
        # Check early stopping
        should_stop, reason = self.convergence.check_early_stopping()
        if should_stop:
            return True, reason

        # Check for critical failures
        gauge_check = self.gauge.check_quality()
        if not gauge_check.passed and gauge_check.value > 1.0:
            return True, "Gauge-fixing severely broken"

        grad_check = self.convergence.check_gradient_health()
        if not grad_check.passed and grad_check.value > 1000:
            return True, "Gradient explosion detected"

        return False, "Training should continue"
