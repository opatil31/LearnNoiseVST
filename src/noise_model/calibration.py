"""
Calibration residual generation for noise model fitting.

After training the transform T and denoiser D, we generate a calibration
dataset of residuals r = z - ẑ to fit the noise model. This module handles:

1. Generating residuals from a calibration data split
2. Grouping by channel/feature
3. Storing predictions and residuals for downstream fitting

The calibration dataset enables post-training noise characterization
without requiring ground truth clean signals.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class ResidualData:
    """Container for residual data for a single group/channel."""
    z_hat: np.ndarray  # Predicted clean signal
    r: np.ndarray  # Residuals (z - z_hat)
    z: Optional[np.ndarray] = None  # Original transformed data (optional)

    def __len__(self):
        return len(self.z_hat)

    def get_standardized(self) -> np.ndarray:
        """Get residuals standardized to unit variance."""
        std = np.std(self.r) + 1e-8
        return self.r / std


@dataclass
class CalibrationResult:
    """Complete calibration result with all groups."""
    residuals: Dict[int, ResidualData] = field(default_factory=dict)
    is_image: bool = False
    num_groups: int = 0
    total_samples: int = 0

    def __getitem__(self, group_id: int) -> ResidualData:
        return self.residuals[group_id]

    def __iter__(self):
        return iter(self.residuals.items())

    def get_all_residuals(self) -> np.ndarray:
        """Get all residuals concatenated."""
        return np.concatenate([rd.r for rd in self.residuals.values()])

    def get_group_ids(self) -> List[int]:
        """Get list of group IDs."""
        return list(self.residuals.keys())


class CalibrationDatasetGenerator:
    """
    Generate residual dataset from calibration split.

    For each sample x_i:
        z_i = T(x_i)
        ẑ_i = D(z_i)
        r_i = z_i - ẑ_i

    Store tuples: (ẑ, r, group_id) organized by group.

    Args:
        transform: Trained transform module.
        denoiser: Trained denoiser module.
        margin: Crop margin for images (removes boundary artifacts).
        device: Device for computation.
    """

    def __init__(
        self,
        transform: nn.Module,
        denoiser: nn.Module,
        margin: int = 8,
        device: str = "cpu",
    ):
        self.transform = transform
        self.denoiser = denoiser
        self.margin = margin
        self.device = torch.device(device)

        # Move models to device and set to eval mode
        self.transform.to(self.device).eval()
        self.denoiser.to(self.device).eval()

    @torch.no_grad()
    def generate(
        self,
        dataloader: DataLoader,
        is_image: bool = False,
        store_z: bool = False,
    ) -> CalibrationResult:
        """
        Generate calibration residuals from dataloader.

        Args:
            dataloader: DataLoader for calibration data.
            is_image: Whether data is image format [B, C, H, W].
            store_z: Whether to store original transformed data.

        Returns:
            CalibrationResult with residuals organized by group.
        """
        residuals_dict: Dict[int, Dict[str, List]] = defaultdict(
            lambda: {'z_hat': [], 'r': [], 'z': []}
        )

        total_samples = 0

        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)
            batch_size = x.shape[0]
            total_samples += batch_size

            # Forward pass
            z = self.transform(x)
            z_hat = self.denoiser(z)
            r = z - z_hat

            if is_image:
                self._process_image_batch(
                    z, z_hat, r, residuals_dict, store_z
                )
            else:
                self._process_tabular_batch(
                    z, z_hat, r, residuals_dict, store_z
                )

        # Concatenate and convert to numpy
        result = CalibrationResult(
            is_image=is_image,
            total_samples=total_samples,
        )

        for group_id, data in residuals_dict.items():
            z_hat_arr = np.concatenate(data['z_hat'])
            r_arr = np.concatenate(data['r'])
            z_arr = np.concatenate(data['z']) if store_z and data['z'] else None

            result.residuals[group_id] = ResidualData(
                z_hat=z_hat_arr,
                r=r_arr,
                z=z_arr,
            )

        result.num_groups = len(result.residuals)

        return result

    def _process_image_batch(
        self,
        z: torch.Tensor,
        z_hat: torch.Tensor,
        r: torch.Tensor,
        residuals_dict: Dict,
        store_z: bool,
    ):
        """Process image batch, cropping margins and grouping by channel."""
        B, C, H, W = z.shape
        m = self.margin

        # Crop margins
        if m > 0 and H > 2 * m and W > 2 * m:
            z = z[:, :, m:-m, m:-m]
            z_hat = z_hat[:, :, m:-m, m:-m]
            r = r[:, :, m:-m, m:-m]

        # Group by channel
        for c in range(C):
            z_hat_c = z_hat[:, c].flatten().cpu().numpy()
            r_c = r[:, c].flatten().cpu().numpy()

            residuals_dict[c]['z_hat'].append(z_hat_c)
            residuals_dict[c]['r'].append(r_c)

            if store_z:
                z_c = z[:, c].flatten().cpu().numpy()
                residuals_dict[c]['z'].append(z_c)

    def _process_tabular_batch(
        self,
        z: torch.Tensor,
        z_hat: torch.Tensor,
        r: torch.Tensor,
        residuals_dict: Dict,
        store_z: bool,
    ):
        """Process tabular batch, grouping by feature."""
        B, F = z.shape

        # Group by feature
        for f in range(F):
            z_hat_f = z_hat[:, f].cpu().numpy()
            r_f = r[:, f].cpu().numpy()

            residuals_dict[f]['z_hat'].append(z_hat_f)
            residuals_dict[f]['r'].append(r_f)

            if store_z:
                z_f = z[:, f].cpu().numpy()
                residuals_dict[f]['z'].append(z_f)

    def generate_with_delta_method(
        self,
        dataloader: DataLoader,
        is_image: bool = False,
    ) -> Tuple[CalibrationResult, Dict[int, np.ndarray]]:
        """
        Generate calibration residuals with delta-method variance propagation.

        For each observation, also computes the Jacobian-based variance
        contribution from the transform for more accurate noise characterization.

        Returns:
            (CalibrationResult, dict of log_derivatives per group)
        """
        calibration = self.generate(dataloader, is_image, store_z=True)

        # Get log derivatives for delta method
        log_derivs: Dict[int, List] = defaultdict(list)

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device)

                # Get log derivative from transform
                if hasattr(self.transform, 'log_derivative'):
                    log_d = self.transform.log_derivative(x)

                    if is_image:
                        m = self.margin
                        if m > 0:
                            log_d = log_d[:, :, m:-m, m:-m]

                        for c in range(log_d.shape[1]):
                            log_derivs[c].append(
                                log_d[:, c].flatten().cpu().numpy()
                            )
                    else:
                        for f in range(log_d.shape[1]):
                            log_derivs[f].append(log_d[:, f].cpu().numpy())

        # Concatenate
        log_deriv_arrays = {
            g: np.concatenate(vals) for g, vals in log_derivs.items()
        }

        return calibration, log_deriv_arrays


class ResidualAnalyzer:
    """
    Analyze residuals for quality assessment.

    Provides statistics and diagnostics for calibration residuals
    to verify transform quality before fitting noise model.
    """

    def __init__(self, calibration: CalibrationResult):
        self.calibration = calibration

    def compute_statistics(self) -> Dict:
        """Compute summary statistics for all groups."""
        stats = {}

        for group_id, data in self.calibration:
            r = data.r
            z_hat = data.z_hat

            stats[group_id] = {
                'n': len(r),
                'mean': np.mean(r),
                'std': np.std(r),
                'skewness': self._compute_skewness(r),
                'kurtosis': self._compute_kurtosis(r),
                'z_hat_range': (np.min(z_hat), np.max(z_hat)),
            }

        return stats

    def check_homoscedasticity(
        self,
        num_bins: int = 10,
    ) -> Dict[int, float]:
        """
        Check if residual variance is approximately constant across ẑ values.

        Returns coefficient of variation of binned variances for each group.
        """
        cv_scores = {}

        for group_id, data in self.calibration:
            z_hat = data.z_hat
            r = data.r

            # Bin by z_hat
            bins = np.percentile(z_hat, np.linspace(0, 100, num_bins + 1))
            bin_vars = []

            for i in range(num_bins):
                mask = (z_hat >= bins[i]) & (z_hat < bins[i + 1])
                if mask.sum() > 10:
                    bin_vars.append(np.var(r[mask]))

            if len(bin_vars) >= 2:
                bin_vars = np.array(bin_vars)
                cv = np.std(bin_vars) / (np.mean(bin_vars) + 1e-8)
                cv_scores[group_id] = cv
            else:
                cv_scores[group_id] = np.nan

        return cv_scores

    def check_normality(
        self,
        alpha: float = 0.05,
    ) -> Dict[int, Dict]:
        """
        Check if standardized residuals are approximately Gaussian.

        Uses Jarque-Bera test and quantile-quantile comparison.
        """
        try:
            from scipy import stats
            has_scipy = True
        except ImportError:
            has_scipy = False

        results = {}

        for group_id, data in self.calibration:
            u = data.get_standardized()

            result = {
                'skewness': self._compute_skewness(u),
                'kurtosis': self._compute_kurtosis(u),
            }

            if has_scipy and len(u) > 20:
                # Jarque-Bera test
                jb_stat, jb_pval = stats.jarque_bera(u)
                result['jb_stat'] = jb_stat
                result['jb_pval'] = jb_pval
                result['is_normal'] = jb_pval > alpha

                # Shapiro-Wilk on subsample (for large datasets)
                if len(u) > 5000:
                    u_sample = np.random.choice(u, 5000, replace=False)
                else:
                    u_sample = u
                sw_stat, sw_pval = stats.shapiro(u_sample)
                result['sw_stat'] = sw_stat
                result['sw_pval'] = sw_pval

            results[group_id] = result

        return results

    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness."""
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return np.mean(((x - mean) / std) ** 3)

    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return np.mean(((x - mean) / std) ** 4) - 3

    def get_diagnostic_report(self) -> str:
        """Generate text diagnostic report."""
        lines = ["=" * 60, "CALIBRATION RESIDUAL DIAGNOSTIC REPORT", "=" * 60, ""]

        stats = self.compute_statistics()
        homo_scores = self.check_homoscedasticity()
        normality = self.check_normality()

        for group_id in self.calibration.get_group_ids():
            s = stats[group_id]
            lines.append(f"Group {group_id}:")
            lines.append(f"  Samples: {s['n']}")
            lines.append(f"  Mean: {s['mean']:.4f}")
            lines.append(f"  Std: {s['std']:.4f}")
            lines.append(f"  Skewness: {s['skewness']:.4f}")
            lines.append(f"  Kurtosis: {s['kurtosis']:.4f}")

            if group_id in homo_scores:
                cv = homo_scores[group_id]
                status = "PASS" if cv < 0.3 else "WARN" if cv < 0.5 else "FAIL"
                lines.append(f"  Homoscedasticity CV: {cv:.4f} [{status}]")

            if group_id in normality:
                n = normality[group_id]
                if 'jb_pval' in n:
                    status = "PASS" if n['is_normal'] else "FAIL"
                    lines.append(f"  Normality (JB p-value): {n['jb_pval']:.4f} [{status}]")

            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
