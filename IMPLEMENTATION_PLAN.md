# LearnNoiseVST Implementation Plan

## Overview

A domain-agnostic framework for learnable variance-stabilizing transforms (VST) and noise characterization, designed for noise-aligned data augmentation in self-supervised learning.

**Core Components:**
1. **Transform T**: Per-feature monotone Rational Quadratic Spline (RQS) with gauge-fixing
2. **Denoiser D**: Blind-spot architecture (imaging: rotation-based CNN, tabular: leave-one-out pooling)
3. **Alternating Optimization**: Joint training of T and D
4. **Noise Characterization**: Post-training calibration for sampling

---

## Project Structure

```
LearnNoiseVST/
├── src/
│   ├── __init__.py
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── rqs.py                 # Rational Quadratic Spline implementation
│   │   ├── monotone_transform.py  # Per-feature transform with gauge-fixing
│   │   └── utils.py               # Spline utilities, derivative bounds
│   ├── denoisers/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract blind-spot denoiser interface
│   │   ├── imaging/
│   │   │   ├── __init__.py
│   │   │   ├── blind_spot_conv.py # Upward-only convolution layers
│   │   │   ├── rotation_net.py    # 4-rotation blind-spot network
│   │   │   └── unet_backbone.py   # U-Net with restricted receptive field
│   │   └── tabular/
│   │       ├── __init__.py
│   │       ├── loo_pooling.py     # Leave-one-out pooling module
│   │       └── tabular_denoiser.py# Full tabular blind-spot network
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py              # L_homo, L_shape, L_reg, L_prox, J[T]
│   │   ├── alternating_trainer.py # Main training loop
│   │   ├── gauge_fixing.py        # Running statistics for normalization
│   │   └── diagnostics.py         # Convergence monitoring, leakage checks
│   ├── noise_model/
│   │   ├── __init__.py
│   │   ├── calibration.py         # Residual dataset generation
│   │   ├── location_scale.py      # μ_g(ẑ), σ_g(ẑ) fitting
│   │   ├── marginal_fit.py        # Empirical CDF, quantile tables
│   │   ├── copula.py              # Gaussian copula for dependence
│   │   └── sampler.py             # Noise sampling interface
│   └── utils/
│       ├── __init__.py
│       ├── data.py                # Data loading, splitting
│       └── metrics.py             # Evaluation metrics
├── experiments/
│   ├── synthetic/
│   │   ├── generate_data.py       # Synthetic data with known noise
│   │   └── run_synthetic.py       # Synthetic experiments
│   ├── imaging/
│   │   └── run_imaging.py         # Image denoising experiments
│   └── tabular/
│       └── run_tabular.py         # Tabular experiments
├── tests/
│   ├── test_rqs.py                # RQS invertibility, monotonicity
│   ├── test_blind_spot.py         # Leakage-free verification
│   ├── test_losses.py             # Loss function tests
│   └── test_noise_sampler.py      # Sampler correctness
├── configs/
│   ├── default.yaml               # Default hyperparameters
│   ├── imaging.yaml               # Imaging-specific config
│   └── tabular.yaml               # Tabular-specific config
├── requirements.txt
├── setup.py
└── README.md
```

---

## Phase 1: Core Transform Module

### 1.1 Rational Quadratic Spline (RQS) Implementation

**File**: `src/transforms/rqs.py`

**Key Features:**
- Monotonic by construction (positive derivatives via softplus parameterization)
- Analytic forward, inverse, and log-derivative
- Linear tails outside data range
- Configurable number of bins (K = 8 to 64)

**Implementation Details:**

```python
class RationalQuadraticSpline:
    """
    Single-feature RQS transform.

    Parameters:
        - K: number of bins (interior knots = K+1)
        - bound: data range [-bound, bound] for spline, linear outside
        - min_derivative: minimum derivative (for stability)

    Learnable parameters (for K bins):
        - widths: K values (softmax → sum to 2*bound)
        - heights: K values (softmax → sum to 2*bound)
        - derivatives: K+1 values (softplus → positive)
    """
```

**Key Functions:**
- `forward(x)`: Apply spline, returns (y, log_det)
- `inverse(y)`: Analytic inverse
- `derivative(x)`: T'(x) for delta-method calculations

**Tests:**
- Invertibility: `x ≈ inverse(forward(x))` for random x
- Monotonicity: `derivative(x) > 0` everywhere
- Continuity at tail boundaries

### 1.2 Per-Feature Transform with Gauge-Fixing

**File**: `src/transforms/monotone_transform.py`

```python
class MonotoneFeatureTransform:
    """
    Transform for d features, each with its own RQS.
    Includes gauge-fixing to ensure E[z_f] = 0, Var(z_f) = 1.

    Parameters:
        - num_features: d
        - num_bins: K per feature
        - tail_quantiles: (q_low, q_high) for determining linear tail bounds
    """

    def __init__(self, num_features, num_bins=16, tail_quantiles=(0.001, 0.999)):
        self.splines = nn.ModuleList([RQS(num_bins) for _ in range(num_features)])
        # Gauge-fixing stats (not learned, computed from data)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = 0.1

    def forward(self, x, update_stats=True):
        # x: [B, d] or [B, C, H, W]
        # Apply per-feature spline
        # Update running stats if training
        # Return gauge-fixed z

    def inverse(self, z):
        # Un-normalize then invert spline
```

**Gauge-Fixing Strategy:**
- During training: update running mean/var with momentum (EMA)
- Periodically (every N batches): full-pass recomputation on train set
- At inference: use frozen stats

---

## Phase 2: Blind-Spot Denoiser Architectures

### 2.1 Imaging: Rotation-Based Blind-Spot Network

**File**: `src/denoisers/imaging/blind_spot_conv.py`

```python
class UpwardOnlyConv2d(nn.Module):
    """
    Conv2d that only looks upward (restricted receptive field).
    Implemented via padding top + cropping bottom.
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=0, **kwargs)  # No automatic padding
        self.k = kernel_size // 2  # Offset amount

    def forward(self, x):
        # Pad top with k zeros
        x_padded = F.pad(x, (self.k, self.k, self.k, 0))  # (left, right, top, bottom)
        y = self.conv(x_padded)
        # Crop bottom k rows
        return y[:, :, :-self.k, :] if self.k > 0 else y


class UpwardOnlyDownsample(nn.Module):
    """Downsampling that preserves upward-only receptive field."""

class ShiftDown(nn.Module):
    """Shift feature maps down by 1 pixel (final blind-spot offset)."""
```

**File**: `src/denoisers/imaging/rotation_net.py`

```python
class RotationBlindSpotNet(nn.Module):
    """
    Full blind-spot network using 4 rotations + shared upward-only branch.

    Architecture:
        1. ROTATE: Stack 4 rotated versions on batch axis
        2. BranchUp: U-Net with upward-only convolutions
        3. SHIFT: Shift down by 1 pixel
        4. UNROTATE: Split batch, undo rotations, stack on channel axis
        5. FUSE: 1x1 convolutions to combine
    """

    def forward(self, z):
        B, C, H, W = z.shape

        # 1. Create 4 rotations
        z_rot = torch.cat([
            z,                              # 0°
            torch.rot90(z, 1, [2, 3]),      # 90°
            torch.rot90(z, 2, [2, 3]),      # 180°
            torch.rot90(z, 3, [2, 3]),      # 270°
        ], dim=0)  # [4B, C, H, W]

        # 2. Upward-only U-Net
        y_rot = self.branch_up(z_rot)  # [4B, C', H, W]

        # 3. Shift down
        y_rot = self.shift_down(y_rot)

        # 4. Unrotate and stack
        y0, y90, y180, y270 = y_rot.chunk(4, dim=0)
        y_fused = torch.cat([
            y0,
            torch.rot90(y90, -1, [2, 3]),
            torch.rot90(y180, -2, [2, 3]),
            torch.rot90(y270, -3, [2, 3]),
        ], dim=1)  # [B, 4C', H, W]

        # 5. Fuse with 1x1 convs
        z_hat = self.fuse(y_fused)  # [B, C, H, W]
        return z_hat
```

### 2.2 Tabular: Leave-One-Out Pooling Network

**File**: `src/denoisers/tabular/loo_pooling.py`

```python
class LeaveOneOutPooling(nn.Module):
    """
    Efficient leave-one-out mean pooling.

    Given embeddings e ∈ [B, d, H]:
        S = sum(e, dim=1)  # [B, H]
        n = d (or sum of masks)
        c_f = (S - e_f) / (n - 1)  # [B, d, H]
    """

    def forward(self, e, mask=None):
        # e: [B, d, H]
        # mask: [B, d] binary (1 = observed)
        if mask is None:
            mask = torch.ones(e.shape[:2], device=e.device)

        # Masked sum
        S = (e * mask.unsqueeze(-1)).sum(dim=1)  # [B, H]
        n = mask.sum(dim=1, keepdim=True)  # [B, 1]

        # Leave-one-out
        S_minus_f = S.unsqueeze(1) - e * mask.unsqueeze(-1)  # [B, d, H]
        n_minus_f = n.unsqueeze(1) - mask.unsqueeze(-1)  # [B, d, 1]

        c = S_minus_f / n_minus_f.clamp(min=1)  # [B, d, H]
        return c
```

**File**: `src/denoisers/tabular/tabular_denoiser.py`

```python
class TabularBlindSpotDenoiser(nn.Module):
    """
    Tabular denoiser using leave-one-out pooling.

    Architecture:
        1. Per-feature embedding: e_f = FiLM(MLP_val(z_f), u_f)
        2. Leave-one-out pooling: c_f = LOO_mean(e)
        3. Prediction head: ẑ_f = <v_f, MLP_ctx(c_f)> + b_f

    Parameters:
        - num_features: d
        - embed_dim: H
        - hidden_dim: MLP hidden size
    """

    def __init__(self, num_features, embed_dim=64, hidden_dim=128):
        super().__init__()
        # Feature ID embeddings
        self.feature_embed_enc = nn.Embedding(num_features, embed_dim)  # u_f
        self.feature_embed_dec = nn.Embedding(num_features, embed_dim)  # v_f

        # Shared value encoder
        self.mlp_val = nn.Sequential(
            nn.Linear(2, hidden_dim),  # (z_f, mask_f)
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # FiLM modulation
        self.film_gamma = nn.Linear(embed_dim, embed_dim)
        self.film_beta = nn.Linear(embed_dim, embed_dim)

        # Leave-one-out pooling
        self.loo_pool = LeaveOneOutPooling()

        # Context decoder
        self.mlp_ctx = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Per-feature bias
        self.bias = nn.Parameter(torch.zeros(num_features))
```

### 2.3 Leakage Verification

**File**: `src/training/diagnostics.py`

```python
def verify_no_leakage(model, z, num_samples=100):
    """
    Verify blind-spot property: ∂ẑ_j/∂z_j ≈ 0 for all j.

    Returns: max |diagonal jacobian element|
    """
    z = z.requires_grad_(True)
    z_hat = model(z)

    max_diag = 0.0
    for _ in range(num_samples):
        # Random position
        if z.dim() == 4:  # Image
            b = torch.randint(z.shape[0], (1,)).item()
            c = torch.randint(z.shape[1], (1,)).item()
            h = torch.randint(z.shape[2], (1,)).item()
            w = torch.randint(z.shape[3], (1,)).item()

            grad = torch.autograd.grad(z_hat[b, c, h, w], z, retain_graph=True)[0]
            diag_elem = grad[b, c, h, w].abs().item()
        else:  # Tabular
            b = torch.randint(z.shape[0], (1,)).item()
            f = torch.randint(z.shape[1], (1,)).item()

            grad = torch.autograd.grad(z_hat[b, f], z, retain_graph=True)[0]
            diag_elem = grad[b, f].abs().item()

        max_diag = max(max_diag, diag_elem)

    return max_diag  # Should be < 1e-6
```

---

## Phase 3: Training Loop and Losses

### 3.1 Loss Functions

**File**: `src/training/losses.py`

```python
class HomoscedasticityLoss(nn.Module):
    """
    L_homo = Σ_g Σ_j Cov(φ_j(ẑ_g), u_g²)²

    Penalizes correlation between predicted signal and residual variance.
    """

    def __init__(self, basis_degree=2):
        super().__init__()
        self.basis_degree = basis_degree  # φ = [ẑ, ẑ², ...]

    def forward(self, z_hat, residuals, groups=None):
        # z_hat: [B, ...], residuals: [B, ...]
        # groups: optional grouping (e.g., channels for images)

        # Standardize residuals per group
        if groups is None:
            groups = [slice(None)]  # Single group

        loss = 0.0
        for g in groups:
            z_hat_g = z_hat[..., g].flatten()
            r_g = residuals[..., g].flatten()

            # Standardized squared residual
            r_var = r_g.var() + 1e-8
            u_sq = (r_g ** 2) / r_var

            # Compute covariance with each basis function
            for j in range(1, self.basis_degree + 1):
                phi_j = z_hat_g ** j
                phi_j = phi_j - phi_j.mean()
                u_sq_centered = u_sq - u_sq.mean()

                cov = (phi_j * u_sq_centered).mean()
                loss = loss + cov ** 2

        return loss


class VarianceFlatnessLoss(nn.Module):
    """
    Soft version of J[T] = Var(log σ²(ẑ))
    Uses kernel-smoothed local variance estimation.
    """

    def __init__(self, bandwidth=0.5, min_var=1e-6):
        super().__init__()
        self.bandwidth = bandwidth
        self.min_var = min_var

    def forward(self, z_hat, residuals):
        z_hat_flat = z_hat.flatten()
        r_flat = residuals.flatten()

        # Kernel weights
        diffs = z_hat_flat.unsqueeze(0) - z_hat_flat.unsqueeze(1)
        weights = torch.exp(-diffs**2 / (2 * self.bandwidth**2))
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Local variance at each point
        r_sq = r_flat ** 2
        local_var = (weights * r_sq.unsqueeze(0)).sum(dim=1)
        local_var = local_var.clamp(min=self.min_var)

        # Variance of log local variance
        log_var = torch.log(local_var)
        J_T = log_var.var()

        return J_T


class ShapePenalty(nn.Module):
    """
    L_shape = Σ_g (skew(u_g)² + α(kurt(u_g) - 3)²)

    Encourages standardized residuals to be Gaussian-like.
    Uses robust estimators to handle outliers.
    """

    def __init__(self, kurt_weight=0.1, use_robust=True):
        super().__init__()
        self.kurt_weight = kurt_weight
        self.use_robust = use_robust

    def forward(self, standardized_residuals):
        u = standardized_residuals.flatten()

        if self.use_robust:
            # Robust skewness via median
            med = u.median()
            mad = (u - med).abs().median()
            skew = ((u - med) / (mad + 1e-8)).mean()

            # Robust kurtosis approximation
            kurt = ((u - med) / (mad + 1e-8)).pow(4).mean()
        else:
            skew = ((u - u.mean()) / (u.std() + 1e-8)).pow(3).mean()
            kurt = ((u - u.mean()) / (u.std() + 1e-8)).pow(4).mean()

        return skew ** 2 + self.kurt_weight * (kurt - 3) ** 2


class TransformRegularization(nn.Module):
    """
    Regularization for transform parameters:
    - Smoothness (second derivative penalty on spline)
    - Derivative bounds (soft penalty if T' outside [min, max])
    - Proximity to previous (trust region)
    """

    def __init__(self, smoothness_weight=0.01, deriv_min=0.1, deriv_max=10.0,
                 proximity_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.deriv_min = deriv_min
        self.deriv_max = deriv_max
        self.proximity_weight = proximity_weight
        self.prev_params = None

    def forward(self, transform, x_samples=None):
        loss = 0.0

        # Smoothness: penalize large changes in derivative
        if x_samples is not None:
            derivs = transform.derivative(x_samples)
            # Finite difference approximation of second derivative
            d2 = (derivs[:, 1:] - derivs[:, :-1]).pow(2).mean()
            loss = loss + self.smoothness_weight * d2

        # Derivative bounds
        if x_samples is not None:
            derivs = transform.derivative(x_samples)
            below_min = F.relu(self.deriv_min - derivs)
            above_max = F.relu(derivs - self.deriv_max)
            loss = loss + (below_min.pow(2) + above_max.pow(2)).mean()

        # Proximity to previous
        if self.prev_params is not None:
            curr_params = torch.cat([p.flatten() for p in transform.parameters()])
            diff = (curr_params - self.prev_params).pow(2).sum()
            loss = loss + self.proximity_weight * diff

        return loss

    def update_prev_params(self, transform):
        self.prev_params = torch.cat(
            [p.detach().clone().flatten() for p in transform.parameters()]
        )
```

### 3.2 Alternating Trainer

**File**: `src/training/alternating_trainer.py`

```python
class AlternatingTrainer:
    """
    Main training loop implementing alternating optimization.

    Outer loop:
        Step A: Train denoiser D on z = T(x) with T frozen
        Step B: Update transform T with D frozen
    """

    def __init__(self, transform, denoiser, config):
        self.T = transform
        self.D = denoiser
        self.config = config

        # Losses
        self.homo_loss = HomoscedasticityLoss(basis_degree=config.basis_degree)
        self.vf_loss = VarianceFlatnessLoss(bandwidth=config.vf_bandwidth)
        self.shape_loss = ShapePenalty(kurt_weight=config.kurt_weight)
        self.reg_loss = TransformRegularization(
            smoothness_weight=config.smoothness_weight,
            proximity_weight=config.proximity_weight
        )

        # Optimizers
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=config.lr_D)
        self.opt_T = torch.optim.Adam(self.T.parameters(), lr=config.lr_T)

        # Diagnostics
        self.history = defaultdict(list)

    def train(self, train_loader, val_loader, num_outer_iters=10):
        for k in range(num_outer_iters):
            print(f"\n=== Outer iteration {k+1}/{num_outer_iters} ===")

            # Step A: Train denoiser
            self._train_denoiser(train_loader, val_loader)

            # Freeze D, unfreeze T
            self._freeze(self.D)
            self._unfreeze(self.T)

            # Step B: Update transform
            self._update_transform(train_loader)

            # Refresh gauge-fixing stats
            self._refresh_gauge_stats(train_loader)

            # Diagnostics
            self._compute_diagnostics(val_loader, k)

            # Check convergence
            if self._check_convergence():
                print("Converged!")
                break

            # Prepare for next iteration
            self.reg_loss.update_prev_params(self.T)
            self._freeze(self.T)
            self._unfreeze(self.D)

    def _train_denoiser(self, train_loader, val_loader):
        """Step A: Train D on z = T(x) with blind-spot MSE."""
        self.D.train()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_D_epochs):
            train_loss = 0.0
            for x, _ in train_loader:
                x = x.to(self.config.device)

                with torch.no_grad():
                    z = self.T(x)

                z_hat = self.D(z)
                loss = F.mse_loss(z_hat, z)  # Valid because D is blind-spot

                self.opt_D.zero_grad()
                loss.backward()
                self.opt_D.step()

                train_loss += loss.item()

            # Validation
            val_loss = self._validate_denoiser(val_loader)

            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

    def _update_transform(self, train_loader):
        """Step B: Update T with frozen D."""
        self.T.train()

        for epoch in range(self.config.T_epochs_per_outer):
            for x, _ in train_loader:
                x = x.to(self.config.device)

                z = self.T(x)

                # Option: stop gradient through D or not
                if self.config.stop_grad_D:
                    with torch.no_grad():
                        z_hat = self.D(z)
                    z_hat = z_hat.detach()
                else:
                    z_hat = self.D(z)

                r = z - z_hat

                # Standardize residuals
                sigma_hat = (r ** 2).mean().sqrt()
                u = r / (sigma_hat + 1e-8)

                # Compute losses
                loss_homo = self.homo_loss(z_hat, r)
                loss_vf = self.vf_loss(z_hat, r)
                loss_shape = self.shape_loss(u)
                loss_reg = self.reg_loss(self.T, x)

                # Total
                loss_T = (
                    self.config.lambda_homo * loss_homo +
                    self.config.lambda_vf * loss_vf +
                    self.config.lambda_shape * loss_shape +
                    self.config.lambda_reg * loss_reg
                )

                self.opt_T.zero_grad()
                loss_T.backward()
                self.opt_T.step()

    def _refresh_gauge_stats(self, train_loader):
        """Recompute running mean/var for gauge-fixing."""
        self.T.eval()

        all_s = []
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(self.config.device)
                # Get pre-normalized spline output
                s = self.T.forward_prenorm(x)
                all_s.append(s)

        all_s = torch.cat(all_s, dim=0)
        self.T.running_mean = all_s.mean(dim=0)
        self.T.running_var = all_s.var(dim=0)

    def _compute_diagnostics(self, val_loader, outer_iter):
        """Compute and store diagnostic metrics."""
        self.T.eval()
        self.D.eval()

        all_z_hat, all_r = [], []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(self.config.device)
                z = self.T(x)
                z_hat = self.D(z)
                r = z - z_hat
                all_z_hat.append(z_hat)
                all_r.append(r)

        z_hat = torch.cat(all_z_hat, dim=0)
        r = torch.cat(all_r, dim=0)

        # 1. Variance flatness (J[T])
        J_T = self.vf_loss(z_hat, r).item()
        self.history['J_T'].append(J_T)

        # 2. Residual statistics
        u = r / (r.std() + 1e-8)
        skew = u.pow(3).mean().item()
        kurt = u.pow(4).mean().item()
        self.history['skewness'].append(skew)
        self.history['kurtosis'].append(kurt)

        # 3. Correlation of variance with signal level
        corr = self._corr_var_signal(z_hat, r)
        self.history['var_signal_corr'].append(corr)

        print(f"  J[T]={J_T:.4f}, skew={skew:.4f}, kurt={kurt:.4f}, corr={corr:.4f}")

    def _check_convergence(self):
        """Check if training has converged."""
        if len(self.history['J_T']) < 3:
            return False

        # Check if J[T] is stable
        recent = self.history['J_T'][-3:]
        return max(recent) - min(recent) < self.config.convergence_threshold
```

---

## Phase 4: Noise Characterization and Sampling

### 4.1 Calibration Residual Generation

**File**: `src/noise_model/calibration.py`

```python
class CalibrationDataset:
    """
    Generate residual dataset from calibration split.

    For each sample x_i:
        z_i = T(x_i)
        ẑ_i = D(z_i)
        r_i = z_i - ẑ_i

    Store tuples: (ẑ, r, group_id)
    """

    def __init__(self, transform, denoiser, margin=8):
        self.T = transform
        self.D = denoiser
        self.margin = margin  # Crop margin for images

    @torch.no_grad()
    def generate(self, dataloader, is_image=False):
        residuals = defaultdict(lambda: {'z_hat': [], 'r': []})

        for x, _ in dataloader:
            z = self.T(x)
            z_hat = self.D(z)
            r = z - z_hat

            if is_image:
                # Crop margins
                m = self.margin
                z_hat = z_hat[:, :, m:-m, m:-m]
                r = r[:, :, m:-m, m:-m]

                # Group by channel
                for c in range(z_hat.shape[1]):
                    residuals[c]['z_hat'].append(z_hat[:, c].flatten())
                    residuals[c]['r'].append(r[:, c].flatten())
            else:
                # Group by feature
                for f in range(z_hat.shape[1]):
                    residuals[f]['z_hat'].append(z_hat[:, f])
                    residuals[f]['r'].append(r[:, f])

        # Concatenate
        for g in residuals:
            residuals[g]['z_hat'] = torch.cat(residuals[g]['z_hat'])
            residuals[g]['r'] = torch.cat(residuals[g]['r'])

        return dict(residuals)
```

### 4.2 Location-Scale Fitting

**File**: `src/noise_model/location_scale.py`

```python
class LocationScaleModel:
    """
    Fit r = μ_g(ẑ) + σ_g(ẑ) * u

    μ_g: bias function (ideally ~0)
    σ_g: scale function (ideally constant if VST worked)
    """

    def __init__(self, num_knots=8):
        self.num_knots = num_knots
        self.mu_spline = None
        self.log_sigma_spline = None

    def fit(self, z_hat, r):
        """Fit location and scale functions."""
        z_hat_np = z_hat.numpy()
        r_np = r.numpy()

        # 1. Fit μ(ẑ) robustly
        self.mu_spline = self._fit_robust_spline(z_hat_np, r_np)

        # 2. Compute de-meaned residuals
        mu_pred = self.mu_spline(z_hat_np)
        r_tilde = r_np - mu_pred

        # 3. Check if variance is flat
        is_flat = self._test_variance_flatness(z_hat_np, r_tilde)

        if is_flat:
            # Constant σ
            self.sigma_const = np.std(r_tilde)
            self.log_sigma_spline = None
        else:
            # Fit log σ(ẑ)
            log_var = np.log(r_tilde ** 2 + 1e-8)
            self.log_sigma_spline = self._fit_robust_spline(z_hat_np, log_var)
            self.sigma_const = None

    def get_mu(self, z_hat):
        return self.mu_spline(z_hat)

    def get_sigma(self, z_hat):
        if self.sigma_const is not None:
            return np.full_like(z_hat, self.sigma_const)
        else:
            return np.exp(0.5 * self.log_sigma_spline(z_hat))

    def standardize(self, z_hat, r):
        """Return standardized residuals u."""
        mu = self.get_mu(z_hat)
        sigma = self.get_sigma(z_hat)
        return (r - mu) / sigma
```

### 4.3 Marginal Distribution Fitting

**File**: `src/noise_model/marginal_fit.py`

```python
class EmpiricalMarginal:
    """
    Fit and sample from empirical CDF of standardized residuals.
    Uses quantile table for fast inverse CDF sampling.
    """

    def __init__(self, num_quantiles=1023):
        self.K = num_quantiles
        self.quantiles = None  # [K] array

    def fit(self, u):
        """Fit from standardized residual samples."""
        u_sorted = np.sort(u)

        # Compute quantiles
        probs = (np.arange(1, self.K + 1)) / (self.K + 1)
        indices = (probs * len(u_sorted)).astype(int)
        indices = np.clip(indices, 0, len(u_sorted) - 1)

        self.quantiles = u_sorted[indices]

        # Re-standardize to ensure mean=0, std=1
        q_mean = self.quantiles.mean()
        q_std = self.quantiles.std()
        self.quantiles = (self.quantiles - q_mean) / q_std

    def sample(self, shape):
        """Sample via inverse CDF."""
        p = np.random.uniform(0, 1, shape)

        # Linear interpolation in quantile table
        idx_float = p * (self.K - 1)
        idx_low = np.floor(idx_float).astype(int)
        idx_high = np.ceil(idx_float).astype(int)
        frac = idx_float - idx_low

        u = (1 - frac) * self.quantiles[idx_low] + frac * self.quantiles[idx_high]
        return u

    def cdf(self, u):
        """Evaluate CDF at u."""
        return np.searchsorted(self.quantiles, u) / self.K
```

### 4.4 Gaussian Copula for Dependence

**File**: `src/noise_model/copula.py`

```python
class GaussianCopula:
    """
    Model dependence structure via Gaussian copula.

    For image channels or correlated tabular features.
    """

    def __init__(self, shrinkage=0.1):
        self.shrinkage = shrinkage
        self.correlation = None
        self.cholesky = None

    def fit(self, u_matrix, marginals):
        """
        Fit correlation from residuals.

        u_matrix: [N, d] residuals
        marginals: list of EmpiricalMarginal for each dimension
        """
        d = u_matrix.shape[1]

        # Convert to normal scores via probability integral transform
        y = np.zeros_like(u_matrix)
        for j in range(d):
            p = marginals[j].cdf(u_matrix[:, j])
            p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid infinities
            y[:, j] = scipy.stats.norm.ppf(p)

        # Estimate correlation with shrinkage
        corr_raw = np.corrcoef(y.T)
        self.correlation = (1 - self.shrinkage) * corr_raw + self.shrinkage * np.eye(d)

        # Cholesky for sampling
        self.cholesky = np.linalg.cholesky(self.correlation)

    def sample(self, n, marginals):
        """Sample n vectors with fitted correlation structure."""
        d = len(marginals)

        # Sample correlated normals
        z = np.random.randn(n, d)
        y = z @ self.cholesky.T

        # Convert to uniforms
        p = scipy.stats.norm.cdf(y)

        # Apply inverse marginal CDFs
        u = np.zeros_like(p)
        for j in range(d):
            u[:, j] = marginals[j].sample_from_uniform(p[:, j])

        return u
```

### 4.5 Full Noise Sampler

**File**: `src/noise_model/sampler.py`

```python
class NoiseModelSampler:
    """
    Complete noise sampler combining all components.

    Given ẑ, sample ε such that z = ẑ + ε matches true noise.
    """

    def __init__(self):
        self.location_scale = {}  # Per-group LocationScaleModel
        self.marginals = {}       # Per-group EmpiricalMarginal
        self.copula = None        # Optional dependence
        self.use_copula = False

    def fit(self, calibration_data, fit_copula=False):
        """
        Fit from calibration residuals.

        calibration_data: dict {group_id: {'z_hat': tensor, 'r': tensor}}
        """
        for g, data in calibration_data.items():
            z_hat = data['z_hat'].numpy()
            r = data['r'].numpy()

            # Fit location-scale
            self.location_scale[g] = LocationScaleModel()
            self.location_scale[g].fit(z_hat, r)

            # Get standardized residuals
            u = self.location_scale[g].standardize(z_hat, r)

            # Fit marginal
            self.marginals[g] = EmpiricalMarginal()
            self.marginals[g].fit(u)

        # Optionally fit copula for dependence
        if fit_copula and len(calibration_data) > 1:
            self._fit_copula(calibration_data)

    def sample(self, z_hat, groups=None):
        """
        Sample noise given predicted clean signal.

        z_hat: [B, d] or [B, C, H, W]
        Returns: ε with same shape
        """
        if self.use_copula:
            return self._sample_with_copula(z_hat, groups)
        else:
            return self._sample_independent(z_hat, groups)

    def _sample_independent(self, z_hat, groups):
        """Sample independently per coordinate."""
        eps = torch.zeros_like(z_hat)

        for g, model in self.location_scale.items():
            if z_hat.dim() == 4:  # Image: g is channel
                z_hat_g = z_hat[:, g].numpy()
                shape = z_hat_g.shape
            else:  # Tabular: g is feature
                z_hat_g = z_hat[:, g].numpy()
                shape = z_hat_g.shape

            # Sample standardized noise
            u = self.marginals[g].sample(shape)

            # Apply location-scale
            mu = model.get_mu(z_hat_g)
            sigma = model.get_sigma(z_hat_g)
            eps_g = mu + sigma * u

            if z_hat.dim() == 4:
                eps[:, g] = torch.from_numpy(eps_g)
            else:
                eps[:, g] = torch.from_numpy(eps_g)

        return eps
```

---

## Phase 5: Experiments

### 5.1 Synthetic Data Generation

**File**: `experiments/synthetic/generate_data.py`

```python
def generate_poisson_like(n_samples, n_features, signal_range=(1, 100)):
    """
    Generate data with Poisson-like heteroscedastic noise.
    Var(X|μ) = μ (variance proportional to mean).

    True VST: T(x) = 2*sqrt(x) (Anscombe-like)
    """
    # Clean signal
    mu = np.random.uniform(*signal_range, size=(n_samples, n_features))

    # Heteroscedastic noise
    sigma = np.sqrt(mu)
    noise = np.random.randn(n_samples, n_features) * sigma

    x = mu + noise
    x = np.maximum(x, 0.1)  # Ensure positive

    return x, mu, sigma


def generate_multiplicative(n_samples, n_features, signal_range=(1, 10), noise_cv=0.2):
    """
    Generate data with multiplicative noise.
    X = μ * (1 + ε), where ε ~ N(0, cv²)

    True VST: T(x) = log(x)
    """
    mu = np.random.uniform(*signal_range, size=(n_samples, n_features))
    noise_mult = 1 + np.random.randn(n_samples, n_features) * noise_cv
    x = mu * noise_mult
    x = np.maximum(x, 0.1)

    return x, mu, mu * noise_cv


def generate_mixed(n_samples, n_features):
    """
    Mixed: some features Poisson-like, some multiplicative.
    Tests whether per-feature transforms are learned correctly.
    """
    x1, mu1, sigma1 = generate_poisson_like(n_samples, n_features // 2)
    x2, mu2, sigma2 = generate_multiplicative(n_samples, n_features - n_features // 2)

    x = np.concatenate([x1, x2], axis=1)
    mu = np.concatenate([mu1, mu2], axis=1)
    sigma = np.concatenate([sigma1, sigma2], axis=1)

    return x, mu, sigma
```

### 5.2 Evaluation Metrics

**File**: `src/utils/metrics.py`

```python
def variance_flatness_score(z_hat, r, num_bins=20):
    """
    Compute binned variance flatness score.
    Returns coefficient of variation of conditional variances.
    """
    z_hat_flat = z_hat.flatten().numpy()
    r_flat = r.flatten().numpy()

    bins = np.percentile(z_hat_flat, np.linspace(0, 100, num_bins + 1))
    bin_vars = []

    for i in range(num_bins):
        mask = (z_hat_flat >= bins[i]) & (z_hat_flat < bins[i + 1])
        if mask.sum() > 10:
            bin_vars.append(r_flat[mask].var())

    bin_vars = np.array(bin_vars)
    return bin_vars.std() / bin_vars.mean()  # CV of variances


def transform_quality_metrics(T, x, mu_true, sigma_true):
    """
    Compare learned transform to ground truth (for synthetic data).
    """
    z = T(torch.from_numpy(x).float())

    # Compute empirical variance at each mu level
    # Compare to theoretical flat variance
    pass


def noise_sampler_two_sample_test(sampler, calibration_data, n_samples=1000):
    """
    Two-sample test: compare real residuals vs sampled residuals.
    Uses Kolmogorov-Smirnov test per group.
    """
    results = {}

    for g, data in calibration_data.items():
        z_hat = data['z_hat'][:n_samples]
        r_real = data['r'][:n_samples].numpy()

        # Sample from model
        r_sampled = sampler.sample(z_hat.unsqueeze(0), groups=[g])
        r_sampled = r_sampled[0, g].numpy()

        # KS test
        stat, pval = scipy.stats.ks_2samp(r_real, r_sampled)
        results[g] = {'ks_stat': stat, 'p_value': pval}

    return results
```

---

## Phase 6: Configuration and Hyperparameters

### Default Configuration

**File**: `configs/default.yaml`

```yaml
# Transform
transform:
  num_bins: 16
  tail_quantiles: [0.001, 0.999]
  deriv_min: 0.1
  deriv_max: 10.0

# Denoiser
denoiser:
  type: "tabular"  # or "imaging"
  embed_dim: 64
  hidden_dim: 128
  # Imaging-specific
  base_channels: 32
  depth: 4

# Training
training:
  lr_T: 1e-3
  lr_D: 1e-3
  max_D_epochs: 50
  T_epochs_per_outer: 10
  num_outer_iters: 20
  patience: 10
  min_delta: 1e-4
  batch_size: 256

  # Loss weights
  lambda_homo: 1.0
  lambda_vf: 0.5
  lambda_shape: 0.1
  lambda_reg: 0.01

  # Options
  stop_grad_D: false
  basis_degree: 2
  vf_bandwidth: 0.5
  convergence_threshold: 0.01

# Noise model
noise_model:
  num_quantiles: 1023
  shrinkage: 0.1
  fit_copula: true

# Data
data:
  train_split: 0.7
  val_split: 0.15
  calibration_split: 0.15
```

---

## Implementation Timeline (Phases, Not Dates)

### Phase 1: Core Modules (Foundation)
1. Implement RQS with tests for invertibility/monotonicity
2. Implement MonotoneFeatureTransform with gauge-fixing
3. Unit tests for all transform operations

### Phase 2: Denoisers
1. Implement tabular blind-spot denoiser (simpler, for initial testing)
2. Implement imaging blind-spot denoiser
3. Leakage verification tests for both

### Phase 3: Training Loop
1. Implement all loss functions with unit tests
2. Implement AlternatingTrainer
3. Diagnostic tracking and visualization

### Phase 4: Noise Model
1. Calibration residual generation
2. Location-scale fitting
3. Marginal fitting and sampling
4. Copula (optional dependence)

### Phase 5: Synthetic Experiments
1. Generate synthetic datasets with known ground truth
2. Run full pipeline, compare learned T to oracle
3. Ablation studies on loss components

### Phase 6: Real Data and Downstream
1. Image denoising benchmarks
2. Tabular data experiments
3. SSL augmentation experiments (if applicable)

---

## Key Implementation Notes

### Critical Details to Get Right

1. **RQS Numerical Stability**
   - Use log-space for derivative computations
   - Clamp values near knot boundaries
   - Test with extreme inputs

2. **Gauge-Fixing Timing**
   - Update running stats with momentum during batches
   - Full recomputation only at end of outer iteration
   - Never update stats during T gradient computation

3. **Blind-Spot Leakage**
   - Test with autodiff Jacobian check before training
   - Watch for batch norm / layer norm leaking info across positions
   - Skip connections must not include center pixel

4. **Alternating Optimization Stability**
   - Always warm-start D from previous iteration
   - Use trust-region (proximity penalty) for T
   - Monitor for oscillations in metrics

5. **Noise Model Fitting**
   - Use robust estimators (median, MAD) for location-scale
   - Crop image boundaries for calibration
   - Winsorize quantile tails for small calibration sets

---

## Success Criteria

### Module-Level Tests
- [ ] RQS: `max |x - inverse(forward(x))| < 1e-5`
- [ ] RQS: `min(derivative(x)) > 0` everywhere
- [ ] Blind-spot: `max |∂ẑ_j/∂z_j| < 1e-6`
- [ ] Sampler: KS test p-value > 0.05 vs real residuals

### System-Level Metrics (Synthetic Data)
- [ ] Variance flatness: CV of binned variance < 0.1
- [ ] Residual shape: |skewness| < 0.3, |kurtosis - 3| < 0.5
- [ ] Transform correlation with oracle > 0.95 (for known-truth data)

### Downstream (If Applicable)
- [ ] SSL with learned augmentation outperforms generic Gaussian
- [ ] Approaches oracle-aligned augmentation performance
