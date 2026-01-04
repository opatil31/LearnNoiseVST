"""
Base class for blind-spot denoisers.

All denoisers must satisfy the blind-spot property:
    ∂ẑ_j/∂z_j = 0 for all j

This ensures that when predicting coordinate j, the network
cannot see the input at coordinate j (self-supervised by construction).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BlindSpotDenoiser(ABC, nn.Module):
    """
    Abstract base class for blind-spot denoisers.

    Subclasses must implement:
        - forward(z): Returns denoised ẑ with blind-spot property
        - verify_no_leakage(): Checks that ∂ẑ_j/∂z_j ≈ 0
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Denoise input while maintaining blind-spot property.

        Args:
            z: Input tensor (transformed data).

        Returns:
            z_hat: Denoised output where ẑ_j is independent of z_j.
        """
        pass

    def verify_no_leakage(
        self,
        z: torch.Tensor,
        num_samples: int = 100,
        threshold: float = 1e-5
    ) -> Tuple[bool, float]:
        """
        Verify the blind-spot property by checking diagonal Jacobian elements.

        Args:
            z: Sample input tensor.
            num_samples: Number of diagonal elements to check.
            threshold: Maximum allowed diagonal value.

        Returns:
            (passed, max_diagonal): Whether test passed and max diagonal value.
        """
        z = z.clone().requires_grad_(True)
        z_hat = self.forward(z)

        max_diag = 0.0

        for _ in range(num_samples):
            # Sample random position
            idx = self._sample_position(z)

            # Compute gradient of output at idx w.r.t. input
            z_hat_flat = z_hat.flatten()
            z_flat = z.flatten()

            flat_idx = self._position_to_flat_idx(idx, z.shape)
            if flat_idx >= z_hat_flat.shape[0]:
                continue

            grad = torch.autograd.grad(
                z_hat_flat[flat_idx],
                z,
                retain_graph=True,
                create_graph=False
            )[0]

            # Check diagonal element
            diag_elem = grad.flatten()[flat_idx].abs().item()
            max_diag = max(max_diag, diag_elem)

        passed = max_diag < threshold
        return passed, max_diag

    def _sample_position(self, z: torch.Tensor) -> Tuple[int, ...]:
        """Sample a random position in the tensor."""
        return tuple(torch.randint(0, s, (1,)).item() for s in z.shape)

    def _position_to_flat_idx(self, pos: Tuple[int, ...], shape: torch.Size) -> int:
        """Convert multi-dimensional position to flat index."""
        idx = 0
        multiplier = 1
        for p, s in zip(reversed(pos), reversed(shape)):
            idx += p * multiplier
            multiplier *= s
        return idx


class ImageBlindSpotDenoiser(BlindSpotDenoiser):
    """
    Base class for image blind-spot denoisers.

    Input shape: [B, C, H, W]
    Output shape: [B, C, H, W]
    """

    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

    def _sample_position(self, z: torch.Tensor) -> Tuple[int, ...]:
        """Sample random (batch, channel, height, width) position."""
        b = torch.randint(0, z.shape[0], (1,)).item()
        c = torch.randint(0, z.shape[1], (1,)).item()
        h = torch.randint(0, z.shape[2], (1,)).item()
        w = torch.randint(0, z.shape[3], (1,)).item()
        return (b, c, h, w)


class TabularBlindSpotDenoiser(BlindSpotDenoiser):
    """
    Base class for tabular blind-spot denoisers.

    Input shape: [B, F] where F is number of features
    Output shape: [B, F]
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def _sample_position(self, z: torch.Tensor) -> Tuple[int, ...]:
        """Sample random (batch, feature) position."""
        b = torch.randint(0, z.shape[0], (1,)).item()
        f = torch.randint(0, z.shape[1], (1,)).item()
        return (b, f)
