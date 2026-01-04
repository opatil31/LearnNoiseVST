"""
Rotation-based blind-spot network for image denoising.

This implements the architecture from Laine et al. (2019) that achieves
blind-spot denoising through:
1. Processing 4 rotated versions of the input through an upward-only branch
2. Shifting each branch output by 1 pixel
3. Unrotating and fusing the results

The key insight is that by combining 4 directional branches (up, down, left, right),
we can predict each pixel using context from all directions except the pixel itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .blind_spot_conv import ShiftDown
from .unet_backbone import SimpleUpwardOnlyUNet
from ..base import ImageBlindSpotDenoiser


class RotationBlindSpotNet(ImageBlindSpotDenoiser):
    """
    Blind-spot network using 4 rotations + shared upward-only branch.

    Architecture:
        1. ROTATE: Stack 4 rotated versions of input on batch axis
        2. BRANCH_UP: Process through upward-only U-Net (weight sharing)
        3. SHIFT: Shift down by 1 pixel to exclude center
        4. UNROTATE: Split batch, undo rotations, stack on channel axis
        5. FUSE: 1x1 convolutions to combine into final prediction

    This yields an effective 4-branch blind-spot network in one forward pass.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        out_channels: Number of output channels (usually same as in_channels).
        base_features: Number of features in first U-Net layer.
        depth: Number of U-Net encoder/decoder levels.
        fuse_hidden: Hidden channels in fusion network.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        base_features: int = 32,
        depth: int = 4,
        fuse_hidden: int = 64,
    ):
        super().__init__(in_channels, out_channels)

        out_channels = out_channels or in_channels

        # Feature list for U-Net
        features = [base_features * (2 ** i) for i in range(depth)]

        # Shared upward-only U-Net branch
        # Output channels = in_channels (we'll fuse 4 of these)
        self.branch = SimpleUpwardOnlyUNet(
            in_channels=in_channels,
            out_channels=in_channels,  # Each branch outputs same channels as input
            features=features,
            kernel_size=3,
        )

        # Shift down by 1 pixel (critical for blind-spot property)
        self.shift = ShiftDown(shift=1)

        # Fusion network: 4*in_channels -> out_channels
        # Uses 1x1 convolutions (doesn't expand receptive field)
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * in_channels, fuse_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_hidden, fuse_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_hidden, out_channels, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with rotation-based blind-spot processing.

        Args:
            z: Input tensor [B, C, H, W].

        Returns:
            z_hat: Denoised output [B, C, H, W] with blind-spot property.
        """
        B, C, H, W = z.shape

        # 1. ROTATE: Create 4 rotated versions
        # Stack on batch axis: [4*B, C, H, W]
        z_rot = torch.cat([
            z,                              # 0° (upward)
            torch.rot90(z, 1, dims=[2, 3]), # 90° CCW (rightward -> upward)
            torch.rot90(z, 2, dims=[2, 3]), # 180° (downward -> upward)
            torch.rot90(z, 3, dims=[2, 3]), # 270° CCW (leftward -> upward)
        ], dim=0)

        # 2. BRANCH_UP: Process through upward-only U-Net
        # The branch only sees pixels above (due to receptive field restriction)
        y_rot = self.branch(z_rot)  # [4*B, C, H', W']

        # 3. SHIFT: Shift down by 1 pixel
        # This ensures the output at (i,j) doesn't depend on input at (i,j)
        y_rot = self.shift(y_rot)  # [4*B, C, H', W']

        # Handle potential size changes from U-Net
        if y_rot.shape[2:] != (H, W):
            y_rot = F.interpolate(y_rot, size=(H, W), mode='bilinear', align_corners=False)

        # 4. UNROTATE: Split batch and undo rotations
        y_0, y_90, y_180, y_270 = y_rot.chunk(4, dim=0)

        y_fused = torch.cat([
            y_0,                                # Already oriented correctly
            torch.rot90(y_90, -1, dims=[2, 3]), # Undo 90° rotation
            torch.rot90(y_180, -2, dims=[2, 3]), # Undo 180° rotation
            torch.rot90(y_270, -3, dims=[2, 3]), # Undo 270° rotation
        ], dim=1)  # [B, 4*C, H, W]

        # 5. FUSE: Combine with 1x1 convolutions
        z_hat = self.fuse(y_fused)  # [B, C, H, W]

        return z_hat


class LightweightRotationBlindSpotNet(ImageBlindSpotDenoiser):
    """
    Lightweight version of rotation blind-spot network.

    Uses a simpler backbone for faster training/inference.
    Suitable for quick experiments or resource-constrained settings.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        hidden_channels: int = 32,
        num_blocks: int = 4,
    ):
        super().__init__(in_channels, out_channels)

        out_channels = out_channels or in_channels

        # Simple upward-only encoder
        from .blind_spot_conv import UpwardOnlyConv2dSame, BlindSpotConvBlock

        layers = [BlindSpotConvBlock(in_channels, hidden_channels, kernel_size=3)]
        for _ in range(num_blocks - 1):
            layers.append(BlindSpotConvBlock(hidden_channels, hidden_channels, kernel_size=3))
        layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=1))

        self.branch = nn.Sequential(*layers)
        self.shift = ShiftDown(shift=1)

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape

        # Rotate
        z_rot = torch.cat([
            z,
            torch.rot90(z, 1, dims=[2, 3]),
            torch.rot90(z, 2, dims=[2, 3]),
            torch.rot90(z, 3, dims=[2, 3]),
        ], dim=0)

        # Process
        y_rot = self.branch(z_rot)
        y_rot = self.shift(y_rot)

        # Ensure size matches
        if y_rot.shape[2:] != (H, W):
            y_rot = F.interpolate(y_rot, size=(H, W), mode='bilinear', align_corners=False)

        # Unrotate and fuse
        y_0, y_90, y_180, y_270 = y_rot.chunk(4, dim=0)
        y_fused = torch.cat([
            y_0,
            torch.rot90(y_90, -1, dims=[2, 3]),
            torch.rot90(y_180, -2, dims=[2, 3]),
            torch.rot90(y_270, -3, dims=[2, 3]),
        ], dim=1)

        return self.fuse(y_fused)


class MultiScaleRotationBlindSpotNet(ImageBlindSpotDenoiser):
    """
    Multi-scale version with dilated convolutions for larger receptive field.

    Uses multiple dilation rates to capture both local and global context
    while maintaining the blind-spot property.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        base_features: int = 32,
        dilations: List[int] = [1, 2, 4, 8],
    ):
        super().__init__(in_channels, out_channels)

        out_channels = out_channels or in_channels

        from .blind_spot_conv import UpwardOnlyConv2dSame

        # Multi-scale branches (each with different dilation)
        self.branches = nn.ModuleList()
        for dilation in dilations:
            branch = nn.Sequential(
                UpwardOnlyConv2dSame(in_channels, base_features, kernel_size=3, dilation=dilation),
                nn.BatchNorm2d(base_features),
                nn.ReLU(inplace=True),
                UpwardOnlyConv2dSame(base_features, base_features, kernel_size=3, dilation=dilation),
                nn.BatchNorm2d(base_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_features, in_channels, kernel_size=1),
            )
            self.branches.append(branch)

        self.shift = ShiftDown(shift=1)

        # Fuse multi-scale + multi-direction
        num_branches = len(dilations)
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * in_channels * num_branches, base_features * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, out_channels, kernel_size=1),
        )

        self.num_branches = num_branches

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape

        # Rotate
        z_rot = torch.cat([
            z,
            torch.rot90(z, 1, dims=[2, 3]),
            torch.rot90(z, 2, dims=[2, 3]),
            torch.rot90(z, 3, dims=[2, 3]),
        ], dim=0)

        # Process through each multi-scale branch
        branch_outputs = []
        for branch in self.branches:
            y = branch(z_rot)
            y = self.shift(y)
            if y.shape[2:] != (H, W):
                y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            branch_outputs.append(y)

        # Stack branch outputs
        y_rot = torch.cat(branch_outputs, dim=1)  # [4*B, num_branches*C, H, W]

        # Unrotate
        chunks = y_rot.chunk(4, dim=0)
        y_fused = torch.cat([
            chunks[0],
            torch.rot90(chunks[1], -1, dims=[2, 3]),
            torch.rot90(chunks[2], -2, dims=[2, 3]),
            torch.rot90(chunks[3], -3, dims=[2, 3]),
        ], dim=1)  # [B, 4*num_branches*C, H, W]

        return self.fuse(y_fused)
