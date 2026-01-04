"""
Blind-spot convolution primitives for imaging.

These modules implement convolutions with restricted receptive fields
that only look in one direction (upward), enabling blind-spot denoising
when combined with rotations.

Based on Laine et al. (2019) "High-Quality Self-Supervised Deep Image Denoising"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class UpwardOnlyConv2d(nn.Module):
    """
    2D convolution that only looks upward (restricted receptive field).

    The receptive field is restricted to the half-plane above the current pixel.
    This is achieved by:
        1. Padding the top of the input with zeros
        2. Applying a standard convolution
        3. Cropping the bottom of the output

    Note: The center row IS included in the receptive field. To achieve
    true blind-spot (excluding center pixel), use ShiftDown after this layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel (int or tuple).
        stride: Stride of the convolution.
        dilation: Dilation of the convolution.
        groups: Groups for grouped convolution.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Compute padding needed
        # For upward-only: pad top, not bottom; pad left and right equally
        self.pad_h = (kernel_size[0] - 1) * dilation  # Full height padding at top
        self.pad_w = ((kernel_size[1] - 1) * dilation) // 2  # Half width padding each side

        # Standard convolution with no automatic padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply upward-only convolution.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            y: Output tensor [B, C_out, H', W'] where H' and W' depend on stride.
        """
        # Pad: (left, right, top, bottom)
        # For upward-only: pad top fully, no bottom padding
        x_padded = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, 0), mode='constant', value=0)

        # Apply convolution
        y = self.conv(x_padded)

        return y


class UpwardOnlyConv2dSame(nn.Module):
    """
    Upward-only convolution that maintains spatial dimensions (same padding).

    This is useful for encoder/decoder architectures where we want to
    preserve spatial size before explicit downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.dilation = dilation

        # For same-size output with upward-only:
        # - Pad top with (k-1)*d rows
        # - Pad left/right with (k-1)*d//2 columns each
        self.pad_h = (kernel_size[0] - 1) * dilation
        self.pad_w = ((kernel_size[1] - 1) * dilation) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad top and sides
        x_padded = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, 0), mode='constant', value=0)
        y = self.conv(x_padded)

        # The output should have same H, W as input
        # Due to upward-only padding, we may need to crop
        h_out = x.shape[2]
        w_out = x.shape[3]

        # Crop to match input size if needed
        y = y[:, :, :h_out, :w_out]

        return y


class ShiftDown(nn.Module):
    """
    Shift feature maps down by a specified number of pixels.

    This is the final step to achieve blind-spot property: after the upward-only
    branch processes the input, shifting down by 1 pixel ensures that the
    output at position (i, j) only depends on inputs above row i.

    Args:
        shift: Number of pixels to shift down (default: 1).
    """

    def __init__(self, shift: int = 1):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift feature maps down.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            y: Shifted tensor [B, C, H, W] (padded at top, cropped at bottom).
        """
        if self.shift == 0:
            return x

        # Pad top with zeros, crop bottom
        # This shifts content down
        y = F.pad(x, (0, 0, self.shift, 0), mode='constant', value=0)
        y = y[:, :, :-self.shift, :]

        return y


class UpwardOnlyMaxPool2d(nn.Module):
    """
    Max pooling that preserves upward-only receptive field.

    Standard pooling can leak information from below. This version
    pads the top before pooling to maintain the upward-only property.
    """

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

        # For upward-only: pad top with (kernel_size - 1) rows
        self.pad_h = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad top
        x_padded = F.pad(x, (0, 0, self.pad_h, 0), mode='constant', value=float('-inf'))

        # Apply max pooling
        y = F.max_pool2d(x_padded, self.kernel_size, self.stride)

        # Crop if needed to get correct output size
        expected_h = (x.shape[2] + self.stride - 1) // self.stride
        y = y[:, :, :expected_h, :]

        return y


class UpwardOnlyAvgPool2d(nn.Module):
    """
    Average pooling that preserves upward-only receptive field.
    """

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.pad_h = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad top with zeros
        x_padded = F.pad(x, (0, 0, self.pad_h, 0), mode='constant', value=0)

        # For average pooling, we need to account for the padded zeros
        # Use a mask to compute proper averages
        ones = torch.ones_like(x)
        ones_padded = F.pad(ones, (0, 0, self.pad_h, 0), mode='constant', value=0)

        # Sum pooling
        sum_pooled = F.avg_pool2d(x_padded, self.kernel_size, self.stride) * (self.kernel_size ** 2)
        count_pooled = F.avg_pool2d(ones_padded, self.kernel_size, self.stride) * (self.kernel_size ** 2)

        # Proper average
        y = sum_pooled / (count_pooled + 1e-8)

        expected_h = (x.shape[2] + self.stride - 1) // self.stride
        y = y[:, :, :expected_h, :]

        return y


class UpwardOnlyUpsample(nn.Module):
    """
    Upsampling that preserves upward-only receptive field.

    Uses nearest-neighbor or bilinear upsampling, which doesn't
    expand the receptive field downward.
    """

    def __init__(self, scale_factor: int = 2, mode: str = 'nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class BlindSpotConvBlock(nn.Module):
    """
    A convolution block for blind-spot networks.

    Consists of: UpwardOnlyConv -> BatchNorm -> ReLU

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        use_bn: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()

        self.conv = UpwardOnlyConv2dSame(
            in_channels, out_channels, kernel_size
        )

        # Note: BatchNorm is safe for blind-spot as it normalizes over batch,
        # not spatially in a way that would leak center pixel info
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'none':
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Rotate90(nn.Module):
    """
    Rotate tensor by 90 degrees counter-clockwise k times.
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k % 4  # Normalize to 0-3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            return x
        return torch.rot90(x, self.k, dims=[2, 3])


class UnRotate90(nn.Module):
    """
    Undo rotation by k*90 degrees (rotate by -k*90 = (4-k)*90).
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k % 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            return x
        return torch.rot90(x, -self.k, dims=[2, 3])
