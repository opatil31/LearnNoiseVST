"""
U-Net backbone with upward-only (restricted) receptive field.

This implements a U-Net architecture where all convolutions only look
upward, enabling blind-spot denoising when combined with rotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .blind_spot_conv import (
    UpwardOnlyConv2dSame,
    BlindSpotConvBlock,
    UpwardOnlyMaxPool2d,
    UpwardOnlyUpsample,
    ShiftDown,
)


class UpwardOnlyEncoderBlock(nn.Module):
    """
    Encoder block with upward-only convolutions.

    Structure: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()

        self.conv1 = BlindSpotConvBlock(
            in_channels, out_channels, kernel_size, use_bn
        )
        self.conv2 = BlindSpotConvBlock(
            out_channels, out_channels, kernel_size, use_bn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpwardOnlyDecoderBlock(nn.Module):
    """
    Decoder block with upward-only convolutions.

    Structure: Upsample -> Concat skip -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()

        self.upsample = UpwardOnlyUpsample(scale_factor=2, mode='nearest')

        # After concat, channels = in_channels + skip_channels
        self.conv1 = BlindSpotConvBlock(
            in_channels + skip_channels, out_channels, kernel_size, use_bn
        )
        self.conv2 = BlindSpotConvBlock(
            out_channels, out_channels, kernel_size, use_bn
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch due to pooling
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpwardOnlyDownsample(nn.Module):
    """
    Downsampling that preserves upward-only receptive field.

    Uses strided convolution with proper padding to ensure
    the receptive field doesn't extend downward.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride

        # Pad top before strided conv
        self.pad_h = stride - 1

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=stride, stride=stride, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad top to maintain upward-only property
        x_padded = F.pad(x, (0, 0, self.pad_h, 0), mode='constant', value=0)
        y = self.conv(x_padded)

        # Crop to expected size
        expected_h = (x.shape[2] + self.stride - 1) // self.stride
        y = y[:, :, :expected_h, :]

        return y


class UpwardOnlyUNet(nn.Module):
    """
    U-Net with upward-only receptive field.

    All convolutions are restricted to look only upward, which when combined
    with 4-rotation processing, enables blind-spot denoising.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        base_channels: Number of channels in first encoder block.
        depth: Number of encoder/decoder levels.
        kernel_size: Convolution kernel size.
        use_bn: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Channel progression: base, 2*base, 4*base, ...
        channels = [base_channels * (2 ** i) for i in range(depth)]

        # Initial convolution
        self.initial_conv = BlindSpotConvBlock(
            in_channels, base_channels, kernel_size, use_bn
        )

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i in range(depth - 1):
            self.encoders.append(
                UpwardOnlyEncoderBlock(
                    channels[i], channels[i + 1], kernel_size, use_bn
                )
            )
            self.downsamplers.append(
                UpwardOnlyDownsample(channels[i + 1], channels[i + 1], stride=2)
            )

        # Bottleneck
        self.bottleneck = UpwardOnlyEncoderBlock(
            channels[-1], channels[-1], kernel_size, use_bn
        )

        # Decoder
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.decoders.append(
                UpwardOnlyDecoderBlock(
                    channels[i + 1], channels[i + 1], channels[i], kernel_size, use_bn
                )
            )

        # Final convolution (1x1, doesn't affect receptive field)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through upward-only U-Net.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            y: Output tensor [B, out_channels, H, W].
        """
        # Initial conv
        x = self.initial_conv(x)

        # Encoder path (save skip connections)
        skips = [x]
        for encoder, downsampler in zip(self.encoders, self.downsamplers):
            x = encoder(x)
            skips.append(x)
            x = downsampler(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (use skip connections)
        for decoder, skip in zip(self.decoders, reversed(skips[1:])):
            x = decoder(x, skip)

        # Final conv to get output channels
        # Need to upsample to match input size if needed
        if x.shape[2:] != skips[0].shape[2:]:
            x = F.interpolate(x, size=skips[0].shape[2:], mode='nearest')

        x = torch.cat([x, skips[0]], dim=1) if x.shape[1] == skips[0].shape[1] else x

        # Actually, let's simplify: just use final conv
        # Handle the channel dimension properly
        if x.shape[1] != self.final_conv.in_channels:
            # Adjust with 1x1 conv
            x = nn.functional.conv2d(
                x,
                torch.ones(self.final_conv.in_channels, x.shape[1], 1, 1, device=x.device) / x.shape[1]
            )

        y = self.final_conv(x)

        return y


class SimpleUpwardOnlyUNet(nn.Module):
    """
    Simplified U-Net with upward-only receptive field.

    A cleaner implementation that's easier to verify for correctness.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
    ):
        super().__init__()

        self.features = features
        self.num_levels = len(features)

        # Encoder
        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for feat in features:
            self.enc_convs.append(nn.Sequential(
                UpwardOnlyConv2dSame(in_ch, feat, kernel_size),
                nn.BatchNorm2d(feat),
                nn.ReLU(inplace=True),
                UpwardOnlyConv2dSame(feat, feat, kernel_size),
                nn.BatchNorm2d(feat),
                nn.ReLU(inplace=True),
            ))
            self.pools.append(UpwardOnlyMaxPool2d(2, 2))
            in_ch = feat

        # Bottleneck
        self.bottleneck = nn.Sequential(
            UpwardOnlyConv2dSame(features[-1], features[-1] * 2, kernel_size),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            UpwardOnlyConv2dSame(features[-1] * 2, features[-1] * 2, kernel_size),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        in_ch = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(nn.Sequential(
                UpwardOnlyUpsample(2, mode='nearest'),
                UpwardOnlyConv2dSame(in_ch, feat, kernel_size),
            ))
            self.dec_convs.append(nn.Sequential(
                UpwardOnlyConv2dSame(feat * 2, feat, kernel_size),
                nn.BatchNorm2d(feat),
                nn.ReLU(inplace=True),
                UpwardOnlyConv2dSame(feat, feat, kernel_size),
                nn.BatchNorm2d(feat),
                nn.ReLU(inplace=True),
            ))
            in_ch = feat

        # Output
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc_features = []
        for enc_conv, pool in zip(self.enc_convs, self.pools):
            x = enc_conv(x)
            enc_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upconv, dec_conv, enc_feat in zip(
            self.upconvs, self.dec_convs, reversed(enc_features)
        ):
            x = upconv(x)

            # Handle size mismatch
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode='nearest')

            x = torch.cat([x, enc_feat], dim=1)
            x = dec_conv(x)

        return self.out_conv(x)
