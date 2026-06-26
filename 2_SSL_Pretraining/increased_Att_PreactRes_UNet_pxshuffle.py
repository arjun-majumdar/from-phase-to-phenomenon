import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    '''
    Pre-activation residual block.
    Reference: "Identity Mappings in Deep Residual Networks", He et al.
    '''
    def __init__(
            self, in_planes:int=32, out_planes:int=64, stride:int=1, dropRate:float=0.0
            ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        # Pre-activation (must not be in-place: the result feeds the shortcut).
        out = F.leaky_relu(self.bn1(x), negative_slope=0.1)

        # When dimensions change, the shortcut is applied to the pre-activated
        # input rather than the raw input.
        shortcut = self.convShortcut(out) if not self.equalInOut else x

        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), negative_slope=0.1, inplace=True)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.conv2(out)

        return out + shortcut

class EncoderStage(nn.Module):
    """PixelUnshuffle downsample followed by a pre-activation residual block.

    Input [B, C, H, W] -> output [B, out_c, H/2, W/2].
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        # Unshuffle increases channels by 4x (2^2), so the block input is in_c * 4.
        self.block = BasicBlock(in_planes=in_c * 4, out_planes=out_c)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.block(x)
        return x


class Att_PreactRes_UNet_Encoder(nn.Module):
    def __init__(self, in_channels:int=3, features=None):
        super().__init__()

        features = features or [128, 256, 384, 512]

        self.pad_divisor = (2 ** len(features))  # e.g. 2^4 = 16

        self.initial_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False)
        self.bn_initial_conv = nn.BatchNorm2d(features[0])

        self.encoder_stages = nn.ModuleList()
        self.bottleneck_pool = nn.MaxPool2d(2)

        for i in range(len(features) - 1):
            self.encoder_stages.append(EncoderStage(in_c=features[i], out_c=features[i+1]))

        self.bottleneck_block = BasicBlock(in_planes=features[-1], out_planes=features[-1]*2, dropRate=0.2)

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape

        # Dynamically pad so the spatial dims are divisible by the downsampling factor.
        target_h = ((orig_h + self.pad_divisor - 1) // self.pad_divisor) * self.pad_divisor
        target_w = ((orig_w + self.pad_divisor - 1) // self.pad_divisor) * self.pad_divisor
        pad_h, pad_w = target_h - orig_h, target_w - orig_w
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        # Initial conv.
        x = F.leaky_relu(self.bn_initial_conv(self.initial_conv(x)), negative_slope=0.1)

        # Encoder (downsampling stages).
        for stage in self.encoder_stages:
            x = stage(x)

        # Bottleneck.
        x = self.bottleneck_pool(x)
        x = self.bottleneck_block(x)

        return x

