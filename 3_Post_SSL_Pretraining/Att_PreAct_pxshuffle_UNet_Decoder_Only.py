
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(model) -> None:
    for m in model.modules():
        # print(m)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            '''
            # Do not initialize bias (due to batchnorm)-
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            '''
        elif isinstance(m, nn.BatchNorm2d):
            # Standard initialization for batch normalization-
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    return None

def icnr_init(weight, upscale_factor=2):
    """
    ICNR initialization for PixelShuffle.
    Works with any conv where out_channels is divisible by upscale_factor^2.
    Pass the weight tensor directly (e.g., conv.weight)
    """
    out_ch, in_ch, h, w = weight.shape
    scale_sq = upscale_factor ** 2
    
    out_ch_after_shuffle = out_ch // scale_sq
    
    subkernel = torch.zeros(out_ch_after_shuffle, in_ch, h, w)
    nn.init.kaiming_normal_(subkernel)
    
    kernel = subkernel.repeat_interleave(scale_sq, dim=0)
    
    with torch.no_grad():
        weight.copy_(kernel)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi  # Element-wise multiplication


class BasicBlock(nn.Module):
    '''
    Pre-activation Residual blocks.
    Identity Mappings in Deep Residual Networks by Kaiming He et al.
    '''
    def __init__(
            self, in_planes:int=32, out_planes:int=64, stride:int=1, dropRate:float=0.0
            ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        # 1. Pre-activation (Do NOT use inplace=True here)
        out = F.leaky_relu(self.bn1(x), negative_slope=0.1)

        # 2. Shortcut path
        # In Pre-act ResNet, if dimensions change, the shortcut is applied
        # to the pre-activated input, not the raw input.
        shortcut = self.convShortcut(out) if not self.equalInOut else x

        # 3. Main path
        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), negative_slope=0.1, inplace=True)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.conv2(out)

        # 4. Residual addition
        return out + shortcut

class EncoderStage(nn.Module):
    """
    Wraps PixelUnshuffle (Downsample) -> PreAct Residual Block.
    Input: [B, C, H, W] -> Output: [B, C*4, H/2, W/2] (roughly)
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        # Unshuffle increases channels by 4x (2^2), so input to block is in_c * 4
        self.block = BasicBlock(in_planes=in_c * 4, out_planes=out_c)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.block(x)
        return x


class Att_PreactRes_UNet_Encoder(nn.Module):
    def __init__(self, in_channels:int=3, features=None):
        super().__init__()

        features = features or [128, 256, 384, 512]

        self.pad_divisor = (2 ** len(features)) # e.g. 2^4 = 16

        self.initial_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False)
        self.bn_initial_conv = nn.BatchNorm2d(features[0])

        self.encoder_stages = nn.ModuleList()
        self.bottleneck_pool = nn.MaxPool2d(2)

        for i in range(len(features) - 1):
            self.encoder_stages.append(EncoderStage(in_c=features[i], out_c=features[i+1]))

        self.bottleneck_block = BasicBlock(in_planes=features[-1], out_planes=features[-1]*2, dropRate=0.2)

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape

        # 1. Dynamic Padding-
        target_h = ((orig_h + self.pad_divisor - 1) // self.pad_divisor) * self.pad_divisor
        target_w = ((orig_w + self.pad_divisor - 1) // self.pad_divisor) * self.pad_divisor
        pad_h, pad_w = target_h - orig_h, target_w - orig_w
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        # 2. Initial Conv
        x = F.leaky_relu(self.bn_initial_conv(self.initial_conv(x)), negative_slope=0.1)

        # 3. Encoder Loop (Down sample)-
        for stage in self.encoder_stages:
            x = stage(x)

        # 4. Bottleneck
        x = self.bottleneck_pool(x)
        x = self.bottleneck_block(x)

        return x


class DecoderStage(nn.Module):
    """
    Wraps the entire decoding step:
    1. Expand & PixelShuffle (Upsample)
    2. Smoothing
    3. Attention Gating
    4. Concatenation
    5. Residual Block

    Wraps Decoder Logic: Expand -> PixelShuffle -> Smooth -> Attention -> Concat -> Block
    """
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        
        # 1. Expand dimensions for PixelShuffle (ICNR target)
        self.expand_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c * 4, kernel_size=1)
        
        # 2. PixelShuffle (Upsample)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        
        # 3. Smoothing Conv (Post-shuffle artifacts)
        self.smooth_conv = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        
        # 4. Attention Block (Gates the skip connection)
        # F_g (gating) = current decoder features (out_c)
        # F_l (local) = skip connection features (skip_c)
        self.attention = Attention_block(F_g=out_c, F_l=skip_c, F_int=out_c // 2)
        
        # 5. Final Block (After concatenation)
        # Input is (Skip_channels + Upsampled_channels) -> Output
        self.final_block = BasicBlock(in_planes=skip_c + out_c, out_planes=out_c)

        # Apply Special Init immediately
        self._init_icnr()

    def _init_icnr(self):
        # We pass the WEIGHT TENSOR, not the layer, to the init function
        icnr_init(self.expand_conv.weight, upscale_factor=2)
        # Optional: Initialize bias to 0
        if self.expand_conv.bias is not None:
            nn.init.constant_(self.expand_conv.bias, 0)

    def forward(self, x, skip):
        # x: Low res, High semantics
        # skip: High res, Low semantics
        
        # 1. Upsample
        x = self.expand_conv(x)
        x = self.pixel_shuffle(x)
        x = self.smooth_conv(x)
        
        # 2. Handle Dimension Mismatch (Robustness)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # 3. Attention Gating on Skip Connection
        # g=x (gating signal), x=skip (candidate features)
        gated_skip = self.attention(g=x, x=skip)
        
        # 4. Concatenate
        x = torch.cat([gated_skip, x], dim=1)
        
        # 5. Process
        x = self.final_block(x)
        return x


class Att_PreActRes_UNet_Pxshuffle_DecoderOnly(nn.Module):
    def __init__(self, in_channels=18, out_channels=3, features=None):
        super().__init__()
        features = features or [64, 128, 256, 512]
 
        self.trained_encoder = Att_PreactRes_UNet_Encoder(in_channels=in_channels, features=features)

        self.decoder_stages = nn.ModuleList()

        # Final output
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # 3. Build Decoder
        # We need to look at features in reverse.
        # decoder input channels = features[i] * 2 (because of bottleneck doubling)
        # skip channels = features[i] (from the encoder)
        reversed_feats = features[::-1]

        for i in range(len(reversed_feats)):
            # Input to this stage is the output of the previous level (or bottleneck)
            # Bottleneck output is features[-1]*2.
            # DecoderStage(in_c, skip_c, out_c)

            # Logic:
            # Stage 0: In=1024, Skip=512, Out=512
            # Stage 1: In=512,  Skip=384, Out=384
            # Stage 2: In=384,  Skip=256, Out=256
            # Stage 3: In=256,  Skip=128, Out=128

            if i == 0:
                in_c = reversed_feats[i] * 2  # Bottleneck output channels
            else:
                in_c = reversed_feats[i - 1]  # Previous decoder stage output
            skip_c = reversed_feats[i]
            out_c = reversed_feats[i]
            
            self.decoder_stages.append(
                DecoderStage(in_c=in_c, skip_c=skip_c, out_c=out_c)
            )

        # Calculate padding requirement dynamically
        self.pad_divisor = (2 ** len(features)) # e.g. 2^4 = 16

    def forward(self, x):

        _, _, orig_h, orig_w = x.shape
                
        # 1. Dynamic Padding
        target_h = ((orig_h + self.pad_divisor - 1) // self.pad_divisor) * self.pad_divisor
        target_w = ((orig_w + self.pad_divisor - 1) // self.pad_divisor) * self.pad_divisor
        pad_h, pad_w = target_h - orig_h, target_w - orig_w
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        # 2. Initial Conv
        x = F.leaky_relu(self.trained_encoder.bn_initial_conv(self.trained_encoder.initial_conv(x)), negative_slope=0.1)

        skip_connections = [x] # Save First Skip

        # 3. Encoder Loop (Down sample)-
        for stage in self.trained_encoder.encoder_stages:
            x = stage(x)
            skip_connections.append(x)

        # 4. Bottleneck
        x = self.trained_encoder.bottleneck_pool(x)
        x = self.trained_encoder.bottleneck_block(x)

        # 5. Decoder Loop
        # We pop skip connections from the end of the list
        for stage in self.decoder_stages:
            skip = skip_connections.pop()
            x = stage(x, skip)

        # 6. Final Output & Crop
        x = self.out_conv(x)

        crop_top, crop_left = pad_h // 2, pad_w // 2
        x = x[:, :, crop_top:crop_top + orig_h, crop_left:crop_left + orig_w]

        return torch.sigmoid(x)


'''
# Encoder hyper-params-
num_phase_shifts = 4
in_channels = num_phase_shifts * 2 * 3
out_channels = 3
features = [128, 256, 384, 512]


trained_model_decoder_only = Att_PreActRes_UNet_Pxshuffle_DecoderOnly(
    in_channels=in_channels, out_channels=out_channels,
    features=features
)

path_sav_outputs = '/graphics/scratch3/staff/majumdar/CVPR_Work/Attention_UNet_SSL_files/'
filename = 'AttPreactUNet_Applegreen_allviews_ImageNet_dataaug_SimSiampretrain_DecoderOnly.pth'

trained_model_decoder_only.load_state_dict(torch.load(path_sav_outputs + filename, weights_only=True))
trained_model_decoder_only.eval()
trained_model_decoder_only = trained_model_decoder_only.cuda()

tot_params = sum(p.numel() for p in trained_model_decoder_only.parameters())
print(f'\nTotal number of trainble parameters = {tot_params}\n')
'''

