import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# The encoder backbone is defined inline so this visualization script is
# self-contained (it mirrors the encoder used during SimSiam pretraining).


class BasicBlock(nn.Module):
    '''
    Pre-activation residual block.
    Reference: "Identity Mappings in Deep Residual Networks", He et al.
    '''
    def __init__(self, in_planes: int = 32, out_planes: int = 64, stride: int = 1, dropRate: float = 0.0):
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

        # When dimensions change, the shortcut is applied to the pre-activated input.
        shortcut = self.convShortcut(out) if not self.equalInOut else x

        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), negative_slope=0.1, inplace=True)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.conv2(out)

        return out + shortcut


class EncoderStage(nn.Module):
    """PixelUnshuffle downsample followed by a pre-activation residual block."""
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
    def __init__(self, in_channels: int = 3, features=None):
        super().__init__()

        features = features or [128, 256, 384, 512]

        self.pad_divisor = (2 ** len(features))  # e.g. 2^4 = 16

        self.initial_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False)
        self.bn_initial_conv = nn.BatchNorm2d(features[0])

        self.encoder_stages = nn.ModuleList()
        self.bottleneck_pool = nn.MaxPool2d(2)

        for i in range(len(features) - 1):
            self.encoder_stages.append(EncoderStage(in_c=features[i], out_c=features[i + 1]))

        self.bottleneck_block = BasicBlock(in_planes=features[-1], out_planes=features[-1] * 2, dropRate=0.2)

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


def load_pretrained_encoder(encoder_ckpt, num_phase_shifts, features, device):
    """Load only the encoder weights from a full SimSiam checkpoint."""
    simsiam_state = torch.load(encoder_ckpt, weights_only=True)
    encoder_state = {}
    for k, v in simsiam_state.items():
        # Keep only encoder keys, excluding the avgpool added by SimSiamEncoderWrapper.
        if k.startswith('encoder.') and not k.startswith('encoder.avgpool'):
            encoder_state[k[len('encoder.'):]] = v

    trained_encoder = Att_PreactRes_UNet_Encoder(in_channels=num_phase_shifts * 2 * 3, features=features)
    trained_encoder.load_state_dict(encoder_state)
    print('\nSuccessfully loaded SimSiam pre-trained encoder parameters\n')

    trained_encoder.eval()
    trained_encoder = trained_encoder.to(device)

    tot_params = sum(p.numel() for p in trained_encoder.parameters())
    print(f'Trained encoder has {tot_params} params\n')

    return trained_encoder


def main():
    parser = argparse.ArgumentParser(
        description='PCA-RGB visualization of frozen SimSiam encoder features over an object')

    parser.add_argument('--encoder-ckpt', required=True,
                        help='Path to the SimSiam checkpoint to load the (frozen) encoder from')
    parser.add_argument('--psp-data-dir', required=True,
                        help='Directory with the demosaiced camera-captured PSP images (psp_{height,width}_fhigh_*.exr)')
    parser.add_argument('--valid-pts-mask', required=True,
                        help='Path to the SAM object mask .npy giving the valid surface points')
    parser.add_argument('--output', default='pca_rgb_sss.exr',
                        help='Output path for the PCA-RGB visualization (.exr)')

    parser.add_argument('--bbox-size', type=int, default=90, help='Square patch size (H = W)')
    parser.add_argument('--num-phase-shifts', type=int, default=4, help='Phase shifts per direction')
    parser.add_argument('--batch-size', type=int, default=1024, help='Inference batch size')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_phase_shifts = args.num_phase_shifts
    features = [128, 256, 384, 512]
    width_bbox = height_bbox = args.bbox_size

    trained_encoder = load_pretrained_encoder(
        encoder_ckpt=args.encoder_ckpt,
        num_phase_shifts=num_phase_shifts,
        features=features,
        device=device,
    )

    # Load the SAM mask and extract valid (non-background) surface points.
    valid_pts_mask = np.load(args.valid_pts_mask)
    non_zero_pts = (valid_pts_mask > 0)
    coords = np.argwhere(non_zero_pts.squeeze())
    y_coords_valid = coords[:, 0]
    x_coords_valid = coords[:, 1]

    mask_rgb = np.stack([valid_pts_mask.squeeze()] * 3, axis=2)

    # Read camera-captured high-freq PSP images (input to the encoder).
    flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
    psp_imgs_height = [cv2.imread(os.path.join(args.psp_data_dir, f'psp_height_fhigh_{i}.exr'), flags)
                       for i in range(num_phase_shifts)]
    psp_imgs_width = [cv2.imread(os.path.join(args.psp_data_dir, f'psp_width_fhigh_{i}.exr'), flags)
                      for i in range(num_phase_shifts)]

    psp_imgs_height = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in psp_imgs_height]
    psp_imgs_width = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in psp_imgs_width]

    psp_imgs_height = [img * mask_rgb for img in psp_imgs_height]
    psp_imgs_width = [img * mask_rgb for img in psp_imgs_width]

    psp_imgs_height = np.stack(psp_imgs_height, axis=0)
    psp_imgs_width = np.stack(psp_imgs_width, axis=0)

    psp_imgs = np.vstack((psp_imgs_height, psp_imgs_width)).astype(np.float32)
    psp_imgs = torch.from_numpy(psp_imgs)

    img_H, img_W = psp_imgs.shape[1], psp_imgs.shape[2]
    psp_imgs = psp_imgs.permute(0, 3, 1, 2)
    psp_imgs = psp_imgs.reshape(-1, img_H, img_W)
    psp_imgs = psp_imgs.unsqueeze(dim=0)

    print(f'\nfinal input shape: {psp_imgs.shape}\n')

    # PCA-RGB map: compute encoder features for all N valid points -> (N, 1024)
    # matrix -> PCA to 3 components -> assign to RGB. Regions with similar SSS
    # naturally cluster into similar colors.
    batch_size = args.batch_size
    all_feats = []
    all_yc, all_xc = [], []

    for batch_start in tqdm(range(0, x_coords_valid.shape[0], batch_size)):
        batch_end = min(batch_start + batch_size, x_coords_valid.shape[0])
        current_batch_size = batch_end - batch_start
        yc_batch = y_coords_valid[batch_start:batch_end]
        xc_batch = x_coords_valid[batch_start:batch_end]

        # Bounding boxes for the whole batch.
        start_left_batch = xc_batch - width_bbox // 2
        end_right_batch = start_left_batch + width_bbox
        start_top_batch = yc_batch - height_bbox // 2
        end_bottom_batch = start_top_batch + height_bbox

        inp_patches = torch.stack([
            psp_imgs[:, :, start_top_batch[i]:end_bottom_batch[i],
                            start_left_batch[i]:end_right_batch[i]]
            for i in range(current_batch_size)
        ], dim=0).squeeze(1).to(device)

        with torch.no_grad():
            output_patches = trained_encoder(inp_patches)  # (batch, C, H, W)
        output_patches = F.adaptive_avg_pool2d(output_patches, output_size=(1, 1)).squeeze().detach()

        all_feats.append(output_patches.cpu().numpy())
        all_yc.append(yc_batch)
        all_xc.append(xc_batch)

    all_feats = np.concatenate(all_feats, axis=0)  # (N, 1024)
    all_yc = np.concatenate(all_yc)
    all_xc = np.concatenate(all_xc)

    print('\nFinished feature extraction. Running torch.pca_lowrank() SVD/PCA\n')

    # PCA -> RGB on GPU via randomized SVD (far faster than sklearn on N x 1024).
    all_feats_t = torch.from_numpy(all_feats).to(device)
    _, _, V = torch.pca_lowrank(all_feats_t, q=3, center=True, niter=4)
    feats_pca = ((all_feats_t - all_feats_t.mean(0)) @ V).cpu().numpy()  # (N, 3)
    del all_feats_t
    torch.cuda.empty_cache()

    # Normalize each principal component to [0, 1].
    feats_pca -= feats_pca.min(axis=0)
    feats_pca /= feats_pca.max(axis=0)

    rgb_map = np.zeros((img_H, img_W, 3), dtype=np.float32)
    rgb_map[all_yc, all_xc] = feats_pca

    iio.imwrite(args.output, rgb_map)
    print(f'Saved PCA-RGB visualization to {args.output}')


if __name__ == '__main__':
    main()
