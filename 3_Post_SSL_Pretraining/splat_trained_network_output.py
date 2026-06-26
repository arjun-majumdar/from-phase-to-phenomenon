import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse

import numpy as np
import cv2
import imageio.v3 as iio
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from Att_PreAct_pxshuffle_UNet_Decoder_Only import Att_PreActRes_UNet_Pxshuffle_DecoderOnly


def bilinear_interpolate_batch(img, x, y):
    """Bilinear interpolation for a batch of coordinates.

    Args:
        img: (H, W, 3) image
        x: (N,) array of x coordinates
        y: (N,) array of y coordinates

    Returns:
        (N, 3) interpolated colors
    """
    H, W, C = img.shape

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    # Clip to valid range
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # Interpolation weights
    wx = x - np.floor(x)
    wy = y - np.floor(y)

    # Gather corner values: (N, 3)
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    # Bilinear weights
    wa = (1 - wx) * (1 - wy)
    wb = (1 - wx) * wy
    wc = wx * (1 - wy)
    wd = wx * wy

    # Weighted sum: (N, 3)
    result = (wa[:, None] * Ia + wb[:, None] * Ib +
              wc[:, None] * Ic + wd[:, None] * Id)

    return result


def batched_splat_relighting(
    all_keys, psp_imgs, trained_model,
    proj_virtual_img, px2px_corresp_width, px2px_corresp_height,
    width_bbox, height_bbox, device, batch_size=64):

    # Image dimensions. psp_imgs is (1, C, H, W).
    _, _, img_H, img_W = psp_imgs.shape

    splat_output_relit_img = np.zeros((img_H, img_W, 3), dtype=np.float32)
    accumulation_relit_buffer = np.zeros((img_H, img_W, 3), dtype=np.float32)

    # Track how many patches contribute to each pixel, useful for normalizing the
    # accumulation and for diagnosing coverage and consistency.
    count_buffer = np.zeros((img_H, img_W), dtype=np.int32)

    # Move source data to GPU once for faster slicing.
    train_img_patches_gpu = psp_imgs.to(device)

    all_keys = list(all_keys)
    num_keys = len(all_keys)

    for batch_start in tqdm(range(0, num_keys, batch_size)):
        batch_end = min(batch_start + batch_size, num_keys)
        batch_keys = all_keys[batch_start:batch_end]
        current_batch_size = len(batch_keys)

        # Parse coords
        yc_batch = np.array([k[1] for k in batch_keys])
        xc_batch = np.array([k[0] for k in batch_keys])

        # Safe bounding-box calculation (clamp to stay in bounds).
        half_w, half_h = width_bbox // 2, height_bbox // 2
        st_batch = np.clip(yc_batch - half_h, 0, img_H - height_bbox)
        sl_batch = np.clip(xc_batch - half_w, 0, img_W - width_bbox)
        sb_batch = st_batch + height_bbox
        sr_batch = sl_batch + width_bbox

        # Projector colors per point.
        xp_batch = px2px_corresp_width[yc_batch, xc_batch]
        yp_batch = px2px_corresp_height[yc_batch, xc_batch]
        col_batch = bilinear_interpolate_batch(img=proj_virtual_img, x=xp_batch, y=yp_batch)

        # Gather input patches on the GPU.
        inp_patches = torch.stack([
            train_img_patches_gpu[0, :, st_batch[i]:sb_batch[i], sl_batch[i]:sr_batch[i]]
            for i in range(current_batch_size)
        ], dim=0)

        with torch.no_grad():
            output_patches = trained_model(inp_patches)

        output_patches = output_patches.permute(0, 2, 3, 1).cpu().numpy()

        # Accumulation (patches may overlap, so this stays a Python loop).
        for i in range(current_batch_size):
            st, sb, sl, sr = st_batch[i], sb_batch[i], sl_batch[i], sr_batch[i]

            # Apply color modulation
            modulated_patch = output_patches[i] * col_batch[i].reshape(1, 1, 3)

            splat_output_relit_img[st:sb, sl:sr, :] += modulated_patch
            accumulation_relit_buffer[st:sb, sl:sr, :] += output_patches[i]
            count_buffer[st:sb, sl:sr] += 1

    return splat_output_relit_img, accumulation_relit_buffer, count_buffer


def main():
    parser = argparse.ArgumentParser(
        description='Splat the trained decoder output over an object to render a relit result')

    parser.add_argument('--psp-corresp-dir', required=True,
                        help='Directory with px2px_corresp_width.npy and px2px_corresp_height.npy')
    parser.add_argument('--psp-data-dir', required=True,
                        help='Directory with the demosaiced camera-captured PSP images (psp_{height,width}_fhigh_*.exr)')
    parser.add_argument('--sam-mask', required=True,
                        help='Path to the SAM object mask .npy')
    parser.add_argument('--gt-sss-dir', required=True,
                        help='Directory containing the GT SSS HDF5 file and white_frame.exr')
    parser.add_argument('--hdf5-filename', default='sssfootprint_90x90_patch.h5',
                        help='HDF5 filename (within --gt-sss-dir) whose keys give the (x, y) point coordinates')
    parser.add_argument('--projector-img', required=True,
                        help='Path to the virtual projector image used to colorize the relit result')
    parser.add_argument('--decoder-ckpt', required=True,
                        help='Path to the trained decoder-only checkpoint (.pth)')
    parser.add_argument('--output', required=True,
                        help='Output path for the rendered relit image (.exr)')

    parser.add_argument('--bbox-size', type=int, default=90, help='Square patch size (H = W)')
    parser.add_argument('--num-phase-shifts', type=int, default=4, help='Phase shifts per direction')
    parser.add_argument('--batch-size', type=int, default=256, help='Inference batch size')
    parser.add_argument('--no-show', action='store_true', help='Do not display the result with matplotlib')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    height_bbox = width_bbox = args.bbox_size
    num_phase_shifts = args.num_phase_shifts

    # Read in the virtual projector image.
    proj_virtual_img = iio.imread(args.projector_img)
    proj_virtual_img = (proj_virtual_img / 255.).astype(np.float32)

    # Load the SAM object mask.
    sam_mask = np.load(args.sam_mask)
    mask_rgb = np.stack([sam_mask] * 3, axis=2)

    # Load pre-processed px-to-px correspondences.
    px2px_corresp_width = np.load(os.path.join(args.psp_corresp_dir, 'px2px_corresp_width.npy')).astype(np.float32)
    px2px_corresp_height = np.load(os.path.join(args.psp_corresp_dir, 'px2px_corresp_height.npy')).astype(np.float32)

    print(f'\nxp: min = {px2px_corresp_width.min():.4f} & max = {px2px_corresp_width.max():.4f}')
    print(f'yp: min = {px2px_corresp_height.min():.4f} & max = {px2px_corresp_height.max():.4f}\n')

    # Read the (x, y) point coordinates from the GT SSS HDF5 keys.
    x_coords, y_coords = [], []
    with h5py.File(os.path.join(args.gt_sss_dir, args.hdf5_filename), 'r') as f:
        for key in tqdm(f.keys()):
            xc, yc = key.split('_')
            x_coords.append(int(xc))
            y_coords.append(int(yc))

    all_keys = np.stack([np.array(x_coords), np.array(y_coords)], axis=1)

    # Read camera-captured high-freq PSP images (input to the network).
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

    # Read the white image used to compute the correction matrix.
    img_white = cv2.imread(os.path.join(args.gt_sss_dir, 'white_frame.exr'), flags)
    img_white = cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB)
    img_white = img_white * mask_rgb

    # Build the trained decoder-only network.
    in_channels = num_phase_shifts * 2 * 3
    out_channels = 3
    features = [128, 256, 384, 512]

    trained_model_decoder_only = Att_PreActRes_UNet_Pxshuffle_DecoderOnly(
        in_channels=in_channels, out_channels=out_channels, features=features
    )
    trained_model_decoder_only.load_state_dict(torch.load(args.decoder_ckpt, weights_only=True))
    trained_model_decoder_only.eval()
    trained_model_decoder_only = trained_model_decoder_only.to(device)

    tot_params = sum(p.numel() for p in trained_model_decoder_only.parameters())
    print(f'\nLoaded trained decoder-only network ({tot_params} parameters)\n')

    torch.cuda.empty_cache()

    splat_output_relit_img, accumulation_relit_buffer, count_buffer = batched_splat_relighting(
        all_keys=all_keys,
        psp_imgs=psp_imgs,
        trained_model=trained_model_decoder_only,
        proj_virtual_img=proj_virtual_img,
        px2px_corresp_width=px2px_corresp_width,
        px2px_corresp_height=px2px_corresp_height,
        width_bbox=width_bbox, height_bbox=height_bbox,
        device=device, batch_size=args.batch_size
    )

    # Correct the splat using the white image and per-pixel accumulation.
    positive_vals = accumulation_relit_buffer[accumulation_relit_buffer > 0]
    eps = 1e-5
    threshold = np.percentile(positive_vals, 1) if len(positive_vals) > 0 else eps

    X = np.divide(img_white, accumulation_relit_buffer, where=accumulation_relit_buffer > threshold)
    X = np.clip(X, 0.0, 5.0)

    splat_corr = splat_output_relit_img * X
    splat_corr = splat_corr * mask_rgb
    print(f'\nCorrected splat: min = {splat_corr.min():.2f} & max = {splat_corr.max():.4f}\n')

    iio.imwrite(args.output, splat_corr)
    print(f'Saved rendered relit result to {args.output}')

    if not args.no_show:
        plt.imshow(splat_corr ** (1. / 2.2))
        plt.title('corrected relit result')
        plt.show()


if __name__ == '__main__':
    main()
