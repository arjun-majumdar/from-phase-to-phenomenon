import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse

import numpy as np
import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt


# Phase-shift decoding adapted from https://github.com/elerac/structuredlight


def split(imgs):
    return list(imgs.transpose(2, 0, 1))


def merge(imlist):
    return np.dstack(imlist)


def decodePhase_divbyzero(imlist):
    '''Decode the wrapped phase from a phase-shift image list.

    Uses an epsilon and arccos-domain clamping to stay robust when both
    quadrature terms vanish or floating-point error pushes the ratio outside
    [-1, 1].
    '''
    U = calcParam(imlist)
    U1, U2, U3 = split(U)
    eps = 1e-10
    amplitude = np.sqrt(U2**2 + U3**2) + eps
    img_phase = np.arccos(np.clip(U2 / amplitude, -1.0, 1.0))
    img_phase[U3 < 0] = 2 * np.pi - img_phase[U3 < 0]
    return img_phase


def decodeAmplitude(imlist):
    U = calcParam(imlist)
    U1, U2, U3 = split(U)
    return np.sqrt(U2**2 + U3**2)


def decodeOffset(imlist):
    U = calcParam(imlist)
    U1, U2, U3 = split(U)
    return U1


def calcParam(imlist):
    num = len(imlist)
    n = np.arange(0, num)
    R = merge(imlist)
    M = np.array([np.ones_like(n), np.cos(2 * np.pi * n / num), -np.sin(2 * np.pi * n / num)]).T  # (num, 3)
    M_pinv = np.linalg.inv(M.T @ M) @ M.T  # (3, num)
    U = np.tensordot(M_pinv, R, axes=(1, 2)).transpose(1, 2, 0)  # (height, width, 3)
    return U


def unwrap_hierarchical(phi_wrapped_high, phi_unwrapped_low, f_high, f_low):
    """Unwrap a high-frequency phase using a previously unwrapped lower frequency.

    phi_wrapped_high:  the current wrapped phase to unwrap
    phi_unwrapped_low: the already-unwrapped phase from the previous step
    f_high, f_low:     the two frequencies
    """
    ratio = f_high / f_low

    # Predict the high phase from the low phase, then snap to the nearest fringe order k.
    phi_predict = phi_unwrapped_low * ratio
    k = np.round((phi_predict - phi_wrapped_high) / (2 * np.pi))

    return phi_wrapped_high + 2 * np.pi * k


def unwrap_cascade(wrapped_phases, frequencies):
    """Unwrap through multiple frequencies from lowest to highest."""
    unwrapped = wrapped_phases[0].copy()
    for i in range(1, len(frequencies)):
        unwrapped = unwrap_hierarchical(
            wrapped_phases[i], unwrapped, frequencies[i], frequencies[i - 1])
    return unwrapped


def refine_coordinates(phase_width, phase_height, kernel_size=5):
    """Median-filter the unwrapped coordinates to remove phase-unwrapping spikes."""
    phase_width_filtered = cv2.medianBlur(phase_width.astype(np.float32), kernel_size)
    phase_height_filtered = cv2.medianBlur(phase_height.astype(np.float32), kernel_size)
    return phase_width_filtered, phase_height_filtered


def compute_valid_mask_gamma(
    amp_h, amp_w, off_h, off_w, sam_mask,
    off_pct=5, gamma_pct=20, eps=1e-6, morph_ks=5, return_stats=False
):
    """Compute a valid-point mask within the object via offset/gamma thresholds.

    Gamma = amplitude / offset is an exposure-robust modulation measure. Pixels
    inside the SAM object mask with low offset or low gamma (shadows, occlusions,
    poorly modulated regions) are discarded. Thresholds are chosen from object-only
    percentiles so they adapt to each capture.
    """
    obj = (sam_mask > 0)

    gamma_h = amp_h / (off_h + eps)
    gamma_w = amp_w / (off_w + eps)

    off_h_obj = off_h[obj]
    off_w_obj = off_w[obj]
    g_h_obj = gamma_h[obj]
    g_w_obj = gamma_w[obj]

    if off_h_obj.size == 0:
        raise ValueError("SAM mask empty (no object pixels).")

    thresh_offset_h = np.percentile(off_h_obj, off_pct)
    thresh_offset_w = np.percentile(off_w_obj, off_pct)
    thresh_gamma_h = np.percentile(g_h_obj, gamma_pct)
    thresh_gamma_w = np.percentile(g_w_obj, gamma_pct)

    offset_mask = (off_h >= thresh_offset_h) & (off_w >= thresh_offset_w)
    gamma_mask = (gamma_h >= thresh_gamma_h) & (gamma_w >= thresh_gamma_w)
    valid_mask = obj & offset_mask & gamma_mask

    # Spatial cleanup.
    if morph_ks is not None and morph_ks >= 3:
        m = (valid_mask.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        valid_mask = m.astype(bool)

    if not return_stats:
        return valid_mask

    stats = {
        "thresh_offset_h": float(thresh_offset_h),
        "thresh_offset_w": float(thresh_offset_w),
        "thresh_gamma_h": float(thresh_gamma_h),
        "thresh_gamma_w": float(thresh_gamma_w),
        "valid_fraction_in_object": float(valid_mask[obj].mean()),
    }
    return valid_mask, stats


def read_psp_stack(cam_dir, prefix, num_phase_shifts, dark_img, sam_mask):
    """Read a phase-shift stack, subtract the dark frame, and apply the SAM mask."""
    imgs = [cv2.imread(os.path.join(cam_dir, f'{prefix}_{i}.exr'), cv2.IMREAD_UNCHANGED)
            for i in range(num_phase_shifts)]
    imgs = [np.clip(img - dark_img, 0.0, 1.0) for img in imgs]
    imgs = [img * sam_mask for img in imgs]
    return imgs


def main():
    parser = argparse.ArgumentParser(
        description='Multi-frequency hierarchical PSP unwrapping and camera-to-projector '
                    'pixel correspondence computation')

    parser.add_argument('--cam-imgs-dir', required=True,
                        help='Directory with the captured PSP .exr images and dark_frame.exr')
    parser.add_argument('--sam-mask', required=True,
                        help='Path to the SAM object mask .npy (selects the object, excludes background)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to write the unwrapped phase / correspondence / valid-mask .npy '
                             '(default: --cam-imgs-dir)')
    parser.add_argument('--projector-img', default=None,
                        help='Optional virtual projector image (e.g. ColorGrid.png) for the relighting preview')
    parser.add_argument('--white-frame', default=None,
                        help='Optional captured white-frame .exr used to light the relighting preview')

    parser.add_argument('--proj-width', type=int, default=1920, help='Projector width (pixels)')
    parser.add_argument('--proj-height', type=int, default=1080, help='Projector height (pixels)')
    parser.add_argument('--num-phase-shifts', type=int, default=4, help='Phase shifts per pattern')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display matplotlib figures (for headless runs)')

    args = parser.parse_args()

    cam_dir = args.cam_imgs_dir
    out_dir = args.output_dir or cam_dir
    os.makedirs(out_dir, exist_ok=True)
    show = not args.no_show

    proj_width, proj_height = args.proj_width, args.proj_height
    num_phase_shifts = args.num_phase_shifts

    # Per-direction frequency ladders (must match the projected patterns).
    F_height = [1, 3, 9, 27, 60, 108]
    F_width = [1, 4, 16, 64, 128, 240]
    print(f'\nnumber of frequencies: Height = {len(F_height)} & Width = {len(F_width)}\n')

    # The SAM mask selects only the object so downstream training patches are not
    # poisoned by background pixels.
    sam_mask = np.load(args.sam_mask)
    print(f"\nSAM's output mask shape: {sam_mask.shape}\n")

    # Dark frame (for dark-current subtraction).
    dark_img = cv2.imread(os.path.join(cam_dir, 'dark_frame.exr'), cv2.IMREAD_UNCHANGED)
    dark_img = np.clip(dark_img, 0.0, 1.0)

    # Read every frequency's phase-shift stack (dark-subtracted and masked).
    phase_imgs_height_f1 = read_psp_stack(cam_dir, 'psp_height_f1', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_width_f1 = read_psp_stack(cam_dir, 'psp_width_f1', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_height_f3 = read_psp_stack(cam_dir, 'psp_height_f3', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_width_f4 = read_psp_stack(cam_dir, 'psp_width_f4', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_height_f9 = read_psp_stack(cam_dir, 'psp_height_f9', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_width_f16 = read_psp_stack(cam_dir, 'psp_width_f16', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_height_f27 = read_psp_stack(cam_dir, 'psp_height_f27', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_width_f64 = read_psp_stack(cam_dir, 'psp_width_f64', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_height_f60 = read_psp_stack(cam_dir, 'psp_height_f60', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_width_f128 = read_psp_stack(cam_dir, 'psp_width_f128', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_height_fhigh = read_psp_stack(cam_dir, 'psp_height_fhigh', num_phase_shifts, dark_img, sam_mask)
    phase_imgs_width_fhigh = read_psp_stack(cam_dir, 'psp_width_fhigh', num_phase_shifts, dark_img, sam_mask)

    # Wrapped phase maps per frequency.
    wrapped_height = [
        decodePhase_divbyzero(phase_imgs_height_f1),
        decodePhase_divbyzero(phase_imgs_height_f3),
        decodePhase_divbyzero(phase_imgs_height_f9),
        decodePhase_divbyzero(phase_imgs_height_f27),
        decodePhase_divbyzero(phase_imgs_height_f60),
        decodePhase_divbyzero(phase_imgs_height_fhigh),
    ]
    wrapped_width = [
        decodePhase_divbyzero(phase_imgs_width_f1),
        decodePhase_divbyzero(phase_imgs_width_f4),
        decodePhase_divbyzero(phase_imgs_width_f16),
        decodePhase_divbyzero(phase_imgs_width_f64),
        decodePhase_divbyzero(phase_imgs_width_f128),
        decodePhase_divbyzero(phase_imgs_width_fhigh),
    ]

    # Hierarchical (cascade) unwrapping from lowest to highest frequency.
    unwrapped_height = unwrap_cascade(wrapped_phases=wrapped_height, frequencies=F_height)
    unwrapped_width = unwrap_cascade(wrapped_phases=wrapped_width, frequencies=F_width)

    # Smooth the unwrapped phase maps.
    unwrapped_width_refined, unwrapped_height_refined = refine_coordinates(
        phase_width=unwrapped_width, phase_height=unwrapped_height)

    if show:
        plt.subplot(121)
        plt.imshow(unwrapped_width_refined, cmap='twilight')
        plt.colorbar(label='Unwrapped phase (rad)')
        plt.title('Unwrapped Phase - Width')
        plt.subplot(122)
        plt.imshow(unwrapped_height_refined, cmap='twilight')
        plt.colorbar(label='Unwrapped phase (rad)')
        plt.title('Unwrapped Phase - Height')
        plt.show()

    np.save(os.path.join(out_dir, 'unwrapped_width.npy'), unwrapped_width_refined)
    np.save(os.path.join(out_dir, 'unwrapped_height.npy'), unwrapped_height_refined)

    # Camera-to-projector pixel correspondences.
    # From the unwrapped phase Phi, each camera pixel maps to a projector column/row:
    #     Phi(x, y) = 2*pi*xp / lambda     (lambda = fringe wavelength in pixels)
    # See Eqn. 93 in "Phase shifting algorithms for fringe projection profilometry:
    # A review", Chao Zuo et al.
    lambda_width = proj_width / F_width[-1]
    lambda_height = proj_height / F_height[-1]
    print(f'\nLowest wav (pxs/cycle): Height = {lambda_height:.2f} & Width = {lambda_width:.2f}\n')

    xp = ((unwrapped_width_refined * lambda_width) / (2 * np.pi)).astype(np.float32)
    yp = ((unwrapped_height_refined * lambda_height) / (2 * np.pi)).astype(np.float32)

    print(f'\nxp: min = {xp.min():.4f} & max = {xp.max():.4f}')
    print(f'yp: min = {yp.min():.4f} & max = {yp.max():.4f}\n')

    # Zero out any out-of-bounds correspondences.
    xp = np.where((xp < 0) | (xp > proj_width - 1), 0, xp)
    yp = np.where((yp < 0) | (yp > proj_height - 1), 0, yp)

    if show:
        plt.subplot(121)
        plt.imshow(xp, cmap='twilight')
        plt.colorbar(label='proj px')
        plt.title('xp')
        plt.subplot(122)
        plt.imshow(yp, cmap='twilight')
        plt.colorbar(label='proj px')
        plt.title('yp')
        plt.show()

    np.save(os.path.join(out_dir, 'px2px_corresp_width.npy'), xp)
    np.save(os.path.join(out_dir, 'px2px_corresp_height.npy'), yp)

    # Optional relighting preview: remap a virtual projector image into the camera
    # view through the computed correspondences.
    final_output = None
    if args.projector_img is not None and args.white_frame is not None:
        proj_virtual_img = iio.imread(args.projector_img).astype(np.float32)
        proj_virtual_img /= proj_virtual_img.max()

        white_frame_img = iio.imread(args.white_frame)
        white_frame_img = np.clip(white_frame_img, 0.0, 1.0)

        px2px_corresp_img_smooth = cv2.remap(
            proj_virtual_img, xp, yp,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        final_output = px2px_corresp_img_smooth * white_frame_img

        if show:
            plt.imshow(final_output ** (1. / 2.2))
            plt.title('px2px corresp')
            plt.show()

    # Valid-point mask from offset/gamma modulation (extra cleanup beyond the SAM mask).
    amp_height_fhigh = decodeAmplitude(phase_imgs_height_fhigh)
    amp_width_fhigh = decodeAmplitude(phase_imgs_width_fhigh)
    offset_height_fhigh = decodeOffset(phase_imgs_height_fhigh)
    offset_width_fhigh = decodeOffset(phase_imgs_width_fhigh)

    valid_mask, stats = compute_valid_mask_gamma(
        amp_h=amp_height_fhigh, amp_w=amp_width_fhigh,
        off_h=offset_height_fhigh, off_w=offset_width_fhigh,
        sam_mask=sam_mask,
        off_pct=2, gamma_pct=10, eps=1e-6, morph_ks=5, return_stats=True)

    print(f"\nValid fraction inside object: {stats['valid_fraction_in_object']:.4f}\n")

    if show:
        plt.imshow(valid_mask, cmap='gray')
        plt.title('valid points mask')
        plt.show()

        if final_output is not None:
            final_output_valid = np.where(valid_mask[..., None], final_output, 0)
            plt.subplot(121)
            plt.imshow(final_output ** (1. / 2.2))
            plt.title('px2px corresp')
            plt.subplot(122)
            plt.imshow(final_output_valid ** (1. / 2.2))
            plt.title('px2px corresp + valid mask')
            plt.show()

    np.save(os.path.join(out_dir, 'valid_pts_mask.npy'), valid_mask)
    print(f'Done. Outputs written to {out_dir}')


if __name__ == '__main__':
    main()
