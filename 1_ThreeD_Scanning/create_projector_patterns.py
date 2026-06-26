import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse

import numpy as np
import cv2


# Pattern generation adapted from https://github.com/elerac/structuredlight


def split(imgs):
    return list(imgs.transpose(2, 0, 1))


def generate_psp_patterns(
    F_height: float = 108.0, F_width: float = 192.0,
    num_phase_shifts: int = 3,
    proj_width: int = 1920, proj_height: int = 1080,
    gamma_val: float = 1.77950210326
):
    """Generate gamma-corrected phase-shift profilometry (PSP) patterns.

    Returns (width-varying patterns, height-varying patterns), each of shape
    (proj_height, proj_width, num_phase_shifts) as uint8.
    """
    w_width = 2 * np.pi / proj_width * F_width
    w_height = 2 * np.pi / proj_height * F_height

    imgs_code_width = np.fromfunction(
        lambda y, x, n: 0.5 * (np.cos(w_width * x + 2 * np.pi * n / num_phase_shifts) + 1),
        (proj_height, proj_width, num_phase_shifts), dtype=np.float32)
    imgs_code_height = np.fromfunction(
        lambda y, x, n: 0.5 * (np.cos(w_height * y + 2 * np.pi * n / num_phase_shifts) + 1),
        (proj_height, proj_width, num_phase_shifts), dtype=np.float32)

    # Apply gamma-correction + quantize to uint8.
    imgs_code_height = np.power(imgs_code_height, 1.0 / gamma_val)
    imgs_code_height = np.round(imgs_code_height * 255.0).astype(np.uint8)
    imgs_code_width = np.power(imgs_code_width, 1.0 / gamma_val)
    imgs_code_width = np.round(imgs_code_width * 255.0).astype(np.uint8)

    return imgs_code_width, imgs_code_height


def save_pattern_stack(stack, out_dir, prefix):
    """Save each phase-shift of a pattern stack as <prefix>_<i>.png."""
    for i, img in enumerate(split(stack)):
        cv2.imwrite(os.path.join(out_dir, f'{prefix}_{i}.png'), img)


def main():
    parser = argparse.ArgumentParser(
        description='Generate phase-shift profilometry (PSP) projector patterns')
    parser.add_argument('--output-dir', default='projector_images',
                        help='Directory where the projector patterns are written')
    parser.add_argument('--proj-width', type=int, default=1920, help='Projector width (pixels)')
    parser.add_argument('--proj-height', type=int, default=1080, help='Projector height (pixels)')
    parser.add_argument('--num-phase-shifts', type=int, default=3,
                        help='Number of phase shifts per pattern')
    parser.add_argument('--gamma', type=float, default=1.77950210326,
                        help='Projector gamma value used for gamma correction')
    parser.add_argument('--f-height-high', type=float, default=60,
                        help='Highest height-direction frequency (e.g. 1080/18)')
    parser.add_argument('--f-width-high', type=float, default=96,
                        help='Highest width-direction frequency (e.g. 1920/20)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    common = dict(
        num_phase_shifts=args.num_phase_shifts,
        proj_width=args.proj_width, proj_height=args.proj_height,
        gamma_val=args.gamma,
    )

    # Generate the multi-frequency PSP patterns (low f1, mid f10, and highest).
    psp_width_f1, psp_height_f1 = generate_psp_patterns(F_height=1, F_width=1, **common)
    psp_width_f10, psp_height_f10 = generate_psp_patterns(F_height=10, F_width=10, **common)
    psp_width_fhigh, psp_height_fhigh = generate_psp_patterns(
        F_height=args.f_height_high, F_width=args.f_width_high, **common)

    print('\nCreating projector patterns and saving to disk...\n')

    save_pattern_stack(psp_height_f1, args.output_dir, 'psp_height_f1')
    save_pattern_stack(psp_width_f1, args.output_dir, 'psp_width_f1')
    save_pattern_stack(psp_height_f10, args.output_dir, 'psp_height_f10')
    save_pattern_stack(psp_width_f10, args.output_dir, 'psp_width_f10')
    save_pattern_stack(psp_height_fhigh, args.output_dir, 'psp_height_fhigh')
    save_pattern_stack(psp_width_fhigh, args.output_dir, 'psp_width_fhigh')

    # Dark and white reference frames.
    img_dark = np.zeros((args.proj_height, args.proj_width), dtype=np.uint8)
    img_white = np.ones((args.proj_height, args.proj_width), dtype=np.uint8) * 255
    cv2.imwrite(os.path.join(args.output_dir, 'dark_frame.png'), img_dark)
    cv2.imwrite(os.path.join(args.output_dir, 'white_frame.png'), img_white)

    print(f'Done. Patterns written to {args.output_dir}')


if __name__ == '__main__':
    main()
