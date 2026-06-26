import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import v2


# Augmentation pipelines and the multi-object SSL dataset, consolidated into a
# single module. Both pipelines expose the same interface: they are called with
# a patch stack of shape (2 * num_phase_shifts, 3, H, W) and return a (weak,
# strong) pair of augmented views, so the dataset can stay augmentation-agnostic.


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR (https://arxiv.org/abs/2002.05709)."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class ImageNetTransforms_weakstrong:
    """Standard ImageNet-style weak/strong SSL augmentations.

    Augmentation design follows "Mean Shift for Self-Supervised Learning"
    (Koohpayegani et al., ICCV 2021): https://github.com/UMBCvision/MSF

    The torchvision transforms operate on PIL images, so each phase-shift image
    in the stack is converted, augmented and stacked individually.
    """

    def __init__(self, final_size: int = 90):
        self.aug_strong = transforms.Compose([
            transforms.RandomResizedCrop(final_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.aug_weak = transforms.Compose([
            transforms.RandomResizedCrop(final_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (2*N, 3, H, W), float in [0, 1].
        weak_list = []
        strong_list = []
        for step in x:
            # (3, H, W) float -> (H, W, 3) uint8 for PIL.
            step_hwc = (np.clip(step.numpy(), 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
            pil_img = Image.fromarray(step_hwc)
            weak_list.append(self.aug_weak(pil_img))
            strong_list.append(self.aug_strong(pil_img))
        return torch.stack(weak_list), torch.stack(strong_list)


class PSPTransforms_weakstrong:
    """Our novel weak/strong augmentation pipeline for phase-shift profilometry (PSP) image stacks.

    Produces two independently cropped views of an input stack: a weakly
    augmented view (spatial crop + color jitter) and a strongly augmented view
    (additionally perturbed with salt-and-pepper noise, Gaussian blur or phase
    jitter, PSP flips and phase-shift shuffling).
    """

    def __init__(
        self,
        num_phase_shifts: int = 4,
        col_jit_bright: float = 0.2, col_jit_contrast: float = 0.3,
        col_jit_saturation: float = 0.1, col_jit_hue: float = 0.1,
        salt_prob: float = 0.05, pepper_prob: float = 0.05,
        gauss_blur_kernel_size: int = 3,
        gauss_blur_sigma: tuple = (0.1, 1.0),
        mean_phase_jitter: float = 0.0,
        std_phase_jitter: float = 0.01,
        base_crop_size: int = 80, final_size: int = 90,
        prob_saltpepper: float = 0.3,
        prob_gauss_blur: float = 0.5,
        prob_phase_shuffle: float = 0.4,
        prob_psp_flip: float = 0.4
    ):
        self.num_phase_shifts = num_phase_shifts

        # Random resized crop: initial crop size and the size it is resized to.
        self.base_crop_size = base_crop_size
        self.final_size = final_size

        # Salt-and-pepper noise probabilities.
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

        # Random phase jitter noise.
        self.mean_phase_jitter = mean_phase_jitter
        self.std_phase_jitter = std_phase_jitter

        # Probabilities of applying each strong augmentation.
        self.prob_saltpepper = prob_saltpepper
        self.prob_gauss_blur = prob_gauss_blur
        self.prob_phase_shuffle = prob_phase_shuffle
        self.prob_psp_flip = prob_psp_flip

        self.gauss_blur_kernel_size = gauss_blur_kernel_size
        self.gauss_blur_sigma = gauss_blur_sigma

        self.color_jitter = v2.ColorJitter(
            brightness=col_jit_bright,
            contrast=col_jit_contrast,
            saturation=col_jit_saturation,
            hue=col_jit_hue
        )

    def base_transform(self, stack: torch.Tensor):
        """Apply a consistent random crop, resize and color jitter to the stack.

        Input stack shape: (num_phase_shifts, C, H, W).
        """
        # Consistent spatial crop and resize (same crop for all PSP images).
        i, j, h, w = transforms.RandomCrop.get_params(
            stack,
            output_size=(self.base_crop_size, self.base_crop_size)
        )
        stack_cropped = TF.crop(stack, i, j, h, w)
        stack_resized = TF.resize(stack_cropped, size=self.final_size)

        # Consistent color jitter applied to all PSP images.
        stack_transformed = self.color_jitter(stack_resized)

        return stack_transformed

    def gaussian_blur(self, stack: torch.Tensor):
        """Apply Gaussian blur. Input stack shape: (num_phase_shifts, C, H, W)."""
        return TF.gaussian_blur(img=stack, kernel_size=self.gauss_blur_kernel_size, sigma=self.gauss_blur_sigma)

    def salt_pepper_noise(
        self,
        stack: torch.Tensor,
        independent_noise_per_img: bool = False
    ):
        """Apply salt-and-pepper noise. Input stack shape: (num_phase_shifts, C, H, W)."""
        n, c, h, w = stack.shape
        device = stack.device

        if not independent_noise_per_img:
            # Same noise pattern applied to all images in the stack.
            random_mask = torch.rand(h, w, device=device)

            pepper_mask = (random_mask < self.pepper_prob)
            salt_mask = (random_mask >= self.pepper_prob) & (random_mask < self.pepper_prob + self.salt_prob)

            img_salt_pepper = stack.clone()

            # Pepper -> 0, salt -> 1 (images assumed normalized to [0, 1]).
            img_salt_pepper[:, :, pepper_mask] = 0.0
            img_salt_pepper[:, :, salt_mask] = 1.0

        else:
            # Independent noise per image: (N, H, W).
            random_mask = torch.rand(n, h, w, device=device)

            pepper_mask = (random_mask < self.pepper_prob)
            salt_mask = (random_mask >= self.pepper_prob) & (random_mask < self.pepper_prob + self.salt_prob)

            img_salt_pepper = stack.clone()

            # Expand masks to (N, 1, H, W) for broadcasting across channels.
            img_salt_pepper[pepper_mask.unsqueeze(1).expand(-1, c, -1, -1)] = 0.0
            img_salt_pepper[salt_mask.unsqueeze(1).expand(-1, c, -1, -1)] = 1.0

        return img_salt_pepper

    def add_random_phase_jitter(self, stack: torch.Tensor):
        """Add Gaussian phase jitter noise. Input stack shape: (num_phase_shifts, C, H, W)."""
        phase_noise = torch.randn_like(stack) * self.std_phase_jitter + self.mean_phase_jitter

        stack = stack + phase_noise

        if stack.min() < 0.0 or stack.max() <= 1.0:
            stack = torch.clamp(stack, 0.0, 1.0)
        else:
            stack = (stack - stack.min()) / (stack.max() - stack.min())

        return stack

    def flip_psp_imgs(self, stack: torch.Tensor) -> torch.Tensor:
        """Flip the height- and width-direction PSP images. Input: (num_phase_shifts, C, H, W)."""
        # First N-step PSP varies along height -> vertical flip (H dimension).
        img_vflip = torch.flip(stack[:self.num_phase_shifts], dims=[-2])

        # Latter N-step PSP varies along width -> horizontal flip (W dimension).
        img_hflip = torch.flip(stack[self.num_phase_shifts:], dims=[-1])

        img_flipped = torch.cat([img_vflip, img_hflip], dim=0)

        return img_flipped

    def random_phaseshift_shuffle(self, stack: torch.Tensor) -> torch.Tensor:
        """Shuffle phase-shift order along the height and width directions.

        Expects an input tensor of shape (num_phase_shifts, C, H, W); the first
        num_phase_shifts images are height patterns and the latter are width
        patterns.
        """
        height_shuffle = torch.randperm(self.num_phase_shifts)
        width_shuffle = torch.randperm(self.num_phase_shifts)

        stack[:self.num_phase_shifts] = stack[:self.num_phase_shifts][height_shuffle]
        stack[self.num_phase_shifts:] = stack[self.num_phase_shifts:][width_shuffle]

        return stack

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (2*N, C, H, W).

        # Pad if the input is smaller than the crop size.
        if x.shape[-1] < self.base_crop_size or x.shape[-2] < self.base_crop_size:
            x = TF.pad(x, padding=self.base_crop_size, padding_mode='reflect')

        # Two independent random crops are generated for the two views, forcing
        # the model to relate different parts of the input to each other.

        # View 1: weak augmentation path.
        i1, j1, h1, w1 = v2.RandomCrop.get_params(
            x, output_size=(self.base_crop_size, self.base_crop_size)
        )
        x_view1 = TF.crop(x, i1, j1, h1, w1)
        x_view1 = TF.resize(x_view1, size=self.final_size)
        x_weak = self.color_jitter(x_view1)

        # View 2: strong augmentation path (independent crop).
        i2, j2, h2, w2 = v2.RandomCrop.get_params(
            x, output_size=(self.base_crop_size, self.base_crop_size)
        )
        x_view2 = TF.crop(x, i2, j2, h2, w2)
        x_view2 = TF.resize(x_view2, size=self.final_size)

        x_strong = x_view2

        if random.random() < self.prob_saltpepper:
            x_strong = self.salt_pepper_noise(stack=x_strong)

        if random.random() < self.prob_gauss_blur:
            x_strong = self.gaussian_blur(stack=x_strong)
        else:
            x_strong = self.add_random_phase_jitter(stack=x_strong)

        if random.random() < self.prob_psp_flip:
            x_strong = self.flip_psp_imgs(stack=x_strong)

        if random.random() < self.prob_phase_shuffle:
            x_strong = self.random_phaseshift_shuffle(stack=x_strong)

        return x_weak, x_strong


class MultiObjectSSS_Dataset(torch.utils.data.Dataset):
    """Scalable multi-object dataset for SSL pretraining on PSP image stacks.

    Coordinates are computed from per-object masks in ``__init__`` while the
    heavy image data is kept memory-mapped on disk and read lazily in
    ``__getitem__``. Augmentation is delegated to ``data_augmentations``, a
    callable that maps a patch stack of shape (2 * num_phase_shifts, 3, H, W) to
    a (weak, strong) pair of views (e.g. ``ImageNetTransforms_weakstrong`` or
    ``PSPTransforms_weakstrong``).
    """

    def __init__(
        self,
        object_configs: list,  # [{'memmap_path': ..., 'mask_path': ...}, ...]
        width_bbox: int = 90,
        height_bbox: int = 90,
        data_augmentations=None
    ):
        super().__init__()
        self.width_bbox = width_bbox
        self.height_bbox = height_bbox
        self.data_augmentations = data_augmentations

        self.samples = []       # (object_idx, x_center, y_center)
        self.objects_data = []  # opened memmaps and masks per object

        print("Initializing dataset...")

        for obj_idx, config in enumerate(object_configs):
            # Load shape metadata.
            shape = tuple(np.load(config['memmap_path'].replace('.dat', '_shape.npy')))

            # Memory-map the data (read-only, not loaded into RAM).
            mmap_img = np.memmap(config['memmap_path'], dtype='float32', mode='r', shape=shape)

            # Load the (small) coordinate mask.
            sam_mask = np.load(config['mask_path'])
            x_coords = sam_mask[:, 0]
            y_coords = sam_mask[:, 1]

            # Keep only coordinates whose crop stays inside the image bounds.
            valid_indices = (
                (x_coords >= width_bbox // 2) &
                (x_coords < shape[-1] - width_bbox // 2) &
                (y_coords >= height_bbox // 2) &
                (y_coords < shape[-2] - height_bbox // 2)
            )

            valid_x = x_coords[valid_indices]
            valid_y = y_coords[valid_indices]

            for x, y in zip(valid_x, valid_y):
                self.samples.append((obj_idx, x, y))

            self.objects_data.append({
                'data': mmap_img,
                'mask': torch.from_numpy(sam_mask).bool(),
                'shape': shape
            })

            print(f"Object {obj_idx}: Loaded {len(valid_x)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_idx, cx, cy = self.samples[idx]
        obj_info = self.objects_data[obj_idx]

        # Crop boundaries centered on the sampled coordinate.
        half_w, half_h = self.width_bbox // 2, self.height_bbox // 2
        start_x = int(cx - half_w)
        start_y = int(cy - half_h)
        end_x = start_x + self.width_bbox
        end_y = start_y + self.height_bbox

        # Slice from the memmap (reads only the requested bytes from disk).
        patch = np.array(obj_info['data'][:, start_y:end_y, start_x:end_x])
        patch_tensor = torch.from_numpy(patch)

        # Reshape from (2*N*3, H, W) to (2*N, 3, H, W).
        c, h, w = patch_tensor.shape
        patch_tensor = patch_tensor.view(-1, 3, h, w)

        if self.data_augmentations:
            x_weak, x_strong = self.data_augmentations(patch_tensor)
            return x_weak, x_strong

        return patch_tensor, patch_tensor
