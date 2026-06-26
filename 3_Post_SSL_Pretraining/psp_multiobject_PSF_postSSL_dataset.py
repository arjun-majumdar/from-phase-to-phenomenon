import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import h5py
import numpy as np
import cv2
from tqdm import tqdm
from typing import Union, Tuple
import random

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import v2


###############################################################################
# STRATEGY A: Lazy HDF5 Reads — Minimal RAM, reads PSF patches on-the-fly
#
# RAM usage:  Only memmaps + coordinate arrays (a few MB total)
# Speed:      Slightly slower __getitem__ due to per-sample HDF5 read
# Best when:  32+ object configs and PSF files are large (GBs total)
#
# NOTE: h5py file handles are NOT safe to share across DataLoader workers.
#       We store the *path* and open per-worker using a thread-local cache.
###############################################################################

class MultiObject_PSFSSS_Dataset_LazyHDF5(torch.utils.data.Dataset):
    def __init__(
        self,
        object_configs: list,
        width_bbox: int = 90,
        height_bbox: int = 90,
        data_augmentations = None
    ):
        '''
        Lazy Strategy: Only reads coordinates in __init__.
        PSF patches are read from HDF5 on-the-fly in __getitem__.
        '''
        super().__init__()
        self.width_bbox = width_bbox
        self.height_bbox = height_bbox
        self.data_augmentations = data_augmentations

        sample_lists = []  # build per-object, then concatenate once
        self.objects_data = []
        self._h5_handles = {}  # worker-local cache: obj_idx -> h5py.File

        print("Initializing dataset (lazy mode)...")

        for obj_idx, config in enumerate(object_configs):
            # 1. Load shape metadata & memmap
            shape = tuple(np.load(config['memmap_path'].replace('.dat', '_shape.npy')))
            mmap_img = np.memmap(config['memmap_path'], dtype='float32', mode='r', shape=shape)

            # 2. Read only the keys from HDF5 — no patch data loaded
            with h5py.File(config['psf_path'], 'r') as f:
                keys = list(f.keys())

            coords = np.array([[int(k.split('_')[0]), int(k.split('_')[1])] for k in keys],
                              dtype=np.int32)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]

            # 3. Filter valid coordinates (boundary check)
            valid_mask = (
                (x_coords >= width_bbox // 2) &
                (x_coords < shape[-1] - width_bbox // 2) &
                (y_coords >= height_bbox // 2) &
                (y_coords < shape[-2] - height_bbox // 2)
            )
            valid_coords = coords[valid_mask]
            n_valid = len(valid_coords)

            # Build (obj_idx, x, y) array for this object
            obj_samples = np.empty((n_valid, 3), dtype=np.int32)
            obj_samples[:, 0] = obj_idx
            obj_samples[:, 1:] = valid_coords
            sample_lists.append(obj_samples)

            # Store per-object resources (no PSF data in RAM)
            self.objects_data.append({
                'data': mmap_img,
                'psf_path': config['psf_path'],
                'shape': shape,
            })

            print(f"Object {obj_idx}: {n_valid} valid samples.")

        # Single contiguous array — much more memory-efficient than list of tuples
        self.samples = np.concatenate(sample_lists, axis=0)
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _get_h5(self, obj_idx):
        '''Per-worker HDF5 handle cache. Avoids reopening the file every __getitem__.'''
        if obj_idx not in self._h5_handles:
            self._h5_handles[obj_idx] = h5py.File(
                self.objects_data[obj_idx]['psf_path'], 'r', swmr=True
            )
        return self._h5_handles[obj_idx]

    def __getitem__(self, idx):
        obj_idx, cx, cy = self.samples[idx]
        obj_info = self.objects_data[obj_idx]

        # 1. Read PSF patch lazily from HDF5 (cached file handle)
        h5f = self._get_h5(obj_idx)
        sss_response = torch.from_numpy(h5f[f'{cx}_{cy}'][()].astype(np.float32))

        # 2. Crop from memmap
        half_w = self.width_bbox // 2
        half_h = self.height_bbox // 2
        sx, sy = int(cx) - half_w, int(cy) - half_h

        patch = np.array(obj_info['data'][:, sy:sy + self.height_bbox, sx:sx + self.width_bbox])
        patch_tensor = torch.from_numpy(patch)  # already float32 from memmap dtype

        # 3. Data augmentations
        if self.data_augmentations is not None:
            patch_tensor = self.data_augmentations(patch_tensor)
            sss_response = self.data_augmentations(sss_response)

        return patch_tensor, sss_response

    def __del__(self):
        for h5f in self._h5_handles.values():
            h5f.close()


###############################################################################
# STRATEGY B: Pre-loaded PSF Tensors — Maximum __getitem__ speed, higher RAM
#
# RAM usage:  All PSF patches stored as contiguous tensors (can be GBs)
# Speed:      Fastest __getitem__ — pure tensor indexing, no disk I/O for PSF
# Best when:  Fewer object configs OR enough RAM to hold all PSF data
###############################################################################

class MultiObject_PSFSSS_Dataset_Preloaded(torch.utils.data.Dataset):
    def __init__(
        self,
        object_configs: list,
        width_bbox: int = 90,
        height_bbox: int = 90,
        data_augmentations = None
    ):
        '''
        Preloaded Strategy: Loads all PSF patches into RAM as contiguous tensors.
        __getitem__ is a pure index lookup — no HDF5 I/O.
        '''
        super().__init__()
        self.width_bbox = width_bbox
        self.height_bbox = height_bbox
        self.data_augmentations = data_augmentations

        sample_lists = []
        self.objects_data = []

        print("Initializing dataset (preloaded mode)...")

        for obj_idx, config in enumerate(object_configs):
            # 1. Load shape metadata & memmap
            shape = tuple(np.load(config['memmap_path'].replace('.dat', '_shape.npy')))
            mmap_img = np.memmap(config['memmap_path'], dtype='float32', mode='r', shape=shape)

            # 2. Read all PSF patches from HDF5 — single pass, build arrays directly
            with h5py.File(config['psf_path'], 'r') as f:
                keys = list(f.keys())
                n_keys = len(keys)

                # Pre-allocate arrays instead of building a dict + converting
                coords = np.empty((n_keys, 2), dtype=np.int32)
                psf_patches = np.empty((n_keys, 90, 90, 3), dtype=np.float32)

                for i, key in enumerate(tqdm(keys, desc=f"Object {obj_idx} PSF")):
                    xc, yc = key.split('_')
                    coords[i] = (int(xc), int(yc))
                    psf_patches[i] = f[key][:]

            x_coords = coords[:, 0]
            y_coords = coords[:, 1]

            # 3. Filter valid coordinates
            valid_mask = (
                (x_coords >= width_bbox // 2) &
                (x_coords < shape[-1] - width_bbox // 2) &
                (y_coords >= height_bbox // 2) &
                (y_coords < shape[-2] - height_bbox // 2)
            )

            valid_coords = coords[valid_mask]
            valid_psf = psf_patches[valid_mask]  # slice keeps it contiguous
            n_valid = len(valid_coords)

            # Build sample index: (obj_idx, local_psf_index)
            # local_psf_index maps into the per-object psf tensor
            obj_samples = np.empty((n_valid, 4), dtype=np.int32)
            obj_samples[:, 0] = obj_idx
            obj_samples[:, 1] = valid_coords[:, 0]  # cx
            obj_samples[:, 2] = valid_coords[:, 1]  # cy
            obj_samples[:, 3] = np.arange(n_valid)   # local index into psf tensor
            sample_lists.append(obj_samples)

            # Store per-object: memmap + contiguous PSF tensor (no dict overhead)
            self.objects_data.append({
                'data': mmap_img,
                'psf_responses': torch.from_numpy(valid_psf),  # (N_valid, 90, 90, 3)
                'shape': shape,
            })

            print(f"Object {obj_idx}: {n_valid} valid samples, "
                  f"PSF tensor {valid_psf.shape} = {valid_psf.nbytes / 1e6:.1f} MB")

        self.samples = np.concatenate(sample_lists, axis=0)
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_idx, cx, cy, psf_local_idx = self.samples[idx]
        obj_info = self.objects_data[obj_idx]

        # 1. PSF response: direct tensor index (zero-copy, no disk I/O)
        sss_response = obj_info['psf_responses'][psf_local_idx]  # (90, 90, 3) float32

        # 2. Crop from memmap
        half_w = self.width_bbox // 2
        half_h = self.height_bbox // 2
        sx, sy = int(cx) - half_w, int(cy) - half_h

        patch = np.array(obj_info['data'][:, sy:sy + self.height_bbox, sx:sx + self.width_bbox])
        patch_tensor = torch.from_numpy(patch)

        # 3. Data augmentations
        if self.data_augmentations is not None:
            patch_tensor = self.data_augmentations(patch_tensor)
            sss_response = self.data_augmentations(sss_response)

        return patch_tensor, sss_response


"""
from psp_multiobject_PSF_postSSL_training import MultiObject_PSFSSS_Dataset_LazyHDF5, MultiObject_PSFSSS_Dataset_Preloaded

# Define your objects-
object_configs = [
     {'memmap_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/3D_Scans/Apple_green/Frontview_4step_demosaiced/stacked_psp.dat', 'psf_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/Footprint_GT_captures/Apple_green/Frontview_demosaiced/sssfootprint_90x90_patch.h5'},
     {'memmap_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/3D_Scans/Apple_green/Backview_4step_demosaiced/stacked_psp.dat', 'psf_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/Footprint_GT_captures/Apple_green/Backview_demosaiced/sssfootprint_90x90_patch.h5'},
     {'memmap_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/3D_Scans/Apple_green/Sideview1_4step_demosaiced/stacked_psp.dat', 'psf_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/Footprint_GT_captures/Apple_green/Sideview1_demosaiced/sssfootprint_90x90_patch.h5'},
     {'memmap_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/3D_Scans/Apple_green/Sideview2_4step_demosaiced/stacked_psp.dat', 'psf_path': '/graphics/scratch3/staff/majumdar/CVPR_Work/Footprint_GT_captures/Apple_green/Sideview2_demosaiced/sssfootprint_90x90_patch.h5'}
]

width_bbox = height_bbox = 90


###############################################################################
# Original class name kept as alias — pick your strategy here:
###############################################################################

MultiObject_PSFSSS_Dataset = MultiObject_PSFSSS_Dataset_LazyHDF5        # low RAM
# MultiObject_PSFSSS_Dataset = MultiObject_PSFSSS_Dataset_Preloaded     # fast getitem

dataset = MultiObject_PSFSSS_Dataset(
    object_configs=object_configs,
    width_bbox=width_bbox,
    height_bbox=height_bbox,
    data_augmentations=None
)

print(f'\nlength of dataset = {len(dataset)}\n')

import torch

# Standard DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=256, 
    shuffle=True, 
    # Start with 4-8. Only go to 20 if you have a massive RAID array 
    # and 8 workers leaves the GPU idle.
    num_workers=8, 
    # Always True for GPU training. 
    # Enables fast DMA transfer to VRAM.
    pin_memory=True,
    # Optional: If your epochs are short, this keeps workers alive 
    # avoiding re-initialization overhead.
    # persistent_workers=True 
)


(x, y) = next(iter(dataloader))

# x.shape, y.shape
# (torch.Size([256, 24, 90, 90]), torch.Size([256, 90, 90, 3]))

x_vis = x.reshape(-1, 8, 3, 90, 90).permute(0,1,3,4,2)

# x_vis.shape
# torch.Size([256, 8, 90, 90, 3])


import matplotlib.pyplot as plt

for i in range(8):
    plt.subplot(4, 2, i + 1)
    plt.imshow(x_vis[100][i] ** (1./2.2))
    plt.axis('off')
plt.savefig('x.png', dpi=400)

plt.imshow(y[100] ** (1./2.2))
plt.savefig('y.png', dpi=400)
"""

