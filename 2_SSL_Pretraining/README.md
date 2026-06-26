
# 🧠 Stage 2 — Self-Supervised Pretraining (SimSiam)

Pretrain an encoder with **SimSiam** on patches from captured, multi-view objects — comparing standard ImageNet-style augmentations against our novel phase-shift-profilometry (PSP) augmentations.


---

## 📖 Overview

This directory contains the **self-supervised pretraining** stage of our pipeline. We pretrain an encoder with **SimSiam** on patches extracted from our captured, multi-view objects, and compare two augmentation strategies:

1. 🖼️ **Standard ImageNet-style augmentations** — the conventional SSL recipe (random resized crop, color jitter, grayscale, Gaussian blur, flips).
2. ✨ **Our novel PSP augmentations** — augmentations designed specifically for phase-shift image stacks (salt-and-pepper noise, Gaussian blur / phase jitter, PSP flips, and phase-shift shuffling).

> 🔀 Both strategies are trained with the **same** training script, encoder, and SimSiam objective; only the dataset/augmentation module differs, selected via the `--aug` flag.

## 📂 Files

| | File | Role |
|:--:|------|------|
| 🚀 | `simsiam_pretraining_torchrun.py` | Main multi-node / multi-GPU SimSiam training script (launched with `torchrun`). |
| 🗂️ | `multiobject_ssl_dataset.py` | Dataset module with both augmentation pipelines: `ImageNetTransforms_weakstrong` (ImageNet-style) and `PSPTransforms_weakstrong` (our novel PSP augmentations), plus the shared `MultiObjectSSS_Dataset`. |
| 🏗️ | `increased_Att_PreactRes_UNet_pxshuffle.py` | The encoder backbone (pre-activation residual U-Net encoder with PixelUnshuffle downsampling). |
| ⚙️ | `object_configs_example.json` | Example data-config file listing the per-object data paths. |

## 🗃️ Data layout

Each object/view is stored as a **memory-mapped float32 array** of stacked phase-shift images (`*.dat`), accompanied by:

- 📐 a shape file (`<name>_shape.npy`) with shape `(2 * num_phase_shifts * 3, H, W)` — i.e. height- and width-direction phase shifts, each with 3 RGB channels, and
- 📍 a coordinate file (`detected_xy_coords.npy`) of valid `(x, y)` patch centers detected on the object's footprint.

The dataset classes keep the heavy image data on disk (memory-mapped) and read only the requested patch bytes in `__getitem__`, so startup is fast and RAM usage stays low regardless of the number of objects.

### ⚙️ Config file

Data paths are **not** hardcoded. They are passed via a JSON file (see [`object_configs_example.json`](object_configs_example.json)): a list of per-object entries, each with a `memmap_path` and the matching `mask_path`:

```json
[
    {"memmap_path": "/path/to/Apple_green/Frontview/stacked_psp.dat", "mask_path": "/path/to/Apple_green/Frontview/detected_xy_coords.npy"},
    {"memmap_path": "/path/to/Pear/Frontview/stacked_psp.dat",        "mask_path": "/path/to/Pear/Frontview/detected_xy_coords.npy"}
]
```

Add one entry per object/view; the dataset is simply the union of all valid patches across every entry.

## 🔬 What gets used for pretraining

`multiobject_ssl_dataset.py` builds the dataset that is finally used for SimSiam pretraining. Its `MultiObjectSSS_Dataset`:

1. Reads every object/view listed in the config file and memory-maps its stacked PSP `.dat` array.
2. Computes the list of valid patch centers from each `detected_xy_coords.npy`, discarding coordinates whose crop would fall outside the image bounds.
3. In `__getitem__`, crops a `bbox_size × bbox_size` patch of shape `(num_phase_shifts * 2, 3, H, W)`, then delegates to the selected **weak/strong transform**, which produces a *weak* view and a *strong* view of the same patch — the two augmented views SimSiam compares.

The result is a single large patch dataset pooled across **all objects and all views**. The two views (`x_weak`, `x_strong`) are each reshaped to `(num_phase_shifts * 2 * 3, H, W)` and fed to the encoder as the two SimSiam branches.

The augmentation pipeline is chosen by the `--aug` flag: `imagenet` uses `ImageNetTransforms_weakstrong` (the conventional SSL recipe), while `psp` uses `PSPTransforms_weakstrong` (the novel augmentations introduced in our paper). Both live in `multiobject_ssl_dataset.py` and share the same `(weak, strong)` interface, so the dataset stays augmentation-agnostic.

## ⚙️ Requirements

- 🐍 Python 3.9+
- 📦 PyTorch (with CUDA + NCCL), torchvision, NumPy, Pillow — see the combined [`requirements.txt`](../requirements.txt) at the repo root.
- 🗄️ A shared filesystem across nodes (for checkpoints) when training multi-node.

```bash
pip install -r ../requirements.txt
```

> ⚠️ **Note:** install a CUDA-enabled PyTorch build matching your system's CUDA version (see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)); the default PyPI wheel may be CPU-only.

## 🚀 Running

Training is launched with `torchrun`, which supplies the distributed configuration (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`) via environment variables. The script reads them automatically.

### 🖼️ 1. ImageNet-style augmentations

To pretrain with the standard ImageNet-style SSL augmentations on the captured, multi-view objects:

```bash
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 \
    simsiam_pretraining_torchrun.py \
    --aug imagenet \
    --config object_configs_example.json \
    --output-dir ./outputs_imagenet \
    --batch-size 200 --epochs 200 --warmup-epochs 20 --init-lr 0.05
```

### ✨ 2. Our novel PSP augmentations

To pretrain with our novel data augmentations on the **same** objects, switch the `--aug` flag to `psp`:

```bash
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 \
    simsiam_pretraining_torchrun.py \
    --aug psp \
    --config object_configs_example.json \
    --output-dir ./outputs_psp \
    --batch-size 200 --epochs 200 --warmup-epochs 20 --init-lr 0.05 \
    --base-crop-size 80 --bbox-size 90 --num-phase-shifts 4
```

### 🌐 Multi-node example

On each node, run `torchrun` with the shared rendezvous endpoint (replace `MASTER_IP`, and set `--node_rank` per node):

```bash
# 2 nodes, 4 GPUs each (run on every node, changing --node_rank)
torchrun \
    --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=MASTER_IP --master_port=12355 \
    simsiam_pretraining_torchrun.py \
    --aug imagenet \
    --config object_configs_example.json \
    --output-dir /shared/outputs_imagenet
```

### ⏯️ Resuming

A resume checkpoint is written every epoch to the output directory. To continue an interrupted run, add `--resume` with the same arguments:

```bash
torchrun --standalone --nproc_per_node=4 \
    simsiam_pretraining_torchrun.py \
    --aug imagenet --config object_configs_example.json \
    --output-dir ./outputs_imagenet --resume
```

## 🎛️ Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | *(required)* | JSON file of per-object `{memmap_path, mask_path}` entries. |
| `--output-dir` | *(required)* | Where checkpoints and the training history are written. |
| `--aug` | `imagenet` | Augmentation pipeline: `imagenet` or `psp`. |
| `--init-weights` | `None` | Optional path to a saved initial `state_dict` (loaded on rank 0, then broadcast to all ranks). |
| `--batch-size` | `200` | Per-GPU batch size. |
| `--epochs` | `200` | Total training epochs. |
| `--warmup-epochs` | `20` | Linear LR warmup epochs (cosine decay afterward). |
| `--init-lr` | `0.05` | Initial (peak) learning rate; no linear batch-size scaling is applied. |
| `--num-workers` | `8` | DataLoader workers per GPU. |
| `--bbox-size` | `90` | Square crop size (H = W) fed to the encoder. |
| `--base-crop-size` | `80` | PSP only: initial random crop size before resizing to `--bbox-size`. |
| `--num-phase-shifts` | `4` | Phase shifts per direction; input channels = `num_phase_shifts * 2 * 3`. |
| `--resume` | off | Resume from the latest resume checkpoint in `--output-dir`. |

## 📤 Outputs

Written to `--output-dir` (suffixed by the chosen `--aug`):

- 🏆 `simsiam_encoder_<aug>.pth` — best encoder weights (lowest training loss).
- ⏯️ `simsiam_resume_ckpt_<aug>.pth` — latest full checkpoint (model + optimizer + history) for resuming.
- 📈 `simsiam_encoder_<aug>_trainhist.pkl` — per-epoch training history (loss, LR, and `z`-vector std used to monitor representational collapse).

➡️ The encoder checkpoint feeds **[Stage 3 — Decoder Training & Visualization](../3_Post_SSL_Pretraining/README.md)**.
