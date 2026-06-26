
# 🎨 Stage 3 — Decoder Training & Visualization

After SimSiam pretraining, **freeze the encoder** and train a decoder to predict per-point subsurface-scattering (SSS) footprints — then render and visualize the results.


---

## 📖 Overview

This stage runs **after** SimSiam pretraining. The pretrained encoder is reused as a frozen feature extractor; a freshly initialized decoder is trained on top of it to regress the `(90, 90, 3)` subsurface-scattering pixel-footprint response for every point on an object.

```
🧠 SimSiam pretrained encoder  ──▶  ❄️ freeze encoder + train decoder  ──▶  🎨 render / visualize
```

Three runnable scripts cover the workflow:

1. 🏋️ **Train** the decoder on the frozen encoder — `sss_psf_decoderonly_training_simsiam_pretrained.py`.
2. 💡 **Render** a relit output for an object by splatting the trained network's per-point predictions — `splat_trained_network_output.py`.
3. 🌈 **Visualize** the encoder's feature structure as a PCA-RGB map (uses the encoder only, no decoder) — `pca_rgb_visualization.py`.

## 📂 Files

| | File | Role |
|:--:|------|------|
| 🏋️ | `sss_psf_decoderonly_training_simsiam_pretrained.py` | **Main training script.** Loads the frozen SimSiam encoder, attaches a new decoder, and trains it (multi-node / multi-GPU) to predict SSS footprints. |
| 💡 | `splat_trained_network_output.py` | **Renderer.** Runs the trained decoder over all object points and splats the predictions into a relit image. |
| 🌈 | `pca_rgb_visualization.py` | **Visualizer.** Extracts frozen-encoder features for all object points and maps them to an RGB image via PCA. |
| 🏗️ | `Att_PreAct_pxshuffle_UNet_Decoder_Only.py` | *Helper:* the encoder+decoder network architecture. |
| 🗂️ | `psp_multiobject_PSF_postSSL_dataset.py` | *Helper:* dataset classes (lazy-HDF5 and preloaded variants) pairing PSP inputs with GT SSS footprints. |
| 📉 | `cost_functions.py` | *Helper:* loss functions used during decoder training. |
| ⚙️ | `object_configs_example.json` | Example data-config for the training script. |

## 🗃️ Data layout

Each object/view provides:

- 🧊 a **memory-mapped PSP stack** (`stacked_psp.dat` + `stacked_psp_shape.npy`) — the camera-captured phase-shift images that feed the network, and
- 🎯 a **GT SSS footprint HDF5** (`sssfootprint_90x90_patch.h5`) whose keys are `"x_y"` point coordinates and whose values are the ground-truth `(90, 90, 3)` footprints.

We use 4-step phase-shift profilometry, so `num_phase_shifts = 4` and the encoder input has `4 × 2 × 3 = 24` channels.

### ⚙️ Config file

The training script's per-object paths are **not** hardcoded; they come from a JSON file (see [`object_configs_example.json`](object_configs_example.json)) — a list of `{memmap_path, psf_path}` entries:

```json
[
    {"memmap_path": "/path/to/Apple_green/Frontview/stacked_psp.dat", "psf_path": "/path/to/Apple_green/Frontview/sssfootprint_90x90_patch.h5"},
    {"memmap_path": "/path/to/Pear/Frontview/stacked_psp.dat",        "psf_path": "/path/to/Pear/Frontview/sssfootprint_90x90_patch.h5"}
]
```

> 🔎 The two visualization scripts work on a *single* object, so their paths are passed directly as command-line flags (no JSON needed).

## ⚙️ Requirements

- 🐍 Python 3.9+
- 📦 PyTorch (with CUDA + NCCL), torchvision, NumPy, OpenCV, imageio, h5py, tqdm, matplotlib, pytorch-msssim — see the combined [`requirements.txt`](../requirements.txt) at the repo root.
- 🧠 The SimSiam encoder checkpoint produced by the [previous stage](../2_SSL_Pretraining/README.md).

```bash
pip install -r ../requirements.txt
```

> ⚠️ **Note:** install a CUDA-enabled PyTorch build matching your system's CUDA version (see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)); the default PyPI wheel may be CPU-only.

## 🏋️ 1. Train the decoder

Freezes the pretrained SimSiam encoder and trains a new decoder to predict SSS footprints. Launched with `torch.multiprocessing` spawn (multi-node / multi-GPU). The encoder is loaded from any SimSiam checkpoint — pass the ImageNet-aug or PSP-aug encoder via `--encoder-ckpt`.

```bash
# Single node, all visible GPUs
python sss_psf_decoderonly_training_simsiam_pretrained.py \
    --config object_configs_example.json \
    --encoder-ckpt /path/to/simsiam_encoder_imagenet.pth \
    --output-dir ./outputs_decoder \
    --ckpt-name decoder_only.pth \
    --dataset-strategy lazy \
    --batch-size 128 --epochs 300 --lr 0.001
```

> 🌐 Multi-node: set `--num_nodes`, `--node_rank` (per node), `--master_addr` and `--master_port`. Use `--resume` to continue from the checkpoint in `--output-dir`.

### 🎛️ Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | *(required)* | JSON of per-object `{memmap_path, psf_path}` entries. |
| `--encoder-ckpt` | *(required)* | SimSiam checkpoint to load the frozen encoder from. |
| `--output-dir` | *(required)* | Where the decoder checkpoint and training history are written. |
| `--ckpt-name` | `decoder_only.pth` | Filename for the saved decoder checkpoint. |
| `--dataset-strategy` | `lazy` | `lazy` (low RAM, on-the-fly HDF5) or `preloaded` (PSF held in RAM). |
| `--batch-size` | `128` | Per-GPU batch size. |
| `--epochs` | `300` | Total training epochs. |
| `--lr` | `0.001` | Adam learning rate. |
| `--num-workers` | `8` | DataLoader workers per GPU. |
| `--num-phase-shifts` | `4` | Phase shifts per direction (input channels = `n × 2 × 3`). |
| `--resume` | off | Resume from the checkpoint in `--output-dir` (also see `--resume_epoch`). |

## 💡 2. Render the relit output

Runs the trained decoder over every object point and splats the per-point predictions into a single relit image, modulated by the projector colors.

```bash
python splat_trained_network_output.py \
    --decoder-ckpt ./outputs_decoder/decoder_only.pth \
    --psp-corresp-dir /path/to/Soap/Backview_4step \
    --psp-data-dir    /path/to/Soap/Backview_4step_demosaiced \
    --sam-mask        /path/to/SAM_masks/Soap/Backview/object_sam_HQ_output.npy \
    --gt-sss-dir      /path/to/Footprint_GT_captures/Soap/Backview_demosaiced \
    --projector-img   /path/to/projector_GT_images/black_white_checkerboard.png \
    --output          soap_backview_relit.exr
```

> 🖥️ The `--gt-sss-dir` must contain the SSS HDF5 (`--hdf5-filename`, default `sssfootprint_90x90_patch.h5`) and `white_frame.exr`. Add `--no-show` to skip the matplotlib preview on headless machines.

## 🌈 3. PCA-RGB visualization

Uses **only the pretrained encoder** (no decoder). It extracts a feature vector for every object point, reduces them to 3 components via PCA, and maps those to RGB — points with similar subsurface scattering cluster into similar colors.

```bash
python pca_rgb_visualization.py \
    --encoder-ckpt /path/to/simsiam_encoder_psp.pth \
    --psp-data-dir /path/to/Soap/Backview_4step_demosaiced \
    --valid-pts-mask /path/to/SAM_masks/Soap/Backview/object_sam_HQ_output.npy \
    --output soap_backview_pca_rgb.exr
```

## 📤 Outputs

- 🏋️ **Training** → `<output-dir>/<ckpt-name>` (best decoder weights) and `<ckpt-name>_trainhist.pkl` (per-epoch loss history).
- 💡 **Render** → the relit `.exr` given by `--output`.
- 🌈 **PCA-RGB** → the visualization `.exr` given by `--output`.
