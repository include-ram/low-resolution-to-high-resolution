# Low Resolution to High Resolution — Super Resolution with Multi-GPU Training

This project implements **image super-resolution** (SR) — enhancing low-resolution images to high-resolution ones at a **4x scale factor** — using two deep learning architectures trained with PyTorch distributed training.

---

## Table of Contents

- [Overview](#overview)
- [Architectures](#architectures)
  - [ESRGAN (RRDB-based)](#1-esrgan-rrdb-based)
  - [VQGAN + Transformer](#2-vqgan--transformer)
- [Distributed Training Strategies](#distributed-training-strategies)
  - [DDP (DistributedDataParallel)](#ddp-distributeddataparallel)
  - [FSDP (Fully Sharded Data Parallel)](#fsdp-fully-sharded-data-parallel)
- [Scripts](#scripts)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Loss Functions](#loss-functions)
- [Outputs](#outputs)
- [Multi-GPU Scaling Tips](#multi-gpu-scaling-tips)
- [Troubleshooting](#troubleshooting)

---

## Overview

Super-resolution is the task of reconstructing a high-resolution (HR) image from a low-resolution (LR) counterpart. This project trains GAN-based models to learn this mapping and evaluates their scalability across 1, 2, and 4 GPUs.

**Pipeline:**
```
Low-Resolution Image (e.g. 12x12)
        ↓
   SR Model (4x upscale)
        ↓
High-Resolution Image (e.g. 48x48)
```

---

## Architectures

### 1. ESRGAN (RRDB-based)

**Files:** `test_minimal_rrdb (4).py`, `esrgan_fsdp (6).py`

ESRGAN (Enhanced Super-Resolution GAN) uses **Residual-in-Residual Dense Blocks (RRDB)** as the core building block of the generator.

#### Generator
- **Input:** Low-resolution image (3 channels)
- **Initial convolution:** 3 → 32 feature maps
- **Dense Blocks:** Each block applies 4 convolutional layers with dense (concatenated) connections:
  ```
  Input → Conv1(→16) → Conv2(Input+16→16) → Conv3(Input+32→16) → Conv4(Input+48→32) → Scaled Residual + Input
  ```
  This dense connectivity allows the model to reuse features from all previous layers, leading to better gradient flow and richer representations.
- **Trunk convolution + residual:** Adds the block output back to the initial features
- **Upsampling (2x → 2x = 4x total):** Nearest-neighbor upscaling followed by convolutions and LeakyReLU
- **Output:** Super-resolved image (3 channels)

#### Discriminator
A lightweight CNN that distinguishes real HR images from generated SR images:
```
Conv(3→16) → Conv(16→32) → Conv(32→64) → AdaptiveAvgPool → Linear(64→1)
```

#### Feature Extractor (VGG19)
Uses the first few layers of a pretrained VGG19 to extract perceptual features for content loss. The parameters are frozen during training.

---

### 2. VQGAN + Transformer

**Files:** `vqgan_transformer_sr_distributed_fixed (2).py`, `vqgan_transformer_fsdp (1).py`

This is a two-stage architecture combining a **Vector Quantized GAN (VQGAN)** with a **Transformer** for super-resolution.

#### Stage 1 — VQGAN

The VQGAN learns a discrete codebook representation of image patches.

**Encoder:**
- Downsamples the input image into a latent feature map
- Each spatial location is mapped to one of `1024` codebook vectors (each of size 128)
- This discretization forces the model to learn compact, meaningful visual tokens

**Decoder:**
- Reconstructs the image from codebook indices
- Uses transposed convolutions to upsample back to the original resolution

**Discriminator (PatchGAN):**
- Operates on image patches rather than the whole image
- Encourages realistic local textures

**Perceptual Loss:**
- Uses VGG features to maintain structural similarity between the reconstructed and real images

#### Stage 2 — Transformer

After training the VQGAN:
- LR images are encoded into discrete token sequences via the VQGAN encoder + quantizer
- HR images are similarly tokenized to produce target sequences
- A **Transformer** (with multi-head self-attention) learns to map LR token sequences → HR token sequences
- The predicted HR tokens are then decoded by the VQGAN decoder into the final HR image

This approach frames super-resolution as a **sequence-to-sequence** problem, enabling the Transformer to model long-range dependencies across image regions.

---

## Distributed Training Strategies

### DDP (DistributedDataParallel)

**Used in:** `test_minimal_rrdb (4).py`, `vqgan_transformer_sr_distributed_fixed (2).py`

- Each GPU holds a **full copy** of the model
- Gradients are **synchronized (all-reduced)** across all GPUs after each backward pass
- Straightforward to implement and debug
- Best for models that fit comfortably in a single GPU's memory

```
GPU 0: [Full Model] ←─ gradient sync ─→ GPU 1: [Full Model]
```

### FSDP (Fully Sharded Data Parallel)

**Used in:** `esrgan_fsdp (6).py`, `vqgan_transformer_fsdp (1).py`

- Model parameters, gradients, and optimizer states are **sharded across all GPUs**
- Each GPU only stores a fraction of the model at any time
- Parameters are gathered on-demand during the forward/backward pass
- Enables training of **much larger models** that would not fit on a single GPU
- Supports optional **CPU offloading** for further memory savings

```
GPU 0: [Shard 0] ←─ all-gather during forward ─→ GPU 1: [Shard 1]
```

**FSDP-specific options:**

| Option | Description |
|--------|-------------|
| `--use_mixed_precision` | Use FP16 for forward/backward, FP32 for optimizer |
| `--cpu_offload` | Offload optimizer states and gradients to CPU RAM |
| `--sharding_strategy` | `FULL_SHARD` (default), `SHARD_GRAD_OP`, or `NO_SHARD` |
| `--min_params_for_wrap` | Minimum parameter count for auto-wrapping sub-modules |

---

## Scripts

| Script | Model | Strategy | Notes |
|--------|-------|----------|-------|
| `test_minimal_rrdb (4).py` | ESRGAN / RRDB | DDP | Lightweight, good for testing |
| `esrgan_fsdp (6).py` | ESRGAN / RRDB | FSDP | Optimized for multi-GPU memory efficiency |
| `vqgan_transformer_sr_distributed_fixed (2).py` | VQGAN + Transformer | DDP | Two-stage training |
| `vqgan_transformer_fsdp (1).py` | VQGAN + Transformer | FSDP | Most memory-efficient for large models |

The `.txt` files (`ERSGAN DDP.txt`, `VQGAN_TRANS_SR_DDP.txt`) contain additional notes and configuration guidance for each architecture.

---

## Dataset Structure

The scripts expect paired LR/HR image datasets at the following paths (configurable via arguments):

```
SR_dataset/
├── high_resolution/     # Ground truth HR images (.png / .jpg / .jpeg)
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── low_resolution/      # Corresponding LR images (same filenames)
    ├── img_001.png
    ├── img_002.png
    └── ...
```

- Files are matched by **sorted order**, so filenames must correspond between the two folders.
- The ESRGAN scripts use up to **100 images** for quick testing. Remove the `[:100]` slice for full dataset training.
- HR images are resized to `crop_size x crop_size` (default 48); LR images to `(crop_size / scale) x (crop_size / scale)`.

---

## Installation

```bash
pip install torch torchvision
pip install Pillow numpy matplotlib einops
```

> `einops` is required only for the VQGAN+Transformer scripts.

Verify your PyTorch installation supports CUDA and NCCL:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## How to Run

### Single GPU

```bash
python test_minimal_rrdb\ \(4\).py --world_size 1
```

```bash
python esrgan_fsdp\ \(6\).py --world_size 1
```

```bash
python vqgan_transformer_sr_distributed_fixed\ \(2\).py --world_size 1
```

### 2 GPUs (DDP / FSDP)

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=65432 \
    "test_minimal_rrdb (4).py" --world_size 2
```

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=65432 \
    "esrgan_fsdp (6).py" --world_size 2
```

### 4 GPUs (DDP / FSDP)

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=65432 \
    "vqgan_transformer_sr_distributed_fixed (2).py" --world_size 4
```

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=65432 \
    "vqgan_transformer_fsdp (1).py" --world_size 4 \
    --use_mixed_precision --cpu_offload --min_params_for_wrap 100000
```

### With custom data paths

```bash
python "vqgan_transformer_sr_distributed_fixed (2).py" \
    --hr_dir /path/to/high_resolution \
    --lr_dir /path/to/low_resolution \
    --output_dir /path/to/output \
    --batch_size 4 \
    --world_size 2
```

---

## Configuration

### ESRGAN (`test_minimal_rrdb`, `esrgan_fsdp`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale` | 4 | Upscaling factor (LR → HR) |
| `batch_size` | 4 (FSDP) / 64 (DDP) | Images per GPU per step |
| `crop_size` | 48 | HR patch size |
| `num_epochs` | 5 | Training epochs |
| `lr_G` | 1e-4 | Generator learning rate |
| `lr_D` | 1e-4 | Discriminator learning rate |
| `lambda_pixel` | 1e-2 | Weight for pixel-wise MSE loss |
| `lambda_content` | 1.0 | Weight for VGG perceptual loss |
| `lambda_adv` | 5e-3 | Weight for adversarial loss |
| `generator_blocks` | 4 | Number of RRDB blocks |
| `gradient_accumulation_steps` | 2/4/8 per GPU count | Effective batch scaling |

### VQGAN + Transformer (`vqgan_transformer_*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `low_res` | 64 | LR image size |
| `high_res` | 256 | HR image size |
| `codebook_size` | 1024 | Number of discrete visual tokens |
| `embedding_dim` | 128 | Codebook vector dimension |
| `transformer_dim` | 128 | Transformer hidden size |
| `transformer_depth` | 4 | Number of Transformer layers |
| `transformer_heads` | 8 | Number of attention heads |
| `vqgan_epochs` | 2 (quick) / 100 (full) | Epochs for Stage 1 |
| `transformer_epochs` | 2 (quick) / 80 (full) | Epochs for Stage 2 |

---

## Loss Functions

All scripts use a combination of three losses:

### 1. Pixel Loss (MSE / L1)
Measures direct pixel-level difference between the generated SR image and the ground truth HR image.
```
L_pixel = ||SR - HR||²
```
Ensures the output is spatially accurate but can produce slightly blurry results on its own.

### 2. Perceptual / Content Loss (VGG)
Compares high-level features extracted from a pretrained VGG19 network.
```
L_content = ||VGG(SR) - VGG(HR)||₁
```
Encourages the output to have perceptually similar textures and structures to the ground truth.

### 3. Adversarial Loss (Relativistic GAN)
Uses a relativistic discriminator that considers *how much more realistic* real images are compared to fake ones.
```
L_adv = E[D(HR) - mean(D(SR))] + E[D(SR) - mean(D(HR))]
```
Pushes the generator to produce sharp, realistic high-frequency details.

**Combined generator loss:**
```
L_G = λ_pixel · L_pixel + λ_content · L_content + λ_adv · L_adv
```

---

## Outputs

All results are saved under the `output/` directory:

```
output/
├── samples/                         # Visual comparison images per epoch
│   ├── sample_epoch1_img1.png       # [LR (upscaled) | SR | HR] side-by-side
│   └── ...
├── checkpoints/                     # Model weights (VQGAN / Transformer)
│   ├── vqgan_epoch_10.pt
│   └── transformer_epoch_10.pt
├── plots/
│   └── performance_scaling.png      # Training time & speedup vs GPU count
└── performance_results.json         # Timing metrics across GPU configs
```

Each sample image is a **3-panel comparison**:
- **Left:** LR image (bicubic upscaled for reference)
- **Middle:** SR image (model output)
- **Right:** HR image (ground truth)

---

## Multi-GPU Scaling Tips

When scaling from 1 to 4 GPUs, adjust the following for best efficiency:

### ESRGAN FSDP
```python
'gradient_accumulation_steps': {
    1: 2,   # 1 GPU
    2: 4,   # 2 GPUs
    4: 8,   # 4 GPUs
},
'batch_size': 4,
'num_workers': 4  # or 4 * NUM_GPUS
```

### VQGAN Transformer FSDP
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=65432 \
    "vqgan_transformer_fsdp (1).py" --world_size 4 \
    --use_mixed_precision \
    --cpu_offload \
    --min_params_for_wrap 100000
```

**General rules:**
- Keep per-GPU `batch_size` constant; increase `gradient_accumulation_steps` as GPU count grows to maintain stable training dynamics.
- Scale `num_workers` with GPU count (e.g., `4 × num_gpus`).
- Use `--use_mixed_precision` for ~2x memory savings and faster compute on Tensor Core GPUs.
- Use `--cpu_offload` only if GPU memory is the bottleneck — it trades speed for memory.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NCCL initialization fails | Set `NCCL_P2P_DISABLE=1` and `NCCL_SOCKET_IFNAME=eth0` |
| `CUDA out of memory` | Reduce `batch_size`, enable `--cpu_offload`, or reduce model size |
| Processes hang at init | Check that `master_port` is not in use; try a different port |
| Loss is NaN | Lower learning rate or check for zero-sized batches |
| FSDP checkpoint error | Use `StateDictType.FULL_STATE_DICT` for saving/loading across GPU counts |
| Port already in use | Change `--master_port` to an unused port (e.g., 29502, 65432) |

**Useful NCCL environment variables:**
```bash
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO  # for verbose NCCL logging
```

---

## Summary

| | ESRGAN (DDP) | ESRGAN (FSDP) | VQGAN+Transformer (DDP) | VQGAN+Transformer (FSDP) |
|--|--|--|--|--|
| **Model size** | Small | Small-Medium | Large | Large |
| **Memory usage** | High per GPU | Low per GPU | High per GPU | Low per GPU |
| **Training stages** | 1 | 1 | 2 (VQGAN then Transformer) | 2 (VQGAN then Transformer) |
| **Best for** | Baseline / testing | Multi-GPU efficiency | High-quality SR | Large-scale multi-GPU |
| **Parallelism** | Gradient sync | Parameter sharding | Gradient sync | Parameter sharding |
