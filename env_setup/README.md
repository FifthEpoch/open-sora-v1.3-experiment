# Environment Setup Guide for Open-Sora v1.3

This directory contains scripts to set up the environment for running the naive fine-tuning experiment on a cluster.

## Execution Order

Follow these scripts **in order** (each depends on the previous):

### 1. Create Conda Environment and Install PyTorch
**File:** `00_conda_env_install_torch.sh`
- Creates conda environment with Python 3.9
- Installs PyTorch 2.2.2 + CUDA 12.1
- Installs xformers
- Installs basic dependencies from requirements

**Run on:** Login node (no GPU needed)

```bash
bash env_setup/00_conda_env_install_torch.sh
```

### 2. Install FFmpeg
**File:** `02_ffmpeg.sh`
- Installs FFmpeg for video processing
- Required by PyAV for HMDB51 preprocessing and inference

**Run on:** Login node (no GPU needed)

```bash
bash env_setup/02_ffmpeg.sh
```

### 3. Build Flash Attention and Apex
**File:** `01_flsh_attn_apex_build.sbatch`
- Builds flash-attn 2.5.8 (must match GPU architecture)
- Builds apex (for FusedLayerNorm, FusedAdam)
- Installs Open-Sora in development mode
- **IMPORTANT:** Edit line 92 to point to your repo path

**Run on:** GPU node (submit via SLURM)

```bash
# Edit the sbatch file first!
nano env_setup/01_flsh_attn_apex_build.sbatch

# Then submit
sbatch env_setup/01_flsh_attn_apex_build.sbatch
```

**Key Configuration:**
- Line 40: `TORCH_CUDA_ARCH_LIST="90"` for H100/H200
- Line 92: Change to your repo path
- Lines 43-44: Set wheel output directory

### 4. Verify Installation
**File:** `04_installation_check.sh`
- Checks that all dependencies are installed correctly
- Verifies imports work

**Run on:** Login node

```bash
bash env_setup/04_installation_check.sh
```

## Required Dependencies Summary

### Core (from requirements.txt)
- colossalai>=0.4.1
- mmengine>=0.10.3
- pandas>=2.0.3
- timm==0.9.16
- rotary_embedding_torch==0.5.3
- ftfy>=6.2.0
- diffusers==0.29.0
- accelerate==0.29.2
- av>=12.0.0

### VAE (from requirements-vae.txt)
- beartype==0.18.5
- einops==0.8.0
- einops-exts==0.0.4
- opencv-python==4.9.0.80
- pillow==10.3.0

### Evaluation (from requirements-eval.txt)
- decord==0.6.0 (for video I/O)
- pytorchvideo==0.1.5
- lpips==0.1.4
- scikit-image>=0.20.0
- scikit-learn>=1.4.2

### Additional Tools
- huggingface-hub (for checkpoint download)
- xformers==0.0.25.post1
- numpy<2.0.0

### Compiled Extensions
- flash-attn==2.5.8
- apex (from NVIDIA)

### System Tools
- ffmpeg>=6,<7

## Quick Start

```bash
# 1. Create environment
bash env_setup/00_conda_env_install_torch.sh

# 2. Install FFmpeg
bash env_setup/02_ffmpeg.sh

# 3. Build compiled extensions (on GPU node)
# Edit 01_flsh_attn_apex_build.sbatch first!
sbatch env_setup/01_flsh_attn_apex_build.sbatch

# 4. Verify
bash env_setup/04_installation_check.sh
```

## Cluster-Specific Notes

### GPU Architecture
Adjust `TORCH_CUDA_ARCH_LIST` in `01_flsh_attn_apex_build.sbatch`:
- H100/H200: `"90"`
- A100: `"80"`
- Both: `"80;90"`

### Path Customization
In `01_flsh_attn_apex_build.sbatch`, update:
- Line 43-44: Wheel output directory (default: `~/wheels/cu121_sm90`)
- Line 92: Your repo path (critical!)

### Module System
If your cluster uses environment modules, you may need to load:
- Modern GCC (if default is too old)
- CUDA toolkit (though PyTorch includes CUDA)

## Troubleshooting

### "No module named 'flash_attn'"
- Run the sbatch script on a GPU node with matching architecture
- Check that TORCH_CUDA_ARCH_LIST matches your GPU

### "ffmpeg not found"
- Run `bash env_setup/02_ffmpeg.sh`

### "No module named 'opensora'"
- Run the sbatch script (installs via `pip install -e .`)
- Check that repo path on line 92 is correct

### "decord" import errors
- Decord is installed via requirements-eval.txt
- If issues persist, may need to build from source

