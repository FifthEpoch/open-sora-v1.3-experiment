# Environment Setup Guide for Open-Sora v1.3

This directory contains scripts to set up the environment for running the naive fine-tuning experiment on a cluster.

## Execution Order

Follow these scripts **in order** (each depends on the previous):

### 0. Setup Scratch Environment Variables (MUST RUN FIRST!)
**File:** `00_set_scratch_env.sh`
- Redirects ALL installations, caches, and temp files to `/scratch/wc3013`
- Prevents filling up `/home` directory
- Sets up organized cache directory structure

**Run on:** Login node (before everything else)

```bash
source env_setup/00_set_scratch_env.sh
```

**Important:** You must `source` (not `bash`) this script to export variables to your shell!

### 1. Create Conda Environment and Install PyTorch
**File:** `01_conda_env_install_torch.sh`
- Creates conda environment with Python 3.9
- Installs PyTorch 2.2.2 + CUDA 12.1
- Installs xformers
- Installs basic dependencies from requirements
- **After Step 0**, conda env will be created in `/scratch`

**Run on:** Login node (no GPU needed)

```bash
bash env_setup/01_conda_env_install_torch.sh
```

### 2. Install FFmpeg
**File:** `03_ffmpeg.sh`
- Installs FFmpeg for video processing
- Required by PyAV for HMDB51 preprocessing and inference

**Run on:** Login node (no GPU needed)

```bash
bash env_setup/03_ffmpeg.sh
```

### 3. Build Flash Attention and Apex
**File:** `02_flsh_attn_apex_build.sbatch`
- Builds flash-attn 2.5.8 (must match GPU architecture)
- Builds apex (for FusedLayerNorm, FusedAdam)
- Installs Open-Sora in development mode
- **IMPORTANT:** Edit line 99 to point to your repo path

**Run on:** GPU node (submit via SLURM)

```bash
# Edit the sbatch file first!
nano env_setup/02_flsh_attn_apex_build.sbatch

# Then submit
sbatch env_setup/02_flsh_attn_apex_build.sbatch
```

**Key Configuration:**
- Line 46: `TORCH_CUDA_ARCH_LIST="90"` for H100/H200
- Line 99: Change to your repo path
- Wheels are built in scratch directory automatically

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
# 0. Setup scratch environment (RUN THIS FIRST!)
source env_setup/00_set_scratch_env.sh

# 1. Create environment
bash env_setup/01_conda_env_install_torch.sh

# 2. Install FFmpeg
bash env_setup/03_ffmpeg.sh

# 3. Build compiled extensions (on GPU node)
# Edit 02_flsh_attn_apex_build.sbatch first!
sbatch env_setup/02_flsh_attn_apex_build.sbatch

# 4. Verify
bash env_setup/04_installation_check.sh
```

## Cluster-Specific Notes

### GPU Architecture
Adjust `TORCH_CUDA_ARCH_LIST` in `02_flsh_attn_apex_build.sbatch`:
- H100/H200: `"90"`
- A100: `"80"`
- Both: `"80;90"`

### Path Customization
In `02_flsh_attn_apex_build.sbatch`, update:
- Line 99: Your repo path (critical!)
- Wheels automatically built to scratch directory

### Module System
If your cluster uses environment modules, you may need to load:
- Modern GCC (if default is too old)
- CUDA toolkit (though PyTorch includes CUDA)

## Troubleshooting

### "No module named 'flash_attn'"
- Run the sbatch script on a GPU node with matching architecture
- Check that TORCH_CUDA_ARCH_LIST matches your GPU

### "ffmpeg not found"
- Run `bash env_setup/03_ffmpeg.sh`

### "No module named 'opensora'"
- Run the sbatch script (installs via `pip install -e .`)
- Check that repo path on line 99 is correct

### "decord" import errors
- Decord is installed via requirements-eval.txt
- If issues persist, may need to build from source

## Scratch Storage Setup

To avoid filling up your `/home` directory, all installations and caches are redirected to `/scratch/wc3013`.

### Directory Structure

After running `00_set_scratch_env.sh`, you'll have this organized structure:

```
/scratch/wc3013/
├── conda-envs/          # Conda environments
├── conda-pkgs/          # Conda package cache
├── py-cache/            # Python/PyTorch caches
│   ├── pip/             # Pip package cache
│   ├── torch/           # PyTorch cache
│   │   └── extensions/  # PyTorch extensions
│   ├── models/          # PyTorch model cache
│   ├── triton/          # Triton compiler cache
│   ├── torch-inductor/  # TorchInductor cache
│   ├── colossalaJit/    # ColossalAI JIT cache
│   └── python/          # Python __pycache__
├── hf-cache/            # HuggingFace caches
│   ├── datasets/
│   ├── hub/
│   └── transformers/
├── tmp/                 # Temporary files
├── wandb/               # Weights & Biases cache
├── tensorboard/         # TensorBoard logs
└── wheels/              # Compiled wheel files
```

### Important Notes

1. **Source the script** to get env variables in your current shell:
   ```bash
   source env_setup/00_set_scratch_env.sh
   ```

2. **Re-source after logout**: If you SSH logout/login, you need to re-source the script

3. **For SLURM jobs**: Add to your sbatch script:
   ```bash
   source env_setup/00_set_scratch_env.sh
   ```

4. **All caches go to scratch**: Torch, HuggingFace, pip, conda, everything!

