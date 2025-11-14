# Environment Setup Guide for Open-Sora v1.3

This directory contains scripts to set up the environment for running the naive fine-tuning experiment on a cluster.

## Execution Order

Follow these scripts **in order** (each depends on the previous):

### 0. Setup Scratch Environment Variables (AUTOMATIC!)
**File:** `00_set_scratch_env.sh`
- Redirects ALL installations, caches, and temp files to `/scratch/wc3013`
- Prevents filling up `/home` directory
- Sets up organized cache directory structure
- **Note:** This is now automatically loaded by all scripts below!

**Run on:** (Not needed - auto-loaded by scripts)

**Manual usage** (if needed):
```bash
source env_setup/00_set_scratch_env.sh
```

### 1. Create Conda Environment and Install All Dependencies
**Files:** 
- `01_conda_env_install_torch.sh` (shell script version)
- `02_create_conda_env.sbatch` (SLURM batch job version, **recommended**)

**What it does:**
- **Auto-loads scratch environment** - no manual sourcing needed!
- Creates conda environment with Python 3.9 in `/scratch/wc3013/conda-envs/`
- **Installs all dependencies in correct order to avoid version conflicts:**
  - NumPy 1.26.4 (NumPy 2.x breaks PyTorch 2.2.2)
  - PyTorch 2.2.2 + CUDA 12.1
  - xformers 0.0.25.post1
  - bitsandbytes 0.43.3 (with `--no-deps` to prevent upgrades)
  - PyAV (av) via conda for video processing
  - All Open-Sora requirements
  - Eval and VAE requirements
- **Verifies all package versions are correct**
- Configures conda to use scratch directories permanently

**Run on:** Login node or submit as batch job (recommended for reliability)

**Option A: SLURM batch job (recommended)**
```bash
cd env_setup
sbatch 02_create_conda_env.sbatch

# Monitor progress
tail -f slurm_create_env.out
```

**Option B: Direct shell execution**
```bash
bash env_setup/01_conda_env_install_torch.sh
```

**Note:** The sbatch version is more reliable for long-running installations and provides better logging.

### 2. Build Flash Attention and Apex (OPTIONAL - Can Be Skipped)

**⚠️ IMPORTANT: This step is OPTIONAL and can be skipped entirely.**

Open-Sora v1.3 works perfectly fine without flash-attn and apex. These packages only provide speed optimizations (2-3x faster) but are not required for correctness.

**Why are they difficult to install?**
- **flash-attn**: Requires exact CUDA toolkit version match with PyTorch's CUDA version. The build process looks for `nvcc` in PyTorch's package directory, but conda-installed PyTorch doesn't include the full CUDA toolkit there. Additionally, it must be compiled for your specific GPU architecture (H100/H200 = compute capability 9.0), which requires the CUDA toolkit to be available during build.
  
- **apex**: NVIDIA's apex library has strict CUDA version requirements and often fails when there's a mismatch between the CUDA version used to build PyTorch (12.1 in our case) and the CUDA toolkit available on the system. The build process is fragile and frequently breaks with newer CUDA versions.

**If you want to try building them anyway:**

**File:** `02_flsh_attn_apex_build.sbatch`
- Builds flash-attn 2.5.8 (must match GPU architecture)
- Builds apex (for FusedLayerNorm, FusedAdam)
- Installs Open-Sora in development mode

**Run on:** GPU node (submit via SLURM)

```bash
sbatch env_setup/02_flsh_attn_apex_build.sbatch
```

**Key Configuration:**
- Line 46: `TORCH_CUDA_ARCH_LIST="90"` for H100/H200
- Builds to `${SCRATCH_BASE}/wheels/`

**If builds fail:** Don't worry! Just ensure your configs have:
- `enable_flash_attn=False`
- `enable_layernorm_kernel=False`

The code will automatically fall back to:
- Standard PyTorch attention (or SDPA if available)
- Standard `nn.LayerNorm` instead of FusedLayerNorm

### 3. Download and Preprocess UCF-101 Dataset
**Directory:** `download_ucf101/`
- Downloads UCF-101 dataset (101 action classes, 13,320 videos)
- Performs stratified sampling (2,000 videos, ~20 per class)
- Preprocesses to 640×480, 24fps, 45 frames
- Generates metadata CSV for training

**Run on:** Download on login node, preprocessing via SLURM batch job

```bash
# Download and sample dataset
cd env_setup/download_ucf101
python download_ucf101.py

# Preprocess videos (submit as batch job)
sbatch preprocess_ucf101.sbatch
```

See `download_ucf101/README.md` for detailed instructions.

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
- Note: detectron2 is **skipped** (only needed for OCR scoring, requires special compilation)

### Additional Tools
- huggingface-hub (for checkpoint download)
- datasets (for UCF-101 download from Hugging Face)
- xformers==0.0.25.post1
- numpy<2.0.0

### Video Processing (Critical for UCF-101)
- av (PyAV) - installed via conda for better ffmpeg compatibility
- opencv-python (cv2) - from requirements-vae.txt
- pandas - for CSV metadata
- tqdm - for progress bars

### Compiled Extensions
- flash-attn==2.5.8 (built with CUDA 12.2, PyTorch is cu121 - may have version mismatch)
- apex (from NVIDIA) - **NOTE: Build may fail due to CUDA 12.1/12.2 mismatch**

### System Tools
- ffmpeg>=6,<7

## Quick Start

```bash
# 1. Create environment with all dependencies (recommended: SLURM batch job)
cd env_setup
sbatch 02_create_conda_env.sbatch
# Monitor: tail -f slurm_create_env.out

# 2. (OPTIONAL) Build flash-attn and apex for speed optimization
# Skip this step if you encounter build errors - it's not required!
# sbatch 02_flsh_attn_apex_build.sbatch

# 3. Download and preprocess UCF-101 dataset
cd download_ucf101
sbatch preprocess_ucf101.sbatch
# Monitor: tail -f slurm_download_prep_ucf101.out

# 4. Run naive experiment
cd ../../naive_experiment/scripts
sbatch run_experiment.sbatch
```

**Note:** 
- All scripts automatically load the scratch environment configuration!
- Step 2 (flash-attn/apex) is optional and can be skipped entirely
- The experiment will work fine without these packages, just 2-3x slower

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

### Flash-attn or apex build failures
**This is expected and OK!** These packages have strict requirements:

**Common errors:**
- `FileNotFoundError: nvcc` - PyTorch's conda package doesn't include the full CUDA toolkit
- `CUDA version mismatch` - apex requires exact CUDA version alignment
- `fatal: not a git repository` - build process expects git context

**Solution:** Skip these packages entirely!
1. Don't run `02_flsh_attn_apex_build.sbatch`
2. Ensure all your config files have:
   ```python
   enable_flash_attn=False
   enable_layernorm_kernel=False
   ```
3. The code will automatically use standard PyTorch implementations
4. Your experiment will run correctly, just 2-3x slower

### "No module named 'flash_attn'" during inference
- This means your config has `enable_flash_attn=True` but flash-attn isn't installed
- Change to `enable_flash_attn=False` in your config file
- The naive experiment configs already have this set correctly

### "No module named 'opensora'"
- Run: `cd /scratch/wc3013/open-sora-v1.3-experiment && pip install -e .`
- Or run the `02_flsh_attn_apex_build.sbatch` which includes this step

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

1. **Auto-loading**: All setup scripts (`01`, `03`, `04`) automatically source `00_set_scratch_env.sh` - no manual sourcing needed!

2. **Manual sourcing**: If you need the env vars in your current shell for custom commands:
   ```bash
   source env_setup/00_set_scratch_env.sh
   ```

3. **Re-source after logout**: If you SSH logout/login, you need to re-source if doing manual work

4. **For SLURM jobs**: Add to your sbatch script:
   ```bash
   source env_setup/00_set_scratch_env.sh
   ```

5. **All caches go to scratch**: Torch, HuggingFace, pip, conda, everything!

