#!/bin/bash
# Create conda environment and install PyTorch + dependencies

set -euo pipefail

# Auto-load scratch environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/00_set_scratch_env.sh" ]; then
    echo "Loading scratch environment configuration..."
    source "${SCRIPT_DIR}/00_set_scratch_env.sh"
else
    echo "Warning: ${SCRIPT_DIR}/00_set_scratch_env.sh not found!"
    exit 1
fi

# login node (no GPU)
module purge || true
module load anaconda3/2025.06

# Configure conda to use scratch directories
conda config --add envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --add pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# Create environment in scratch location
conda create -y -p "${SCRATCH_BASE}/conda-envs/opensora13" python=3.9
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

python -m pip install -U pip setuptools wheel

# ============================================================================
# CRITICAL: Install dependencies in strict order to avoid version conflicts
# ============================================================================
# Issue: bitsandbytes has loose dependency specs that can upgrade torch/numpy
# Solution: Install numpy, torch, xformers, THEN bitsandbytes with --no-deps
# ============================================================================

# Step 1: Pin NumPy to 1.x (NumPy 2.x breaks PyTorch 2.2.2 and OpenCV)
echo "Step 1: Installing NumPy 1.26.4..."
pip install 'numpy<2' --no-cache-dir

# Step 2: Install Open-Sora v1.3 tested pins (CUDA 12.1 wheels)
echo "Step 2: Installing PyTorch 2.2.2 and torchvision 0.17.2..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2

echo "Step 3: Installing xformers 0.0.25.post1..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
  xformers==0.0.25.post1

# Step 4: Install bitsandbytes WITHOUT letting it upgrade dependencies
# bitsandbytes 0.43.3 is the last version supporting PyTorch 2.2.x
echo "Step 4: Installing bitsandbytes 0.43.3 (without dependency upgrades)..."
pip install 'bitsandbytes==0.43.3' --no-deps --no-cache-dir

# Install PyAV (av) via conda first (handles ffmpeg dependencies better than pip)
# This is needed for UCF-101 preprocessing and video loading throughout the project
echo "Installing PyAV (av) via conda..."
conda install -y av -c conda-forge

# pull the v1.3-pinned packages (no CUDA build yet)
pip install -r requirements/requirements-cu121.txt

# Install base requirements (includes av>=12.0.0, but conda version takes precedence)
pip install -r requirements/requirements.txt

# Install eval requirements (includes decord, lpips, scikit-image)
# Note: detectron2 is skipped as it's only needed for OCR scoring and requires special compilation
echo "Installing eval requirements (skipping detectron2)..."
pip install imageio>=2.34.1 pyiqa==0.1.10 scikit-learn>=1.4.2 scikit-image>=0.20.0 \
  lvis==0.5.3 boto3>=1.34.113 easydict>=1.9 fairscale>=0.4.13 \
  decord==0.6.0 pytorchvideo==0.1.5 lpips==0.1.4

# Install VAE requirements (includes opencv-python, pillow, einops)
pip install -r requirements/requirements-vae.txt

# Install huggingface-hub and datasets for checkpoint and UCF-101 dataset downloads
pip install huggingface-hub datasets

# ============================================================================
# Verification: Check that all critical versions are correct
# ============================================================================
echo ""
echo "============================================"
echo "Environment Setup Complete - Version Check"
echo "============================================"
python -c "
import sys
import numpy as np
import torch
import torchvision
try:
    import xformers
    xformers_version = xformers.__version__
except:
    xformers_version = 'NOT INSTALLED'
try:
    import bitsandbytes
    bnb_version = bitsandbytes.__version__
except:
    bnb_version = 'NOT INSTALLED'

print(f'Python: {sys.version.split()[0]}')
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'xformers: {xformers_version}')
print(f'bitsandbytes: {bnb_version}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Verify versions are correct
issues = []
if not np.__version__.startswith('1.'):
    issues.append(f'❌ NumPy should be 1.x, got {np.__version__}')
else:
    print('✓ NumPy version correct (1.x)')

if not torch.__version__.startswith('2.2.2'):
    issues.append(f'❌ PyTorch should be 2.2.2, got {torch.__version__}')
else:
    print('✓ PyTorch version correct (2.2.2)')

if xformers_version != '0.0.25.post1':
    issues.append(f'❌ xformers should be 0.0.25.post1, got {xformers_version}')
else:
    print('✓ xformers version correct (0.0.25.post1)')

if bnb_version != '0.43.3':
    issues.append(f'❌ bitsandbytes should be 0.43.3, got {bnb_version}')
else:
    print('✓ bitsandbytes version correct (0.43.3)')

if issues:
    print()
    print('WARNINGS:')
    for issue in issues:
        print(issue)
    sys.exit(1)
else:
    print()
    print('✓ All package versions are correct!')
"

echo "============================================"
echo ""
echo "Environment is ready! Next steps:"
echo "  1. Download and preprocess UCF-101 dataset:"
echo "     cd download_ucf101 && sbatch preprocess_ucf101.sbatch"
echo "  2. Run the naive experiment:"
echo "     cd ../naive_experiment/scripts && sbatch run_experiment.sbatch"
echo ""