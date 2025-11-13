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

# Open-Sora v1.3 tested pins (CUDA 12.1 wheels)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2

pip install --index-url https://download.pytorch.org/whl/cu121 \
  xformers==0.0.25.post1

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