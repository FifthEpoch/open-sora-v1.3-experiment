#!/bin/bash
#SBATCH --job-name=create_opensora_env
#SBATCH --output=slurm_create_env.out
#SBATCH --error=slurm_create_env.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

# Create conda environment with all dependencies correctly versioned
# This script consolidates environment creation and dependency installation
# into a single job, eliminating the need for separate fix scripts.

set -euo pipefail

echo "========================================"
echo "Creating OpenSora v1.3 Conda Environment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set base directory (absolute path - SLURM runs script in different location)
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"

# Load environment configuration using absolute path
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# Load anaconda module
echo "Loading anaconda module..."
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Configure conda to use scratch directories
echo "Configuring conda paths..."
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --prepend pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# Check if environment already exists
if conda env list | grep -q "opensora13"; then
    echo ""
    echo "WARNING: opensora13 environment already exists!"
    echo "To recreate from scratch, first remove it with:"
    echo "  conda remove -p ${SCRATCH_BASE}/conda-envs/opensora13 --all"
    echo ""
    echo "Skipping environment creation."
    exit 0
fi

# Create environment
echo ""
echo "Creating conda environment at: ${SCRATCH_BASE}/conda-envs/opensora13"
conda create -y -p "${SCRATCH_BASE}/conda-envs/opensora13" python=3.9

# Activate environment
echo "Activating environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

# Upgrade pip
echo ""
echo "Upgrading pip, setuptools, wheel..."
python -m pip install -U pip setuptools wheel

# ============================================================================
# CRITICAL: Install dependencies in strict order to avoid version conflicts
# ============================================================================
echo ""
echo "========================================"
echo "Installing Core Dependencies"
echo "========================================"
echo "Note: Order matters! Installing in sequence to avoid conflicts."
echo ""

# Step 1: Pin NumPy to 1.x (NumPy 2.x breaks PyTorch 2.2.2 and OpenCV)
echo "Step 1/4: Installing NumPy 1.26.4..."
pip install 'numpy<2' --no-cache-dir

# Step 2: Install PyTorch and torchvision (CUDA 12.1)
echo ""
echo "Step 2/4: Installing PyTorch 2.2.2 and torchvision 0.17.2..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2

# Step 3: Install xformers
echo ""
echo "Step 3/4: Installing xformers 0.0.25.post1..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
  xformers==0.0.25.post1

# Step 4: Install bitsandbytes WITHOUT letting it upgrade dependencies
echo ""
echo "Step 4/4: Installing bitsandbytes 0.43.3 (with --no-deps to prevent upgrades)..."
pip install 'bitsandbytes==0.43.3' --no-deps --no-cache-dir

# Install PyAV via conda (better ffmpeg handling)
echo ""
echo "========================================"
echo "Installing PyAV and Video Dependencies"
echo "========================================"
conda install -y av -c conda-forge

# Install Open-Sora requirements
echo ""
echo "========================================"
echo "Installing Open-Sora Requirements"
echo "========================================"
cd "${PROJECT_ROOT}"

echo "Installing Open-Sora v1.3 CUDA requirements..."
pip install -r requirements/requirements-cu121.txt

echo "Installing base requirements..."
pip install -r requirements/requirements.txt

echo "Installing eval requirements (skipping detectron2)..."
pip install imageio>=2.34.1 pyiqa==0.1.10 scikit-learn>=1.4.2 scikit-image>=0.20.0 \
  lvis==0.5.3 boto3>=1.34.113 easydict>=1.9 fairscale>=0.4.13 \
  decord==0.6.0 pytorchvideo==0.1.5 lpips==0.1.4

echo "Installing VAE requirements..."
pip install -r requirements/requirements-vae.txt

echo "Installing Hugging Face tools..."
pip install huggingface-hub datasets

# ============================================================================
# Verification: Check that all critical versions are correct
# ============================================================================
echo ""
echo "========================================"
echo "Environment Verification"
echo "========================================"
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
print(f'CUDA available: {torch.cuda.is_available()} (N/A on CPU node)')
print()

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

echo ""
echo "========================================"
echo "Environment Setup Complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Conda environment location:"
echo "  ${SCRATCH_BASE}/conda-envs/opensora13"
echo ""
echo "To activate this environment in future jobs:"
echo "  conda activate ${SCRATCH_BASE}/conda-envs/opensora13"
echo ""
echo "Next steps:"
echo "  1. Download and preprocess UCF-101 dataset:"
echo "     cd env_setup/download_ucf101 && sbatch preprocess_ucf101.sbatch"
echo "  2. Run the naive experiment:"
echo "     cd naive_experiment/scripts && sbatch run_experiment.sbatch"
echo ""

