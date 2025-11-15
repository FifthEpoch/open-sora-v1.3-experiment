#!/bin/bash
#SBATCH --job-name=setup_opensora_env
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=slurm_setup_env_%j.out
#SBATCH --error=slurm_setup_env_%j.err

# ==============================================================================
# Comprehensive Open-Sora v1.3 Environment Setup
# ==============================================================================
# This script creates and configures the complete environment with proper
# dependency ordering to prevent version conflicts (especially NumPy 1.x vs 2.x)
#
# Safe to run multiple times - will detect existing environment and verify/fix
# ==============================================================================

set -euo pipefail

# Prevent user site-packages from interfering with conda environment
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "Open-Sora v1.3 Environment Setup"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "=============================================================================="
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/opensora13"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"

# Expected versions
EXPECTED_PYTHON="3.10"
EXPECTED_NUMPY_MAJOR="1"
EXPECTED_TORCH="2.2.2+cu121"
EXPECTED_XFORMERS="0.0.25.post1"
EXPECTED_BITSANDBYTES="0.43.3"

# ==============================================================================
# Setup Scratch Environment Variables
# ==============================================================================
echo "Setting up scratch environment variables..."
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# ==============================================================================
# Load Conda
# ==============================================================================
echo ""
echo "Loading conda module..."
module purge
module load anaconda3/2024.02
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Configure conda to use scratch directories
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --prepend pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# ==============================================================================
# Check/Create Environment
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 1: Environment Detection"
echo "=============================================================================="

if [ -d "${ENV_PATH}" ]; then
    echo "✓ Environment exists at: ${ENV_PATH}"
    echo "  Will verify and fix packages if needed..."
    ENV_EXISTS=true
else
    echo "Environment not found at: ${ENV_PATH}"
    echo "  Will create new environment..."
    ENV_EXISTS=false
fi

if [ "$ENV_EXISTS" = false ]; then
    echo ""
    echo "Creating new conda environment with Python 3.10..."
    conda create -y -p "${ENV_PATH}" python=3.10
    echo "✓ Environment created!"
fi

# Activate environment
echo ""
echo "Activating environment..."
conda activate "${ENV_PATH}"

# Verify activation
if [ "$CONDA_PREFIX" != "${ENV_PATH}" ]; then
    echo "❌ ERROR: Failed to activate environment!"
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "Expected: ${ENV_PATH}"
    exit 1
fi

# Explicitly ensure conda env's bin is first in PATH
export PATH="${ENV_PATH}/bin:${PATH}"

echo "✓ Environment activated"
echo "  Python: $(python --version)"
echo "  Location: ${CONDA_PREFIX}"
echo "  Python executable: $(which python)"
echo "  PYTHONNOUSERSITE: ${PYTHONNOUSERSITE}"

# Critical verification: ensure we're using the environment's Python
PYTHON_PATH=$(which python)
if [[ ! "$PYTHON_PATH" =~ ^${ENV_PATH} ]]; then
    echo "❌ ERROR: Python executable is not from the conda environment!"
    echo "  Expected: ${ENV_PATH}/bin/python"
    echo "  Got: ${PYTHON_PATH}"
    echo "  PATH: ${PATH}"
    exit 1
fi
echo "✓ Using environment's Python: ${PYTHON_PATH}"

# ==============================================================================
# Check Python Version
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 2: Python Version Check"
echo "=============================================================================="

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Current Python version: ${PYTHON_VERSION}"

if [[ ! "$PYTHON_VERSION" =~ ^3\.1[0-9] ]]; then
    echo "⚠️  Python 3.10+ required (found ${PYTHON_VERSION})"
    echo "Upgrading Python to 3.10..."
    conda install -y python=3.10
    echo "✓ Python upgraded"
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "New Python version: ${PYTHON_VERSION}"
fi

echo "✓ Python version OK: ${PYTHON_VERSION}"

# Upgrade pip
python -m pip install -U pip setuptools wheel

# ==============================================================================
# Install Core ML Stack (NumPy + PyTorch + xformers)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 3: Core ML Stack Installation"
echo "=============================================================================="
echo "Installing NumPy 1.x + PyTorch 2.2.2 + xformers 0.0.25.post1..."
echo "⚠️  CRITICAL: NumPy must be installed from PyPI first, then PyTorch"
echo ""

# Step 3.1: Install NumPy 1.x from PyPI (PyTorch index only has NumPy 2.x)
echo "Step 3.1: Installing NumPy 1.x from PyPI..."
pip install 'numpy<2,>=1.26'

# Step 3.2: Install PyTorch packages with --no-deps to prevent NumPy upgrade
echo ""
echo "Step 3.2: Installing PyTorch 2.2.2 + xformers with --no-deps..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
  --no-deps \
  torch==2.2.2 \
  torchvision==0.17.2 \
  xformers==0.0.25.post1

# Step 3.3: Install PyTorch dependencies (except NumPy which we already have)
echo ""
echo "Step 3.3: Installing PyTorch dependencies..."
# Install standard Python dependencies
pip install typing-extensions sympy networkx jinja2 fsspec filelock pillow

# Step 3.4: Install CUDA dependencies that PyTorch needs
echo ""
echo "Step 3.4: Installing CUDA dependencies for PyTorch..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
  nvidia-cublas-cu12==12.1.3.1 \
  nvidia-cuda-cupti-cu12==12.1.105 \
  nvidia-cuda-nvrtc-cu12==12.1.105 \
  nvidia-cuda-runtime-cu12==12.1.105 \
  nvidia-cudnn-cu12==8.9.2.26 \
  nvidia-cufft-cu12==11.0.2.54 \
  nvidia-curand-cu12==10.3.2.106 \
  nvidia-cusolver-cu12==11.4.5.107 \
  nvidia-cusparse-cu12==12.1.0.106 \
  nvidia-nccl-cu12==2.19.3 \
  nvidia-nvtx-cu12==12.1.105 \
  triton==2.2.0

echo ""
echo "Verifying core ML stack..."
python -c "
import numpy as np
import torch
import torchvision
import xformers

print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'xformers: {xformers.__version__}')

# Verify versions
assert np.__version__.startswith('1.'), f'NumPy must be 1.x, got {np.__version__}'
assert torch.__version__ == '2.2.2+cu121', f'PyTorch must be 2.2.2+cu121, got {torch.__version__}'
assert xformers.__version__ == '0.0.25.post1', f'xformers must be 0.0.25.post1, got {xformers.__version__}'

print('✓ Core ML stack versions verified!')
"

echo "✓ Core ML stack installed and verified"

# ==============================================================================
# Install bitsandbytes (with --no-deps to prevent upgrades)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 4: bitsandbytes Installation"
echo "=============================================================================="
echo "Installing bitsandbytes 0.43.3 with --no-deps..."
echo "⚠️  Using --no-deps prevents it from upgrading NumPy/PyTorch"
echo ""

pip install 'bitsandbytes==0.43.3' --no-deps --no-cache-dir

echo ""
echo "Verifying bitsandbytes..."
python -c "
import bitsandbytes
print(f'bitsandbytes: {bitsandbytes.__version__}')
assert bitsandbytes.__version__ == '0.43.3', f'bitsandbytes must be 0.43.3, got {bitsandbytes.__version__}'
print('✓ bitsandbytes version verified!')
"

echo "✓ bitsandbytes installed and verified"

# ==============================================================================
# Install PyAV via conda
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 5: PyAV Installation"
echo "=============================================================================="
echo "Installing PyAV via conda for better ffmpeg handling..."
echo ""

conda install -y av -c conda-forge

echo ""
echo "Verifying PyAV..."
python -c "
import av
print(f'PyAV: {av.__version__}')
print('✓ PyAV installed successfully!')
"

echo "✓ PyAV installed and verified"

# ==============================================================================
# Install Open-Sora Requirements
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 6: Open-Sora Requirements"
echo "=============================================================================="

cd "${PROJECT_ROOT}"

echo "Installing Open-Sora CUDA requirements..."
pip install -r requirements/requirements-cu121.txt

echo ""
echo "Installing base requirements..."
pip install -r requirements/requirements.txt

echo ""
echo "Installing VAE requirements..."
pip install -r requirements/requirements-vae.txt

echo "✓ Open-Sora requirements installed"

# ==============================================================================
# Install Evaluation Requirements
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 7: Evaluation Requirements"
echo "=============================================================================="
echo "Installing eval requirements with NumPy constraint..."
echo "⚠️  Explicitly constraining NumPy to prevent upgrade to 2.x"
echo ""

pip install 'numpy<2' \
  imageio>=2.34.1 pyiqa==0.1.10 \
  scikit-learn>=1.4.2 scikit-image>=0.20.0 \
  lvis==0.5.3 boto3>=1.34.113 easydict>=1.9 \
  fairscale>=0.4.13 decord==0.6.0 \
  pytorchvideo==0.1.5 lpips==0.1.4

echo "✓ Evaluation requirements installed"

# ==============================================================================
# Install Hugging Face Tools
# ==============================================================================
echo ""
echo "Installing Hugging Face tools..."
pip install huggingface-hub datasets

echo "✓ Hugging Face tools installed"

# ==============================================================================
# Cleanup Ghost Metadata
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 8: Cleanup Ghost Metadata"
echo "=============================================================================="
echo "Checking for ghost NumPy 2.x metadata directories..."
echo ""

GHOST_DIRS=$(find "${CONDA_PREFIX}/lib/python"*/site-packages -name "numpy-2.*.dist-info" 2>/dev/null || true)

if [ -n "$GHOST_DIRS" ]; then
    echo "⚠️  Found ghost NumPy 2.x metadata:"
    echo "$GHOST_DIRS"
    echo ""
    echo "Removing ghost metadata..."
    echo "$GHOST_DIRS" | while read -r dir; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            echo "  Removed: $dir"
        fi
    done
    echo "✓ Ghost metadata removed"
else
    echo "✓ No ghost metadata found"
fi

# ==============================================================================
# Final Verification
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 9: Final Verification"
echo "=============================================================================="
echo "Running comprehensive verification..."
echo ""

python -c "
import sys
import importlib.metadata
import numpy as np
import torch
import torchvision
import xformers
import bitsandbytes
import pandas as pd
import av
import cv2

print('=' * 70)
print('ENVIRONMENT VERIFICATION')
print('=' * 70)
print()
print(f'Python: {sys.version}')
print(f'  Location: {sys.executable}')
print()

# Check critical versions
print('Critical Package Versions:')
print(f'  NumPy: {np.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  xformers: {xformers.__version__}')
print(f'  bitsandbytes: {bitsandbytes.__version__}')
print(f'  Pandas: {pd.__version__}')
print(f'  PyAV: {av.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print()

# Check GPU
print('GPU Check:')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
print()

# Check importlib.metadata sees NumPy correctly
print('Metadata Check:')
try:
    metadata_version = importlib.metadata.version('numpy')
    print(f'  importlib.metadata sees NumPy: {metadata_version}')
    if metadata_version != np.__version__:
        print(f'  ⚠️  WARNING: Mismatch with numpy.__version__!')
    else:
        print(f'  ✓ Metadata matches numpy.__version__')
except Exception as e:
    print(f'  ❌ ERROR: {e}')
print()

# Verify assertions
print('Version Assertions:')
errors = []

if not np.__version__.startswith('1.'):
    errors.append(f'  ❌ NumPy must be 1.x, got {np.__version__}')
else:
    print(f'  ✓ NumPy 1.x: {np.__version__}')

if torch.__version__ != '2.2.2+cu121':
    errors.append(f'  ❌ PyTorch must be 2.2.2+cu121, got {torch.__version__}')
else:
    print(f'  ✓ PyTorch 2.2.2+cu121: {torch.__version__}')

if xformers.__version__ != '0.0.25.post1':
    errors.append(f'  ❌ xformers must be 0.0.25.post1, got {xformers.__version__}')
else:
    print(f'  ✓ xformers 0.0.25.post1: {xformers.__version__}')

if bitsandbytes.__version__ != '0.43.3':
    errors.append(f'  ❌ bitsandbytes must be 0.43.3, got {bitsandbytes.__version__}')
else:
    print(f'  ✓ bitsandbytes 0.43.3: {bitsandbytes.__version__}')

if not sys.version_info >= (3, 10):
    errors.append(f'  ❌ Python must be 3.10+, got {sys.version_info.major}.{sys.version_info.minor}')
else:
    print(f'  ✓ Python 3.10+: {sys.version_info.major}.{sys.version_info.minor}')

print()

if errors:
    print('ERRORS FOUND:')
    for error in errors:
        print(error)
    sys.exit(1)
else:
    print('✓ All version assertions passed!')

print()

# Test opensora import from local directory
print('Testing opensora import from local directory...')
sys.path.insert(0, '${PROJECT_ROOT}')
try:
    import opensora
    print(f'  ✓ opensora imports from: {opensora.__file__}')
except Exception as e:
    print(f'  ⚠️  Warning: opensora import failed: {e}')
    print(f'  This is OK if running from different directory')

print()
print('=' * 70)
print('✓ ENVIRONMENT SETUP COMPLETE AND VERIFIED!')
print('=' * 70)
"

VERIFICATION_EXIT=$?

if [ $VERIFICATION_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Verification failed! Please check errors above."
    exit 1
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Setup Complete!"
echo "=============================================================================="
echo "Environment location: ${ENV_PATH}"
echo "Project root: ${PROJECT_ROOT}"
echo ""
echo "To use this environment in future jobs:"
echo "  conda activate ${ENV_PATH}"
echo ""
echo "Next steps:"
echo "  1. Download and preprocess UCF-101:"
echo "     cd env_setup/download_ucf101"
echo "     sbatch preprocess_ucf101.sbatch"
echo ""
echo "  2. Run naive experiment:"
echo "     cd naive_experiment/scripts"
echo "     sbatch rtx_run_experiment.sbatch"
echo ""
echo "End time: $(date)"
echo "=============================================================================="

