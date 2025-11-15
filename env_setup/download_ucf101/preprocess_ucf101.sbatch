#!/bin/bash
#SBATCH -J download_prep_ucf101
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_download_prep_ucf101.out

set -euo pipefail

# Prevent user site-packages from interfering with conda environment
export PYTHONNOUSERSITE=1

echo "=========================================="
echo "UCF-101 Download & Preprocessing Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Set scratch environment variables (matching 00_set_scratch_env.sh structure)
SCRATCH_BASE="/scratch/wc3013"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/py-cache/pip"
export TORCH_HOME="${SCRATCH_BASE}/py-cache/torch"
export TORCH_EXTENSIONS_DIR="${SCRATCH_BASE}/py-cache/torch/extensions"
export PYTORCH_KERNEL_CACHE_PATH="${SCRATCH_BASE}/py-cache/models"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/py-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/py-cache/torch-inductor"
export COLOSSALAI_CACHE_DIR="${SCRATCH_BASE}/py-cache/colossalai-jit"
export HF_HOME="${SCRATCH_BASE}/hf-cache"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf-cache/hub"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/hf-cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/hf-cache/transformers"
export TMPDIR="${SCRATCH_BASE}/tmp"
export TEMP="${SCRATCH_BASE}/tmp"
export TMP="${SCRATCH_BASE}/tmp"

# Create all scratch directories
mkdir -p "${CONDA_ENVS_PATH}"
mkdir -p "${CONDA_PKGS_DIRS}"
mkdir -p "${PIP_CACHE_DIR}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
mkdir -p "${COLOSSALAI_CACHE_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${TMPDIR}"

echo "Environment configured to use scratch base: ${SCRATCH_BASE}"

# Load anaconda module
module purge
module load anaconda3/2025.06

# Source conda for shell
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Deactivate any existing environment for clean activation
conda deactivate 2>/dev/null || true

# Configure conda to recognize custom environment location (use --prepend for priority)
conda config --prepend envs_dirs "${CONDA_ENVS_PATH}" 2>/dev/null || true
conda config --prepend pkgs_dirs "${CONDA_PKGS_DIRS}" 2>/dev/null || true

echo ""
echo "Conda configuration:"
conda config --show envs_dirs
conda config --show pkgs_dirs

echo ""
echo "Available conda environments:"
conda env list

# Activate conda environment using full path
echo ""
echo "Activating conda environment: ${CONDA_ENVS_PATH}/opensora13"
conda activate "${CONDA_ENVS_PATH}/opensora13"

echo "Python version:"
python --version

echo ""
echo "Checking required packages..."
python -c "import av; import numpy; import pandas; import cv2; print('✓ Required packages available')"

# Diagnostic: Show which python and packages
echo ""
echo "Diagnostic information:"
echo "Which python: $(which python)"
echo "Python executable: $(python -c 'import sys; print(sys.executable)')"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo ""

# Navigate to script directory
cd "${SCRATCH_BASE}/open-sora-v1.3-experiment/env_setup/download_ucf101"
echo "Working directory: $(pwd)"
echo ""

# ============================================================================
# Step 1: Download UCF-101 from Hugging Face (if not already downloaded)
# ============================================================================
echo "=========================================="
echo "Step 1: Downloading UCF-101 Dataset"
echo "=========================================="
echo ""

if [ -d "ucf101_org" ] && [ "$(ls -A ucf101_org/*.avi 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✓ UCF-101 videos already exist in ucf101_org/"
    echo "  Skipping download step."
    VIDEO_COUNT=$(find ucf101_org -name "*.avi" | wc -l)
    echo "  Found ${VIDEO_COUNT} videos"
else
    echo "Downloading UCF-101 from Hugging Face..."
    echo "Note: This will download ~7GB and may take 15-30 minutes"
    echo ""
    
    # Run download script (it will prompt for cleanup, so we pipe 'yes' to it)
    echo "yes" | python download_ucf101_hf.py
    
    if [ $? -ne 0 ]; then
        echo "✗ ERROR: Download failed!"
        echo "Please check the error messages above."
        exit 1
    fi
    
    echo ""
    echo "✓ Download complete!"
fi

echo ""

# ============================================================================
# Step 2: Preprocess videos
# ============================================================================
echo "=========================================="
echo "Step 2: Preprocessing Videos"
echo "=========================================="
echo ""

# Run preprocessing (non-interactive with --skip-cleanup)
python preprocess_ucf101.py --skip-cleanup

if [ $? -ne 0 ]; then
    echo "✗ ERROR: Preprocessing failed!"
    exit 1
fi

echo ""
echo "✓ Preprocessing complete!"

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

