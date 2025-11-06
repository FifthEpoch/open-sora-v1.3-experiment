#!/bin/bash
#SBATCH -J preprocess_ucf101
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_preprocess_ucf101.out

echo "=========================================="
echo "UCF-101 Video Preprocessing Job"
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
module load anaconda3/2024.02

# Source conda for shell
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Configure conda to recognize custom environment location
conda config --append envs_dirs "${CONDA_ENVS_PATH}"
conda config --append pkgs_dirs "${CONDA_PKGS_DIRS}"

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

echo ""
echo "=========================================="
echo "Starting preprocessing..."
echo "=========================================="

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

# Run preprocessing (non-interactive with --skip-cleanup)
python preprocess_ucf101.py --skip-cleanup

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

