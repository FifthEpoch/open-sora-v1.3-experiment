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

# Set scratch environment variables
SCRATCH_BASE="/scratch/wc3013/open-sora-v1.3-experiment"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export TORCH_HOME="${SCRATCH_BASE}/torch-cache"
export HF_HOME="${SCRATCH_BASE}/huggingface-cache"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface-cache/transformers"
export TMPDIR="${SCRATCH_BASE}/tmp"
export TEMP="${SCRATCH_BASE}/tmp"
export TMP="${SCRATCH_BASE}/tmp"

# Create directories if they don't exist
mkdir -p "${CONDA_ENVS_PATH}"
mkdir -p "${CONDA_PKGS_DIRS}"
mkdir -p "${PIP_CACHE_DIR}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${HF_HOME}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${TMPDIR}"

echo "Environment configured to use: ${SCRATCH_BASE}"

# Load anaconda module
module load anaconda3/2024.02

# Source conda for shell
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Activate conda environment
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

# Navigate to script directory
cd "${SCRATCH_BASE}/env_setup/download_ucf101"

# Run preprocessing (non-interactive with --skip-cleanup)
python preprocess_ucf101.py --skip-cleanup

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

