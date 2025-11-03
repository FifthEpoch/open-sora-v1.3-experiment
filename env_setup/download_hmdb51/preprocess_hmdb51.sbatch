#!/bin/bash
#SBATCH -J preprocess_hmdb51
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_preprocess_hmdb51.out

set -euo pipefail

echo "========================================"
echo "HMDB51 Preprocessing Job"
echo "========================================"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

# --- Environment Setup ---
# Set scratch directory to avoid hitting /home quotas
SCRATCH_BASE="/scratch/wc3013"

# Create necessary directories
mkdir -p "${SCRATCH_BASE}"/{conda-envs,conda-pkgs,py-cache/{pip,torch/extensions,models,triton,torch-inductor,colossalaJit,python},hf-cache/{datasets,hub,transformers},tmp,wandb,tensorboard,wheels}

# Set environment variables for scratch storage
export CONDA_ENVS_DIRS="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"
export XDG_CACHE_HOME="${SCRATCH_BASE}/py-cache"
export TORCH_HOME="${SCRATCH_BASE}/py-cache/models"
export TORCH_EXTENSIONS_DIR="${SCRATCH_BASE}/py-cache/torch/extensions"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/py-cache/torch-inductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/py-cache/triton"
export COLOSSALAI_EXTENSIONS_JIT_CACHE_PATH="${SCRATCH_BASE}/py-cache/colossalaJit"
export HF_HOME="${SCRATCH_BASE}/hf-cache"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/hf-cache/datasets"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf-cache/hub"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/hf-cache/transformers"
export PIP_CACHE_DIR="${SCRATCH_BASE}/py-cache/pip"
export PIP_NO_CACHE_DIR=1
export TMPDIR="${SCRATCH_BASE}/tmp"
export TMP="${SCRATCH_BASE}/tmp"
export TEMP="${SCRATCH_BASE}/tmp"
export PYTHONPYCACHEPREFIX="${SCRATCH_BASE}/py-cache/python"
export WANDB_CACHE_DIR="${SCRATCH_BASE}/wandb"
export TENSORBOARD_LOGDIR="${SCRATCH_BASE}/tensorboard"

echo "Environment configured to use: ${SCRATCH_BASE}"

# Load conda module
module purge
module load anaconda3/2024.02

# Source conda initialization script
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment in scratch location
echo "Activating conda environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

# Verify environment
python --version
python -c "import av; import numpy; import pandas; print('✓ Required packages available')"

# Set OMP threads to match requested CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Run Preprocessing ---
echo "========================================"
echo "Starting HMDB51 preprocessing..."
echo "========================================"

cd "${SCRIPT_DIR}"

# Run with --skip-cleanup to avoid interactive prompt
python preprocess_hmdb51.py --skip-cleanup

EXIT_CODE=$?

echo "========================================"
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE

