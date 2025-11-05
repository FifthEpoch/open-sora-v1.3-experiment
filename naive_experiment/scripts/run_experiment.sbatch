#!/bin/bash
#SBATCH -J naive_finetune_exp
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_run_experiment.out
#SBATCH --error=slurm_run_experiment.err

set -euo pipefail

echo "========================================"
echo "Naive Fine-tuning Experiment"
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

# Activate conda environment in scratch location
echo "Activating conda environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

# Verify GPU availability
echo "========================================"
echo "GPU Information"
echo "========================================"
nvidia-smi

# Verify environment
echo "========================================"
echo "Environment Verification"
echo "========================================"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Set OMP threads to match requested CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Run Experiment ---
echo "========================================"
echo "Starting experiment..."
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to naive_experiment directory
cd "${SCRIPT_DIR}"

# Run experiment
python run_experiment.py \
    --data-csv /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_metadata.csv \
    --output-dir results \
    --num-videos 10

EXIT_CODE=$?

echo "========================================"
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE

