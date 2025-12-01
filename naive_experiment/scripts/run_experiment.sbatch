#!/bin/bash
#SBATCH -J naive_finetune_exp
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 47:59:59
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm_run_experiment.out
#SBATCH --error=slurm_run_experiment.err

# ===================================================================
# CHECKPOINT RECOVERY INSTRUCTIONS
# ===================================================================
# If your job is interrupted (e.g., due to low GPU utilization), you can resume:
#
# 1. Check results/progress.json to find the last completed video:
#    cat results/progress.json | grep video_idx | tail -1
#
# 2. Re-submit with --skip-baseline and --start-from-video:
#    sbatch naive_experiment/scripts/run_experiment.sbatch \
#      --skip-baseline \
#      --start-from-video <last_completed_idx + 1>
#
# Example: If video 11 was the last completed, resume from video 12:
#    # Edit this script to change the python command to:
#    python "${SCRIPT_DIR}/run_experiment.py" \
#        --data-csv "${PROJECT_ROOT}/env_setup/download_ucf101/ucf101_metadata.csv" \
#        --output-dir results \
#        --num-videos 100 \
#        --skip-baseline \
#        --start-from-video 12
# ===================================================================

set -euo pipefail

# Prevent user site-packages from interfering with conda environment
export PYTHONNOUSERSITE=1

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

# Define project locations (needed early for PYTHONPATH)
PROJECT_ROOT="/scratch/wc3013/open-sora-v1.3-experiment"
SCRIPT_DIR="${PROJECT_ROOT}/naive_experiment/scripts"

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
echo "Purging modules..."
module purge
echo "Loading anaconda3/2025.06..."
module load anaconda3/2025.06
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load anaconda3/2025.06 module" >&2
    exit 1
fi
echo "✓ Module loaded"

# Source conda initialization script
echo "Sourcing conda initialization..."
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to source conda.sh" >&2
    exit 1
fi
echo "✓ Conda initialized"

# Verify conda command is available
echo "Verifying conda command is available..."
CONDA_CMD=$(command -v conda 2>/dev/null || echo "")
if [ -z "$CONDA_CMD" ]; then
    echo "ERROR: conda command not found in PATH" >&2
    echo "PATH: $PATH" >&2
    exit 1
fi
echo "✓ Conda command found: $CONDA_CMD"

# Check if we're already in the target environment
TARGET_ENV="${SCRATCH_BASE}/conda-envs/opensora13"
if [ "${CONDA_PREFIX:-}" = "${TARGET_ENV}" ]; then
    echo "✓ Already in target environment: ${TARGET_ENV}"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Deactivating existing environment: ${CONDA_DEFAULT_ENV}"
    # Use set +e to allow deactivate to fail without exiting script
    set +e
    conda deactivate 2>/dev/null
    DEACTIVATE_EXIT=$?
    set -e
    if [ $DEACTIVATE_EXIT -eq 0 ]; then
        echo "✓ Deactivated environment"
    else
        echo "⚠️  Deactivate failed (exit code: $DEACTIVATE_EXIT), continuing anyway..."
    fi
else
    echo "✓ No active environment to deactivate"
fi

# Configure conda to recognize custom environment location (use --prepend for priority)
echo "Configuring conda to use scratch directories..."
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --prepend pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true
echo "✓ Conda configured"

echo ""
echo "Listing conda environments..."
conda env list
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to list conda environments" >&2
    exit 1
fi
echo "✓ Environment list complete"

# Activate conda environment in scratch location
echo ""
echo "Activating conda environment: ${SCRATCH_BASE}/conda-envs/opensora13"
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"
if [ $? -ne 0 ] || [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: Failed to activate conda environment" >&2
    echo "CONDA_PREFIX: $CONDA_PREFIX" >&2
    exit 1
fi
echo "✓ Environment activated: $CONDA_PREFIX"

# Explicitly ensure conda env's bin is first in PATH
export PATH="${CONDA_PREFIX}/bin:${PATH}"
echo "PATH updated to prioritize conda environment"

# Verify we're using the environment's Python
PYTHON_PATH=$(which python)
if [[ ! "$PYTHON_PATH" =~ ^${CONDA_PREFIX} ]]; then
    echo "ERROR: Python executable is not from the conda environment!" >&2
    echo "  Expected: ${CONDA_PREFIX}/bin/python" >&2
    echo "  Got: ${PYTHON_PATH}" >&2
    exit 1
fi
echo "✓ Using environment's Python: ${PYTHON_PATH}"

# Fix libstdc++ compatibility for PyAV (av package)
# PyAV from conda-forge requires GLIBCXX_3.4.30, ensure conda's libstdc++ takes priority
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# Ensure Python can import the repo modules (opensora, etc.)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

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

# Check bitsandbytes version (should be 0.43.x for PyTorch 2.2.2)
echo "========================================"
echo "Checking bitsandbytes version..."
python -c "import bitsandbytes; print(f'bitsandbytes version: {bitsandbytes.__version__}')" || echo "WARNING: bitsandbytes import failed - run env_setup/fix_bitsandbytes.sh first!"
echo "Note: Compatible version for PyTorch 2.2.2 is bitsandbytes 0.43.x"
echo "========================================"

# --- Run Experiment ---
echo "========================================"
echo "Starting experiment..."
echo "========================================"

# Navigate to script directory
cd "${SCRIPT_DIR}"
echo "Working directory: $(pwd)"

# Run experiment with absolute path to script
python "${SCRIPT_DIR}/run_experiment.py" \
    --data-csv "${PROJECT_ROOT}/env_setup/download_ucf101/ucf101_metadata.csv" \
    --output-dir results \
    --num-videos 100

EXIT_CODE=$?

echo "========================================"
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE

