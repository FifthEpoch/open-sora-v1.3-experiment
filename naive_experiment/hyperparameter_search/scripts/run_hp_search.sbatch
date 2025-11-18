#!/bin/bash
#SBATCH --job-name=hp_search
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:h200:1
#SBATCH --output=slurm_hp_search_%j.out
#SBATCH --error=slurm_hp_search_%j.err

##############################################################################
# Hyperparameter Search SLURM Submission Script
#
# Usage:
#   sbatch run_hp_search.sbatch <search_type>
#
# Arguments:
#   search_type: 'lr', 'condition_frames', or 'steps'
#
# Example:
#   sbatch run_hp_search.sbatch lr
#   sbatch run_hp_search.sbatch condition_frames
#   sbatch run_hp_search.sbatch steps
#
# Recovery:
#   The script automatically resumes from progress.json if interrupted.
#   Just resubmit with the same command.
#
##############################################################################

set -eo pipefail
export PYTHONNOUSERSITE=1

# Define paths
PROJECT_ROOT="/scratch/wc3013/open-sora-v1.3-experiment"
SCRIPT_DIR="${PROJECT_ROOT}/naive_experiment/hyperparameter_search/scripts"
RESULTS_DIR="${PROJECT_ROOT}/naive_experiment/hyperparameter_search/results"

# Get search type from command line argument or SLURM array task ID
if [ $# -ge 1 ]; then
    SEARCH_TYPE=$1
else
    echo "ERROR: Search type not specified"
    echo "Usage: sbatch run_hp_search.sbatch <search_type>"
    echo "       where <search_type> is 'lr', 'condition_frames', or 'steps'"
    exit 1
fi

# Validate search type
if [[ ! "${SEARCH_TYPE}" =~ ^(lr|condition_frames|steps)$ ]]; then
    echo "ERROR: Invalid search type '${SEARCH_TYPE}'"
    echo "Must be one of: lr, condition_frames, steps"
    exit 1
fi

# Print job info
echo "========================================"
echo "Hyperparameter Search: ${SEARCH_TYPE}"
echo "========================================"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "========================================"

# Configure environment to use /scratch
export SCRATCH_BASE="/scratch/wc3013"
export TMPDIR="${SCRATCH_BASE}/tmp"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
export PIP_CACHE_DIR="${SCRATCH_BASE}/.cache/pip"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${PIP_CACHE_DIR}"

echo "Environment configured to use: ${SCRATCH_BASE}"

# Load modules
echo "Purging modules..."
module purge

echo "Loading anaconda3/2025.06..."
if ! module load anaconda3/2025.06; then
    echo "ERROR: Failed to load anaconda3/2025.06"
    exit 1
fi
echo "✓ Module loaded"

# Initialize conda
echo "Sourcing conda initialization..."
if ! source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh; then
    echo "ERROR: Failed to source conda.sh"
    exit 1
fi
echo "✓ Conda initialized"

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda command not found after initialization"
    exit 1
fi
echo "✓ Conda command available"

# Deactivate any existing environment (with error handling)
if [ ! -z "${CONDA_DEFAULT_ENV}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    echo "Deactivating existing conda environment: ${CONDA_DEFAULT_ENV}"
    set +e
    conda deactivate
    set -e
fi

# Configure conda to use /scratch for environments and packages
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --prepend pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# List available environments
echo "Available conda environments:"
conda env list

# Activate environment
echo "Activating conda environment: opensora13..."
if ! conda activate opensora13; then
    echo "ERROR: Failed to activate opensora13 environment"
    echo "Please ensure the environment exists at: ${SCRATCH_BASE}/conda-envs/opensora13"
    exit 1
fi
echo "✓ Environment activated: ${CONDA_DEFAULT_ENV}"

# Verify correct Python
export PATH="${CONDA_PREFIX}/bin:${PATH}"
PYTHON_PATH=$(which python)
echo "Python executable: ${PYTHON_PATH}"

if [[ ! "${PYTHON_PATH}" =~ opensora13 ]]; then
    echo "ERROR: Wrong Python executable!"
    echo "Expected: opensora13 environment"
    echo "Got: ${PYTHON_PATH}"
    exit 1
fi
echo "✓ Correct Python executable"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Prioritize conda's libstdc++ to avoid GLIBCXX version errors
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Install PyYAML if not present (required for config loading)
if ! python -c "import yaml" 2>/dev/null; then
    echo "Installing PyYAML..."
    pip install PyYAML --no-cache-dir
fi

# Verify packages
echo "Verifying key packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import yaml; print(f'PyYAML: OK')"

# Create results directory
mkdir -p "${RESULTS_DIR}/logs"

# Run hyperparameter search
echo ""
echo "Starting hyperparameter search..."
echo "Search type: ${SEARCH_TYPE}"
echo ""

cd "${SCRIPT_DIR}"

python run_hp_search.py \
    --search-type "${SEARCH_TYPE}" \
    --data-csv "/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_metadata.csv" \
    --results-dir "${RESULTS_DIR}"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job finished"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}

