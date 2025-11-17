#!/bin/bash
#SBATCH --job-name=eval_continuations
#SBATCH --output=slurm_evaluation.out
#SBATCH --error=slurm_evaluation.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

# Enable strict error handling
set -euo pipefail
export PYTHONNOUSERSITE=1

# Define project root
PROJECT_ROOT="/scratch/wc3013/open-sora-v1.3-experiment"
SCRIPT_DIR="${PROJECT_ROOT}/naive_experiment/scripts"

echo "========================================"
echo "Video Continuation Evaluation"
echo "========================================"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "========================================"

# Source environment configuration
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# Purge modules and load anaconda
echo "Purging modules..."
module purge
echo "Loading anaconda3/2025.06..."
module load anaconda3/2025.06

# Initialize conda
echo "Sourcing conda initialization..."
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Deactivate any existing conda environment
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Deactivating existing conda environment: ${CONDA_DEFAULT_ENV}"
    set +e
    conda deactivate
    set -e
fi

# Activate opensora13 environment
echo "Activating opensora13 environment..."
set +u  # Temporarily disable unbound variable check for conda activation
conda activate opensora13
set -u  # Re-enable unbound variable check

# Verify environment
echo "✓ Conda environment activated: $(conda info --envs | grep '*')"
echo "✓ Python: $(which python)"
echo "✓ CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "✓ GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set LD_LIBRARY_PATH for PyAV
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Change to script directory
cd "${SCRIPT_DIR}"

# Run the evaluation
echo ""
echo "========================================"
echo "Running Evaluation"
echo "========================================"

python evaluate_continuations.py \
    --original-videos "${PROJECT_ROOT}/env_setup/download_ucf101" \
    --baseline-outputs "${SCRIPT_DIR}/results/baselines" \
    --finetuned-outputs "${SCRIPT_DIR}/results/finetuned" \
    --manifest "${SCRIPT_DIR}/results/experiment_manifest.csv" \
    --condition-frames 22 \
    --output-json "${SCRIPT_DIR}/results/metrics.json"

echo ""
echo "========================================"
echo "Evaluation Complete"
echo "========================================"
echo "End time: $(date)"
echo "Results saved to: ${SCRIPT_DIR}/results/metrics.json"

