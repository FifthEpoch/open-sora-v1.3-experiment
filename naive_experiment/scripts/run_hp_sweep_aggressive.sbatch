#!/bin/bash
#SBATCH --job-name=hp_sweep_aggressive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=naive_experiment/scripts/slurm_log/hp_sweep_aggressive_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/hp_sweep_aggressive_%j.err

# ============================================================================
# AGGRESSIVE Hyperparameter Sweep
# Building on Option 2's success (20 steps @ 5e-5), tests:
#   - 50 steps @ 5e-5  (more steps + winning LR)
#   - 100 steps @ 5e-5 (even more steps)
#   - 20 steps @ 1e-4  (2x the winning LR)
#   - 50 steps @ 1e-4  (high LR + more steps)
#   - 20 steps @ 2e-4  (push LR further)
#
# Usage:
#   sbatch naive_experiment/scripts/run_hp_sweep_aggressive.sbatch
# ============================================================================

set -euo pipefail

export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
HP_CONFIG="aggressive"     # Test aggressive configs
NUM_VIDEOS=3               # Videos per configuration
CONDITION_FRAMES=22

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/hp_sweep_aggressive"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"

# ============================================================================
# Environment Setup
# ============================================================================
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "========================================"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/opensora13"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda environment: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"

# ============================================================================
# Validation
# ============================================================================
if [ ! -f "${METADATA_CSV}" ]; then
    echo "ERROR: Metadata CSV not found" >&2
    exit 1
fi

FIRST_VIDEO_HEIGHT=$(head -2 "${METADATA_CSV}" | tail -1 | cut -d',' -f4)
if [ "${FIRST_VIDEO_HEIGHT}" != "360" ]; then
    echo "ERROR: Data is not at 360p!" >&2
    exit 1
fi

echo "✓ Metadata validated: 360p data"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# Run Aggressive Sweep
# ============================================================================
echo "========================================"
echo "Starting AGGRESSIVE Hyperparameter Sweep"
echo "  Testing higher LRs and more steps"
echo "  Config set: ${HP_CONFIG}"
echo "  Videos per config: ${NUM_VIDEOS}"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

python naive_experiment/scripts/run_hp_sweep.py \
    --data-csv "${METADATA_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-videos "${NUM_VIDEOS}" \
    --condition-frames "${CONDITION_FRAMES}" \
    --configs "${HP_CONFIG}"

echo ""
echo "========================================"
echo "Aggressive Sweep Complete"
echo "End Time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "========================================"

