#!/bin/bash
#SBATCH --job-name=hp_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=naive_experiment/scripts/slurm_log/hp_sweep_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/hp_sweep_%j.err

# ============================================================================
# Hyperparameter Sweep for Fine-tuning
# Tests different combinations of steps and learning rates
#
# Usage:
#   sbatch naive_experiment/scripts/run_hp_sweep.sbatch
#
# Optional: Modify HP_CONFIG to test different configuration sets:
#   - "default": Tests baseline, option1, option2 (3 configs)
#   - "extended": Tests 6 configurations including aggressive/gentle
#   - "minimal": Tests only option1 and option2 (2 configs)
# ============================================================================

set -euo pipefail

# Prevent user site-packages from interfering
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration - Modify these as needed
# ============================================================================
HP_CONFIG="default"        # Options: default, extended, minimal
NUM_VIDEOS=3               # Videos per configuration (keep low for quick testing)
CONDITION_FRAMES=22        # Conditioning frames

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/hp_sweep"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"

# ============================================================================
# Environment Setup
# ============================================================================
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "========================================"

# Load Anaconda
module purge
module load anaconda3/2025.06

# Initialize conda
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Activate environment
CONDA_ENV="${SCRATCH_BASE}/conda-envs/opensora13"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda environment: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi

# Set library paths
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Set cache directories
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME"

# Change to project directory
cd "$PROJECT_ROOT"

# ============================================================================
# Validation
# ============================================================================
if [ ! -f "${METADATA_CSV}" ]; then
    echo "ERROR: Metadata CSV not found at ${METADATA_CSV}" >&2
    echo "Please run preprocessing first: sbatch naive_experiment/scripts/preprocess_360p.sbatch" >&2
    exit 1
fi

# Verify dimensions are 360p
FIRST_VIDEO_HEIGHT=$(head -2 "${METADATA_CSV}" | tail -1 | cut -d',' -f4)
if [ "${FIRST_VIDEO_HEIGHT}" != "360" ]; then
    echo "ERROR: Data is not at 360p! Height=${FIRST_VIDEO_HEIGHT}" >&2
    echo "Please reprocess: sbatch naive_experiment/scripts/preprocess_360p.sbatch" >&2
    exit 1
fi

echo "✓ Metadata validated: 360p data"

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# Run Hyperparameter Sweep
# ============================================================================
echo "========================================"
echo "Starting Hyperparameter Sweep"
echo "  Config set: ${HP_CONFIG}"
echo "  Videos per config: ${NUM_VIDEOS}"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the sweep
python naive_experiment/scripts/run_hp_sweep.py \
    --data-csv "${METADATA_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-videos "${NUM_VIDEOS}" \
    --condition-frames "${CONDITION_FRAMES}" \
    --configs "${HP_CONFIG}"

echo ""
echo "========================================"
echo "Hyperparameter Sweep Complete"
echo "End Time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "========================================"

