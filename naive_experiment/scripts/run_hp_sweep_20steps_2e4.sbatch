#!/bin/bash
#SBATCH --job-name=hp_test_20steps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=naive_experiment/scripts/slurm_log/hp_test_20steps_2e4_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/hp_test_20steps_2e4_%j.err

# ============================================================================
# Quick test: 20 steps @ 2e-4
# Comparing to see if 20 steps at high LR beats 15 steps
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/hp_sweep_20steps_2e4"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"

echo "========================================"
echo "Testing: 20 steps @ 2e-4 LR"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "========================================"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/opensora13"
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"

# Validate
if [ ! -f "${METADATA_CSV}" ]; then
    echo "ERROR: Metadata CSV not found" >&2
    exit 1
fi

echo "✓ Metadata validated"
nvidia-smi --query-gpu=name,memory.free --format=csv

mkdir -p "${OUTPUT_DIR}"

python naive_experiment/scripts/run_hp_sweep.py \
    --data-csv "${METADATA_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-videos 3 \
    --condition-frames 22 \
    --configs "test_20steps_2e4"

echo ""
echo "========================================"
echo "Test Complete - Summarizing..."
echo "========================================"
python naive_experiment/scripts/summarize_hp_sweep.py --sweep-dir "${OUTPUT_DIR}"

echo ""
echo "Done: $(date)"

