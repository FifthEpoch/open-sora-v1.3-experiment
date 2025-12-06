#!/bin/bash
#SBATCH --job-name=exp_50steps_5e5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=naive_experiment/scripts/slurm_log/run_50steps_5e5_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/run_50steps_5e5_%j.err

# ============================================================================
# Naive Experiment: 50 steps @ 5e-5
# Good quality/speed tradeoff from HP sweep (PSNR=10.55, SSIM=0.3649, LPIPS=0.5899)
# Estimated time: ~13 hours for 100 videos
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/results_50steps_5e5"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"

echo "========================================"
echo "Experiment: 50 steps @ 5e-5"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output: ${OUTPUT_DIR}"
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

if [ ! -f "${METADATA_CSV}" ]; then
    echo "ERROR: Metadata CSV not found" >&2
    exit 1
fi

echo "✓ Metadata validated"
nvidia-smi --query-gpu=name,memory.free --format=csv

mkdir -p "${OUTPUT_DIR}"

python naive_experiment/scripts/run_experiment.py \
    --data-csv "${METADATA_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-videos 100 \
    --condition-frames 22 \
    --finetune-steps 50 \
    --finetune-lr 5e-5 \
    --skip-baseline

echo ""
echo "========================================"
echo "Experiment Complete"
echo "End Time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "========================================"

