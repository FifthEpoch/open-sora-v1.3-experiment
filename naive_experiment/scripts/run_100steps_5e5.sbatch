#!/bin/bash
#SBATCH --job-name=exp_100steps_5e5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=naive_experiment/scripts/slurm_log/run_100steps_5e5_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/run_100steps_5e5_%j.err

# ============================================================================
# Naive Experiment: 100 steps @ 5e-5
# Best quality config from HP sweep (PSNR=12.38, SSIM=0.4918, LPIPS=0.4432)
# Estimated time: ~27 hours for 100 videos
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/results_100steps_5e5"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"

echo "========================================"
echo "Experiment: 100 steps @ 5e-5"
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

# Copy existing baselines to avoid regenerating
EXISTING_BASELINES="${PROJECT_ROOT}/naive_experiment/scripts/results/baselines"
if [ -d "${EXISTING_BASELINES}" ]; then
    echo "Copying existing baselines from ${EXISTING_BASELINES}..."
    cp -r "${EXISTING_BASELINES}" "${OUTPUT_DIR}/"
    echo "✓ Baselines copied"
else
    echo "WARNING: No existing baselines found at ${EXISTING_BASELINES}"
    echo "Will generate baselines (this adds time)"
fi

# ============================================================================
# AUTO-RESUME: Read progress.json to find where to continue from
# ============================================================================
START_FROM_VIDEO=0
PROGRESS_FILE="${OUTPUT_DIR}/progress.json"
if [ -f "$PROGRESS_FILE" ]; then
    START_FROM_VIDEO=$(python3 -c "
import json
try:
    with open('$PROGRESS_FILE') as f:
        progress = json.load(f)
    completed = [p['video_idx'] for p in progress if p.get('status') == 'completed']
    if completed:
        print(max(completed) + 1)
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)
    echo "========================================"
    echo "AUTO-RESUME: Found progress.json"
    echo "  Last completed video: $((START_FROM_VIDEO - 1))"
    echo "  Resuming from video: ${START_FROM_VIDEO}"
    echo "========================================"
fi

# Skip if already done
if [ "$START_FROM_VIDEO" -ge 100 ]; then
    echo "All 100 videos already completed! Exiting."
    exit 0
fi

python naive_experiment/scripts/run_experiment.py \
    --data-csv "${METADATA_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-videos 100 \
    --condition-frames 22 \
    --finetune-steps 100 \
    --finetune-lr 5e-5 \
    --skip-baseline \
    --start-from-video "${START_FROM_VIDEO}"

echo ""
echo "========================================"
echo "Experiment Complete"
echo "End Time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "========================================"

