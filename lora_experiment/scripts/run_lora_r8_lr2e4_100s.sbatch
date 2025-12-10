#!/bin/bash
#SBATCH --job-name=lora_lr2e4_100s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/scripts/slurm_log/lora_r8_lr2e4_100s_%j.out
#SBATCH --error=lora_experiment/scripts/slurm_log/lora_r8_lr2e4_100s_%j.err

# ============================================================================
# LoRA Experiment: Rank 8, LR 2e-4, 100 steps
# Purpose: Does higher LR help at 100 steps?
# Compare with: R8 LR1e-4 100s (current best)
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

NUM_VIDEOS=10
NUM_STEPS=100
LORA_RANK=8
LORA_ALPHA=16
LEARNING_RATE=2e-4
CONDITION_FRAMES=22

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/rank8_lr2e4_100steps"
CONFIG_FILE="${PROJECT_ROOT}/lora_experiment/configs/lora_finetune.py"
# Reference sampled videos CSV - use lora_20steps as the canonical source
REFERENCE_SAMPLED_CSV="${PROJECT_ROOT}/lora_experiment/results/lora_20steps/sampled_videos.csv"

echo "========================================"
echo "LoRA: R8 LR2e-4 100steps"
echo "Purpose: Test if higher LR helps at 100 steps"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/opensora13"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
cd "$PROJECT_ROOT"

if [ ! -f "${REFERENCE_SAMPLED_CSV}" ]; then
    echo "ERROR: Reference CSV not found at: ${REFERENCE_SAMPLED_CSV}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/generated"
SAMPLED_CSV="${OUTPUT_DIR}/sampled_videos.csv"
cp "${REFERENCE_SAMPLED_CSV}" "${SAMPLED_CSV}"
echo "Using reference CSV from lora_20steps"

mkdir -p "${OUTPUT_DIR}/generated"
cp "${REFERENCE_SAMPLED_CSV}" "${OUTPUT_DIR}/sampled_videos.csv"
SAMPLED_CSV="${OUTPUT_DIR}/sampled_videos.csv"

VIDEO_COUNT=0
while IFS=, read -r video_path num_frames cond_frames height width fps caption class_name || [ -n "$video_path" ]; do
    [ "$video_path" == "path" ] && continue
    [ $VIDEO_COUNT -ge $NUM_VIDEOS ] && break
    
    VIDEO_NAME=$(basename "$video_path" .mp4)
    VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/video_${VIDEO_COUNT}"
    mkdir -p "${VIDEO_OUTPUT_DIR}"
    FULL_VIDEO_PATH="${DATA_DIR}/${video_path}"
    [ ! -f "${FULL_VIDEO_PATH}" ] && continue
    
    echo "Processing video $((VIDEO_COUNT + 1))/${NUM_VIDEOS}: ${VIDEO_NAME}"
    
    TRAIN_START=$(date +%s)
    python lora_experiment/scripts/lora_finetune.py \
        --config "${CONFIG_FILE}" --video-path "${FULL_VIDEO_PATH}" \
        --caption "${caption}" --output-dir "${VIDEO_OUTPUT_DIR}" \
        --num-steps ${NUM_STEPS} --lr ${LEARNING_RATE} \
        --rank ${LORA_RANK} --alpha ${LORA_ALPHA} --seed 42 2>&1 | tee "${VIDEO_OUTPUT_DIR}/train.log"
    TRAIN_END=$(date +%s)
    TRAIN_TIME=$((TRAIN_END - TRAIN_START))
    
    LORA_WEIGHTS="${VIDEO_OUTPUT_DIR}/lora_weights.pt"
    [ ! -f "${LORA_WEIGHTS}" ] && continue
    
    INFER_START=$(date +%s)
    python lora_experiment/scripts/lora_inference.py \
        --config "${CONFIG_FILE}" --lora-weights "${LORA_WEIGHTS}" \
        --video-path "${FULL_VIDEO_PATH}" --caption "${caption}" \
        --output-path "${OUTPUT_DIR}/generated/lora_${VIDEO_COUNT}_${VIDEO_NAME}.mp4" \
        --condition-frames ${CONDITION_FRAMES} --rank ${LORA_RANK} --alpha ${LORA_ALPHA} 2>&1 | tee "${VIDEO_OUTPUT_DIR}/infer.log"
    INFER_END=$(date +%s)
    INFER_TIME=$((INFER_END - INFER_START))
    
    GENERATED_VIDEO="${OUTPUT_DIR}/generated/lora_${VIDEO_COUNT}_${VIDEO_NAME}.mp4"
    if [ -f "${GENERATED_VIDEO}" ]; then
        python -c "
import json, numpy as np, sys
sys.path.insert(0, '${PROJECT_ROOT}')
from naive_experiment.scripts.evaluate_continuations import extract_frames_from_video, load_generated_video, evaluate_pair
try:
    gt = extract_frames_from_video('${FULL_VIDEO_PATH}', start_frame=${CONDITION_FRAMES}, num_frames=27)
    gen = load_generated_video('${GENERATED_VIDEO}')
    if gen is not None and len(gen) > ${CONDITION_FRAMES}: gen = gen[${CONDITION_FRAMES}:]
    metrics = evaluate_pair(gt, gen) if gt is not None and gen is not None else {'error': 'frames'}
    metrics.update({'video_idx': ${VIDEO_COUNT}, 'video_name': '${VIDEO_NAME}', 'train_time_sec': ${TRAIN_TIME}, 'infer_time_sec': ${INFER_TIME}})
    with open('${VIDEO_OUTPUT_DIR}/metrics.json', 'w') as f: json.dump(metrics, f, indent=2, default=float)
    print(f'  PSNR: {metrics.get(\"psnr\", 0):.2f}, SSIM: {metrics.get(\"ssim\", 0):.4f}, LPIPS: {metrics.get(\"lpips\", 1):.4f}')
except Exception as e: print(f'Metrics failed: {e}')
"
    fi
    rm -f "${LORA_WEIGHTS}"
    VIDEO_COUNT=$((VIDEO_COUNT + 1))
done < "${SAMPLED_CSV}"

python -c "
import json, numpy as np
from pathlib import Path
files = list(Path('${OUTPUT_DIR}').glob('video_*/metrics.json'))
if files:
    metrics = [json.load(open(f)) for f in files]
    summary = {'num_videos': len(metrics), 'num_steps': ${NUM_STEPS}, 'rank': ${LORA_RANK}, 'lr': ${LEARNING_RATE},
               'avg_psnr': float(np.mean([m.get('psnr',0) for m in metrics if m.get('psnr')])),
               'avg_ssim': float(np.mean([m.get('ssim',0) for m in metrics if m.get('ssim')])),
               'avg_lpips': float(np.mean([m.get('lpips',1) for m in metrics if m.get('lpips')])),
               'avg_train_time_sec': float(np.mean([m.get('train_time_sec',0) for m in metrics])),
               'avg_infer_time_sec': float(np.mean([m.get('infer_time_sec',0) for m in metrics]))}
    with open('${OUTPUT_DIR}/metrics_summary.json', 'w') as f: json.dump(summary, f, indent=2)
    print(f'Results: PSNR={summary[\"avg_psnr\"]:.2f}, SSIM={summary[\"avg_ssim\"]:.4f}, LPIPS={summary[\"avg_lpips\"]:.4f}')
"
echo "Done! Results: ${OUTPUT_DIR}"

