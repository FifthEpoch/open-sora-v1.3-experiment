#!/bin/bash
#SBATCH --job-name=lora_100steps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/scripts/slurm_log/lora_100steps_%j.out
#SBATCH --error=lora_experiment/scripts/slurm_log/lora_100steps_%j.err

# ============================================================================
# LoRA Experiment: 100 Steps
# Uses same sampled videos as 20-step experiment for comparison
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS=10
NUM_STEPS=100                  # 100 steps (vs 20 in baseline)
LORA_RANK=8
LORA_ALPHA=16
LEARNING_RATE=1e-4
CONDITION_FRAMES=22

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_100steps"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"
CONFIG_FILE="${PROJECT_ROOT}/lora_experiment/configs/lora_finetune.py"

# Reuse sampled videos from 20-step experiment for consistent comparison
REFERENCE_SAMPLED_CSV="${PROJECT_ROOT}/lora_experiment/results/lora_test/sampled_videos.csv"

echo "========================================"
echo "LoRA Experiment: 100 Steps"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "========================================"
echo "Configuration:"
echo "  Videos: ${NUM_VIDEOS}"
echo "  Steps: ${NUM_STEPS}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  LoRA Alpha: ${LORA_ALPHA}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================"

# ============================================================================
# Environment Setup
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/opensora13"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "✓ Activated conda environment: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"

# ============================================================================
# Validation
# ============================================================================
if [ ! -f "${REFERENCE_SAMPLED_CSV}" ]; then
    echo "ERROR: Reference sampled CSV not found at ${REFERENCE_SAMPLED_CSV}" >&2
    echo "Please run the 20-step experiment first to generate sampled_videos.csv" >&2
    exit 1
fi

echo "✓ Using same videos as 20-step experiment"
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Create Output Directory and Copy Sampled CSV
# ============================================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/generated"

SAMPLED_CSV="${OUTPUT_DIR}/sampled_videos.csv"
cp "${REFERENCE_SAMPLED_CSV}" "${SAMPLED_CSV}"
echo "✓ Copied sampled videos list from 20-step experiment"

# ============================================================================
# Run LoRA Experiment
# ============================================================================
echo "========================================"
echo "Starting LoRA Experiment (100 steps)"
echo "========================================"

VIDEO_COUNT=0
TOTAL_TRAIN_TIME=0
TOTAL_INFER_TIME=0

while IFS=, read -r video_path num_frames cond_frames height width fps caption class_name || [ -n "$video_path" ]; do
    if [ "$video_path" == "path" ]; then
        continue
    fi
    
    if [ $VIDEO_COUNT -ge $NUM_VIDEOS ]; then
        break
    fi
    
    VIDEO_NAME=$(basename "$video_path" .mp4)
    
    echo ""
    echo "========================================"
    echo "Processing video $((VIDEO_COUNT + 1))/${NUM_VIDEOS}: ${VIDEO_NAME}"
    echo "========================================"
    
    VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/video_${VIDEO_COUNT}"
    mkdir -p "${VIDEO_OUTPUT_DIR}"
    
    FULL_VIDEO_PATH="${DATA_DIR}/${video_path}"
    
    if [ ! -f "${FULL_VIDEO_PATH}" ]; then
        echo "WARNING: Video not found: ${FULL_VIDEO_PATH}, skipping..."
        continue
    fi
    
    # Step 1: LoRA Training
    echo "Step 1: LoRA Training (100 steps)..."
    TRAIN_START=$(date +%s)
    
    python lora_experiment/scripts/lora_finetune.py \
        --config "${CONFIG_FILE}" \
        --video-path "${FULL_VIDEO_PATH}" \
        --caption "${caption}" \
        --output-dir "${VIDEO_OUTPUT_DIR}" \
        --num-steps ${NUM_STEPS} \
        --lr ${LEARNING_RATE} \
        --rank ${LORA_RANK} \
        --seed 42 \
        2>&1 | tee "${VIDEO_OUTPUT_DIR}/train.log"
    
    TRAIN_END=$(date +%s)
    TRAIN_TIME=$((TRAIN_END - TRAIN_START))
    TOTAL_TRAIN_TIME=$((TOTAL_TRAIN_TIME + TRAIN_TIME))
    echo "Training time: ${TRAIN_TIME} seconds"
    
    LORA_WEIGHTS="${VIDEO_OUTPUT_DIR}/lora_weights.pt"
    if [ ! -f "${LORA_WEIGHTS}" ]; then
        echo "ERROR: LoRA weights not created for ${VIDEO_NAME}"
        continue
    fi
    
    # Step 2: LoRA Inference
    echo "Step 2: LoRA Inference..."
    INFER_START=$(date +%s)
    
    python lora_experiment/scripts/lora_inference.py \
        --config "${CONFIG_FILE}" \
        --lora-weights "${LORA_WEIGHTS}" \
        --video-path "${FULL_VIDEO_PATH}" \
        --caption "${caption}" \
        --output-path "${OUTPUT_DIR}/generated/lora_${VIDEO_COUNT}_${VIDEO_NAME}.mp4" \
        --condition-frames ${CONDITION_FRAMES} \
        2>&1 | tee "${VIDEO_OUTPUT_DIR}/infer.log"
    
    INFER_END=$(date +%s)
    INFER_TIME=$((INFER_END - INFER_START))
    TOTAL_INFER_TIME=$((TOTAL_INFER_TIME + INFER_TIME))
    echo "Inference time: ${INFER_TIME} seconds"
    
    # Step 3: Compute evaluation metrics
    GENERATED_VIDEO="${OUTPUT_DIR}/generated/lora_${VIDEO_COUNT}_${VIDEO_NAME}.mp4"
    if [ -f "${GENERATED_VIDEO}" ]; then
        echo "Step 3: Computing metrics..."
        python -c "
import torch
import json
import os
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

from naive_experiment.scripts.evaluate_continuations import extract_frames_from_video, load_generated_video, evaluate_pair

original = '${FULL_VIDEO_PATH}'
generated = '${GENERATED_VIDEO}'
condition_frames = ${CONDITION_FRAMES}

try:
    gt_frames = extract_frames_from_video(original, start_frame=condition_frames, num_frames=27)
    gen_frames = load_generated_video(generated)
    if gen_frames is not None and len(gen_frames) > condition_frames:
        gen_frames = gen_frames[condition_frames:]
    
    if gt_frames is not None and gen_frames is not None:
        metrics = evaluate_pair(gt_frames, gen_frames)
    else:
        metrics = {'error': 'Could not load frames'}
    
    metrics_file = '${VIDEO_OUTPUT_DIR}/metrics.json'
    metrics['video_idx'] = ${VIDEO_COUNT}
    metrics['video_name'] = '${VIDEO_NAME}'
    metrics['train_time_sec'] = ${TRAIN_TIME}
    metrics['infer_time_sec'] = ${INFER_TIME}
    metrics['num_steps'] = ${NUM_STEPS}
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    
    if 'psnr' in metrics:
        print(f'  PSNR: {metrics[\"psnr\"]:.2f}')
        print(f'  SSIM: {metrics[\"ssim\"]:.4f}')
        print(f'  LPIPS: {metrics[\"lpips\"]:.4f}')
except Exception as e:
    import traceback
    print(f'  Metrics computation failed: {e}')
    traceback.print_exc()
" 2>&1
    fi
    
    # Step 4: Clean up LoRA weights
    echo "Step 4: Cleaning up LoRA weights..."
    rm -f "${LORA_WEIGHTS}"
    
    VIDEO_COUNT=$((VIDEO_COUNT + 1))
    echo "Video ${VIDEO_COUNT} completed. Train: ${TRAIN_TIME}s, Infer: ${INFER_TIME}s"
    
done < "${SAMPLED_CSV}"

# ============================================================================
# Aggregate metrics
# ============================================================================
echo ""
echo "Aggregating metrics..."
python -c "
import json
from pathlib import Path
import numpy as np

output_dir = '${OUTPUT_DIR}'
metrics_files = list(Path(output_dir).glob('video_*/metrics.json'))

if not metrics_files:
    print('No metrics files found')
else:
    all_metrics = []
    for f in metrics_files:
        with open(f) as fp:
            all_metrics.append(json.load(fp))
    
    psnrs = [m.get('psnr', 0) for m in all_metrics if m.get('psnr')]
    ssims = [m.get('ssim', 0) for m in all_metrics if m.get('ssim')]
    lpips_vals = [m.get('lpips', 1) for m in all_metrics if m.get('lpips')]
    train_times = [m.get('train_time_sec', 0) for m in all_metrics]
    infer_times = [m.get('infer_time_sec', 0) for m in all_metrics]
    
    summary = {
        'num_videos': len(all_metrics),
        'num_steps': ${NUM_STEPS},
        'avg_psnr': float(np.mean(psnrs)) if psnrs else 0,
        'avg_ssim': float(np.mean(ssims)) if ssims else 0,
        'avg_lpips': float(np.mean(lpips_vals)) if lpips_vals else 0,
        'avg_train_time_sec': float(np.mean(train_times)),
        'avg_infer_time_sec': float(np.mean(infer_times)),
        'total_train_time_sec': float(sum(train_times)),
        'total_infer_time_sec': float(sum(infer_times)),
    }
    
    with open(f'{output_dir}/metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'Average PSNR: {summary[\"avg_psnr\"]:.2f}')
    print(f'Average SSIM: {summary[\"avg_ssim\"]:.4f}')
    print(f'Average LPIPS: {summary[\"avg_lpips\"]:.4f}')
    print(f'Average train time: {summary[\"avg_train_time_sec\"]:.1f}s')
"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo "LoRA Experiment (100 steps) Complete"
echo "========================================"
echo "Videos processed: ${VIDEO_COUNT}"
echo "Total training time: ${TOTAL_TRAIN_TIME} seconds"
echo "Average training time: $((TOTAL_TRAIN_TIME / VIDEO_COUNT)) seconds/video"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Metrics summary: ${OUTPUT_DIR}/metrics_summary.json"
echo ""
echo "End Time: $(date)"
echo "========================================"

