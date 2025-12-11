#!/bin/bash
#SBATCH --job-name=t2v_infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=t2v_infer_experiment/scripts/slurm_log/t2v_experiment_%j.out
#SBATCH --error=t2v_infer_experiment/scripts/slurm_log/t2v_experiment_%j.err

# ============================================================================
# Text-to-Video Inference Experiment
# 
# Uses the SAME captions as the LoRA experiments but generates videos from
# text only (no conditioning frames). This serves as a baseline to compare:
# - How well does T2V generation match ground truth?
# - How much does video continuation (with conditioning) help?
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS=10                  # Same as LoRA experiments
NUM_FRAMES=49                  # Same total frames as LoRA experiments
SEED=42

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/t2v_infer_experiment/results/t2v_baseline"
CONFIG_FILE="${PROJECT_ROOT}/t2v_infer_experiment/configs/t2v_inference.py"

echo "========================================"
echo "Text-to-Video Inference Experiment"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "========================================"
echo "Configuration:"
echo "  Videos: ${NUM_VIDEOS}"
echo "  Frames: ${NUM_FRAMES}"
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
# Get the SAME sampled videos as LoRA experiments
# ============================================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/generated"

SAMPLED_CSV="${OUTPUT_DIR}/sampled_videos.csv"

# Search for the sampled_videos.csv from LoRA experiments
POSSIBLE_REFS=(
    "${PROJECT_ROOT}/lora_experiment/results/lora_20steps/sampled_videos.csv"
    "${PROJECT_ROOT}/lora_experiment/results/rank8_lr1e4_20steps/sampled_videos.csv"
    "${PROJECT_ROOT}/lora_experiment/results/lora_test/sampled_videos.csv"
    "${PROJECT_ROOT}/lora_experiment/results/rank8_lr1e4_100steps/sampled_videos.csv"
)

FOUND_REF=""
for REF in "${POSSIBLE_REFS[@]}"; do
    if [ -f "$REF" ]; then
        FOUND_REF="$REF"
        break
    fi
done

if [ -n "$FOUND_REF" ]; then
    cp "$FOUND_REF" "${SAMPLED_CSV}"
    echo "✓ Using same videos as LoRA experiments from: $FOUND_REF"
else
    echo "ERROR: Could not find sampled_videos.csv from LoRA experiments" >&2
    exit 1
fi

echo ""
echo "Videos to process:"
head -5 "${SAMPLED_CSV}"
echo "..."

# ============================================================================
# Run T2V Inference
# ============================================================================
echo ""
echo "========================================"
echo "Starting T2V Inference"
echo "========================================"

VIDEO_COUNT=0
TOTAL_GEN_TIME=0

# Read videos from the sampled CSV
while IFS=, read -r video_path num_frames cond_frames height width fps caption class_name || [ -n "$video_path" ]; do
    # Skip header
    if [ "$video_path" == "path" ]; then
        continue
    fi
    
    # Limit number of videos
    if [ $VIDEO_COUNT -ge $NUM_VIDEOS ]; then
        break
    fi
    
    VIDEO_NAME=$(basename "$video_path" .mp4)
    FULL_VIDEO_PATH="${DATA_DIR}/${video_path}"
    
    echo ""
    echo "========================================"
    echo "Processing video $((VIDEO_COUNT + 1))/${NUM_VIDEOS}: ${VIDEO_NAME}"
    echo "Caption: ${caption}"
    echo "========================================"
    
    # Create video output directory
    VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/video_${VIDEO_COUNT}"
    mkdir -p "${VIDEO_OUTPUT_DIR}"
    
    # Generate video using T2V
    GEN_START=$(date +%s)
    
    python t2v_infer_experiment/scripts/t2v_inference.py \
        --config "${CONFIG_FILE}" \
        --prompt "${caption}" \
        --output-path "${OUTPUT_DIR}/generated/t2v_${VIDEO_COUNT}_${VIDEO_NAME}.mp4" \
        --num-frames ${NUM_FRAMES} \
        --seed ${SEED} \
        2>&1 | tee "${VIDEO_OUTPUT_DIR}/inference.log"
    
    GEN_END=$(date +%s)
    GEN_TIME=$((GEN_END - GEN_START))
    TOTAL_GEN_TIME=$((TOTAL_GEN_TIME + GEN_TIME))
    
    echo "Generation time: ${GEN_TIME} seconds"
    
    # Compute metrics comparing T2V output to ground truth
    GENERATED_VIDEO="${OUTPUT_DIR}/generated/t2v_${VIDEO_COUNT}_${VIDEO_NAME}.mp4"
    CONDITION_FRAMES=22  # Same as LoRA experiments
    
    if [ -f "${GENERATED_VIDEO}" ] && [ -f "${FULL_VIDEO_PATH}" ]; then
        echo "Computing metrics..."
        python -c "
import torch
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

from naive_experiment.scripts.evaluate_continuations import extract_frames_from_video, load_generated_video, evaluate_pair

original = '${FULL_VIDEO_PATH}'
generated = '${GENERATED_VIDEO}'
condition_frames = ${CONDITION_FRAMES}

try:
    # Extract ground truth frames (the frames we would have generated in V2V)
    # This is frames 22-49 of the original video
    gt_frames = extract_frames_from_video(original, start_frame=condition_frames, num_frames=27)
    
    # For T2V, we compare the generated continuation frames (after frame 22)
    # to the ground truth continuation
    gen_frames = load_generated_video(generated)
    if gen_frames is not None and len(gen_frames) > condition_frames:
        gen_frames = gen_frames[condition_frames:]
    
    if gt_frames is not None and gen_frames is not None:
        metrics = evaluate_pair(gt_frames, gen_frames)
    else:
        metrics = {'error': 'Could not load frames'}
    
    # Save metrics
    metrics_file = '${VIDEO_OUTPUT_DIR}/metrics.json'
    metrics['video_idx'] = ${VIDEO_COUNT}
    metrics['video_name'] = '${VIDEO_NAME}'
    metrics['caption'] = '${caption}'
    metrics['gen_time_sec'] = ${GEN_TIME}
    metrics['method'] = 't2v'
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    
    if 'psnr' in metrics:
        print(f'  PSNR: {metrics[\"psnr\"]:.2f}')
        print(f'  SSIM: {metrics[\"ssim\"]:.4f}')
        print(f'  LPIPS: {metrics[\"lpips\"]:.4f}')
    else:
        print(f'  Metrics: {metrics}')
        
except Exception as e:
    import traceback
    print(f'  Metrics computation failed: {e}')
    traceback.print_exc()
" 2>&1
    fi
    
    VIDEO_COUNT=$((VIDEO_COUNT + 1))
    echo "Video ${VIDEO_COUNT} completed in ${GEN_TIME}s"
    
done < "${SAMPLED_CSV}"

# ============================================================================
# Aggregate metrics
# ============================================================================
echo ""
echo "========================================"
echo "Aggregating metrics..."
echo "========================================"

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
    for f in sorted(metrics_files):
        with open(f) as fp:
            all_metrics.append(json.load(fp))
    
    psnrs = [m.get('psnr', 0) for m in all_metrics if m.get('psnr')]
    ssims = [m.get('ssim', 0) for m in all_metrics if m.get('ssim')]
    lpips_vals = [m.get('lpips', 1) for m in all_metrics if m.get('lpips')]
    gen_times = [m.get('gen_time_sec', 0) for m in all_metrics]
    
    summary = {
        'method': 't2v_baseline',
        'num_videos': len(all_metrics),
        'avg_psnr': float(np.mean(psnrs)) if psnrs else 0,
        'avg_ssim': float(np.mean(ssims)) if ssims else 0,
        'avg_lpips': float(np.mean(lpips_vals)) if lpips_vals else 0,
        'std_psnr': float(np.std(psnrs)) if psnrs else 0,
        'std_ssim': float(np.std(ssims)) if ssims else 0,
        'std_lpips': float(np.std(lpips_vals)) if lpips_vals else 0,
        'avg_gen_time_sec': float(np.mean(gen_times)),
        'total_gen_time_sec': sum(gen_times),
    }
    
    # Save summary
    with open(f'{output_dir}/metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('=' * 50)
    print('T2V BASELINE RESULTS')
    print('=' * 50)
    print(f'Videos processed: {summary[\"num_videos\"]}')
    print()
    print(f'Average PSNR:  {summary[\"avg_psnr\"]:.2f} ± {summary[\"std_psnr\"]:.2f}')
    print(f'Average SSIM:  {summary[\"avg_ssim\"]:.4f} ± {summary[\"std_ssim\"]:.4f}')
    print(f'Average LPIPS: {summary[\"avg_lpips\"]:.4f} ± {summary[\"std_lpips\"]:.4f}')
    print()
    print(f'Average generation time: {summary[\"avg_gen_time_sec\"]:.1f}s per video')
    print('=' * 50)
"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo "T2V Inference Experiment Complete"
echo "========================================"
echo "Videos processed: ${VIDEO_COUNT}"
echo "Total generation time: ${TOTAL_GEN_TIME} seconds"
echo "Average time per video: $((TOTAL_GEN_TIME / VIDEO_COUNT)) seconds"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Generated videos: ${OUTPUT_DIR}/generated/"
echo "Metrics summary: ${OUTPUT_DIR}/metrics_summary.json"
echo ""
echo "End Time: $(date)"
echo "========================================"

