#!/bin/bash
#SBATCH --job-name=lora_r32_mlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/scripts/slurm_log/lora_rank32_mlp_%j.out
#SBATCH --error=lora_experiment/scripts/slurm_log/lora_rank32_mlp_%j.err

# ============================================================================
# LoRA Experiment: Rank 32 + MLP Targeting
# 
# Hypothesis: Targeting MLP layers (fc1, fc2) in addition to attention
# provides more capacity and can close the gap with full fine-tuning.
#
# Config: rank=32, alpha=64, lr=2e-4, 100 steps, target_mlp=True
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS=10
NUM_STEPS=100
LORA_RANK=32
LORA_ALPHA=64
LEARNING_RATE=2e-4
TARGET_MLP=true                # NEW: Target MLP layers too!
CONDITION_FRAMES=22

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/rank32_lr2e4_100steps_mlp"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"
CONFIG_FILE="${PROJECT_ROOT}/lora_experiment/configs/lora_finetune.py"

echo "========================================"
echo "LoRA Experiment: Rank 32 + MLP"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "========================================"
echo "Configuration:"
echo "  Videos: ${NUM_VIDEOS}"
echo "  Steps: ${NUM_STEPS}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  LoRA Alpha: ${LORA_ALPHA}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Target MLP: ${TARGET_MLP}"
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
if [ ! -f "${METADATA_CSV}" ]; then
    echo "ERROR: Metadata CSV not found at ${METADATA_CSV}" >&2
    exit 1
fi

echo "✓ Metadata validated"
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Create Output Directory & Reuse Sampled Videos
# ============================================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/generated"

SAMPLED_CSV="${OUTPUT_DIR}/sampled_videos.csv"

# Search for sampled_videos.csv in multiple possible locations
POSSIBLE_REFS=(
    "${PROJECT_ROOT}/lora_experiment/results/rank8_lr1e4_20steps/sampled_videos.csv"
    "${PROJECT_ROOT}/lora_experiment/results/lora_20steps/sampled_videos.csv"
    "${PROJECT_ROOT}/lora_experiment/results/lora_test/sampled_videos.csv"
    "${PROJECT_ROOT}/lora_experiment/results/rank8_lr1e4_100steps/sampled_videos.csv"
    "${PROJECT_ROOT}/lora_experiment/results/rank8_lr2e4_100steps/sampled_videos.csv"
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
    echo "✓ Copied sampled_videos.csv from: $FOUND_REF"
else
    echo "ERROR: Could not find sampled_videos.csv in any known location" >&2
    exit 1
fi

# ============================================================================
# Create temporary config with target_mlp=True
# ============================================================================
TEMP_CONFIG="${OUTPUT_DIR}/lora_finetune_mlp.py"
cat > "${TEMP_CONFIG}" << 'EOF'
# LoRA config with MLP targeting

dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

bucket_config = {
    "360p": {
        22: (1, 1),
    },
}

mask_types = {
    "v2v_head": 1,
}

drop_condition = {
    "cond": 0.0,
    "text": 0.0,
    "null": 0.0,
    "keep": 1.0,
}

grad_checkpoint = True
num_workers = 0
num_bucket_build_workers = 1
dtype = "bf16"
plugin = "zero2"

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4-360p",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    kernel_size=(8, 8, -1),
    use_spatial_rope=True,
    class_dropout_prob=0.0,
    force_huggingface=True,
)

vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)

text_encoder = dict(
    type="t5",
    from_pretrained="google/t5-v1_1-xxl",
    model_max_length=300,
)

scheduler = dict(
    type="rflow",
    sample_method="logit-normal",
    use_timestep_transform=True,
    drop_condition=drop_condition,
)

# LoRA config WITH MLP TARGETING
lora = dict(
    rank=32,
    alpha=64,
    target_modules=["qkv", "proj"],
    target_blocks="all",
    target_mlp=True,  # KEY: Target MLP layers too!
)

seed = 42
outputs = "lora_experiment/results/lora_checkpoints"
wandb = False
epochs = 1
log_every = 1
ckpt_every = 10

lr = 2e-4
warmup_steps = 0
use_cosine_scheduler = False
grad_clip = 1.0
adam_eps = 1e-15
accumulation_steps = 1
EOF

echo "✓ Created config with target_mlp=True: ${TEMP_CONFIG}"

# ============================================================================
# Run LoRA Experiment
# ============================================================================
echo "========================================"
echo "Starting LoRA Experiment (with MLP targeting)"
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
    
    # Step 1: LoRA Training (using config with target_mlp=True)
    echo "Step 1: LoRA Training (with MLP)..."
    TRAIN_START=$(date +%s)
    
    python lora_experiment/scripts/lora_finetune.py \
        --config "${TEMP_CONFIG}" \
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
    
    # Step 2: LoRA Inference (with MLP targeting)
    echo "Step 2: LoRA Inference (with MLP)..."
    INFER_START=$(date +%s)
    
    python lora_experiment/scripts/lora_inference.py \
        --config "${TEMP_CONFIG}" \
        --lora-weights "${LORA_WEIGHTS}" \
        --video-path "${FULL_VIDEO_PATH}" \
        --caption "${caption}" \
        --output-path "${OUTPUT_DIR}/generated/lora_${VIDEO_COUNT}_${VIDEO_NAME}.mp4" \
        --condition-frames ${CONDITION_FRAMES} \
        --rank ${LORA_RANK} \
        --alpha ${LORA_ALPHA} \
        --target-mlp \
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
        'avg_psnr': np.mean(psnrs) if psnrs else 0,
        'avg_ssim': np.mean(ssims) if ssims else 0,
        'avg_lpips': np.mean(lpips_vals) if lpips_vals else 0,
        'avg_train_time_sec': np.mean(train_times),
        'avg_infer_time_sec': np.mean(infer_times),
        'config': {
            'rank': ${LORA_RANK}, 
            'alpha': ${LORA_ALPHA}, 
            'lr': ${LEARNING_RATE}, 
            'steps': ${NUM_STEPS},
            'target_mlp': True,
        },
    }
    
    with open(f'{output_dir}/metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'Average PSNR: {summary[\"avg_psnr\"]:.2f}')
    print(f'Average SSIM: {summary[\"avg_ssim\"]:.4f}')
    print(f'Average LPIPS: {summary[\"avg_lpips\"]:.4f}')
    print(f'Average train time: {summary[\"avg_train_time_sec\"]:.1f}s')
"

echo ""
echo "========================================"
echo "LoRA Rank 32 + MLP Experiment Complete"
echo "========================================"
echo "Videos processed: ${VIDEO_COUNT}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "End Time: $(date)"
echo "========================================"

