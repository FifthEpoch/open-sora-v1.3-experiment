#!/bin/bash
#SBATCH --job-name=lora_experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/scripts/slurm_log/lora_experiment_%j.out
#SBATCH --error=lora_experiment/scripts/slurm_log/lora_experiment_%j.err

# ============================================================================
# LoRA Experiment: Test-Time Adaptation with Low-Rank Adaptation
#
# LoRA trains <1% of parameters, expected to be 5-8x faster than full FT
# Default config: rank=8, alpha=16, lr=1e-4, 20 steps
#
# Estimated time: ~3-5 hours for 100 videos (vs ~13h for full FT at 50 steps)
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS=10                  # Number of videos to test (start small)
NUM_STEPS=20                   # LoRA training steps per video
LORA_RANK=8                    # LoRA rank
LORA_ALPHA=16                  # LoRA alpha
LEARNING_RATE=1e-4             # Learning rate for LoRA
CONDITION_FRAMES=22            # Conditioning frames

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_test"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"
CONFIG_FILE="${PROJECT_ROOT}/lora_experiment/configs/lora_finetune.py"

echo "========================================"
echo "LoRA Experiment"
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
if [ ! -f "${METADATA_CSV}" ]; then
    echo "ERROR: Metadata CSV not found at ${METADATA_CSV}" >&2
    exit 1
fi

echo "✓ Metadata validated"
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Create Output Directory
# ============================================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/lora_weights"
mkdir -p "${OUTPUT_DIR}/generated"

# ============================================================================
# Run LoRA Experiment
# ============================================================================
echo "========================================"
echo "Starting LoRA Experiment"
echo "========================================"

# Read video paths and captions from metadata
VIDEO_COUNT=0
TOTAL_TRAIN_TIME=0
TOTAL_INFER_TIME=0

# Process videos
# CSV columns: path, num_frames, conditioning_frames, height, width, fps, text
while IFS=, read -r video_path num_frames cond_frames height width fps caption || [ -n "$video_path" ]; do
    # Skip header
    if [ "$video_path" == "path" ]; then
        continue
    fi
    
    # Limit number of videos
    if [ $VIDEO_COUNT -ge $NUM_VIDEOS ]; then
        break
    fi
    
    # Get video name
    VIDEO_NAME=$(basename "$video_path" .mp4)
    
    echo ""
    echo "========================================"
    echo "Processing video $((VIDEO_COUNT + 1))/${NUM_VIDEOS}: ${VIDEO_NAME}"
    echo "========================================"
    
    # Create video-specific output directory
    VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/video_${VIDEO_COUNT}"
    mkdir -p "${VIDEO_OUTPUT_DIR}"
    
    # Full video path
    FULL_VIDEO_PATH="${DATA_DIR}/${video_path}"
    
    if [ ! -f "${FULL_VIDEO_PATH}" ]; then
        echo "WARNING: Video not found: ${FULL_VIDEO_PATH}, skipping..."
        continue
    fi
    
    # Step 1: LoRA Training
    echo "Step 1: LoRA Training..."
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
    
    # Check if LoRA weights were created
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
    
    # Copy LoRA weights to central location
    cp "${LORA_WEIGHTS}" "${OUTPUT_DIR}/lora_weights/lora_${VIDEO_COUNT}_${VIDEO_NAME}.pt"
    
    VIDEO_COUNT=$((VIDEO_COUNT + 1))
    
    echo "Video ${VIDEO_COUNT} completed. Train: ${TRAIN_TIME}s, Infer: ${INFER_TIME}s"
    
done < "${METADATA_CSV}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo "LoRA Experiment Complete"
echo "========================================"
echo "Videos processed: ${VIDEO_COUNT}"
echo "Total training time: ${TOTAL_TRAIN_TIME} seconds"
echo "Total inference time: ${TOTAL_INFER_TIME} seconds"
echo "Average training time: $((TOTAL_TRAIN_TIME / VIDEO_COUNT)) seconds/video"
echo "Average inference time: $((TOTAL_INFER_TIME / VIDEO_COUNT)) seconds/video"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Generated videos: ${OUTPUT_DIR}/generated/"
echo "LoRA weights: ${OUTPUT_DIR}/lora_weights/"
echo ""
echo "End Time: $(date)"
echo "========================================"

