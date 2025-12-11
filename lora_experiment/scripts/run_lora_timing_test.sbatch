#!/bin/bash
#SBATCH --job-name=lora_timing_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/scripts/slurm_log/lora_timing_test_%j.out
#SBATCH --error=lora_experiment/scripts/slurm_log/lora_timing_test_%j.err

# ============================================================================
# LoRA Timing Test: Measure breakdown of model loading, encoding, training
# Runs 3 videos at different step counts to get timing data
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/timing_test"
CONFIG_FILE="${PROJECT_ROOT}/lora_experiment/configs/lora_finetune.py"
REFERENCE_SAMPLED_CSV="${PROJECT_ROOT}/lora_experiment/results/rank8_lr1e4_20steps/sampled_videos.csv"

echo "========================================"
echo "LoRA Timing Test"
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
cd "$PROJECT_ROOT"

nvidia-smi --query-gpu=name,memory.free --format=csv

mkdir -p "${OUTPUT_DIR}"

# Get first video from sampled CSV
FIRST_VIDEO=$(head -2 "${REFERENCE_SAMPLED_CSV}" | tail -1)
VIDEO_PATH=$(echo "$FIRST_VIDEO" | cut -d',' -f1)
CAPTION=$(echo "$FIRST_VIDEO" | cut -d',' -f7)
FULL_VIDEO_PATH="${DATA_DIR}/${VIDEO_PATH}"
VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)

echo "Test video: ${VIDEO_NAME}"
echo "Caption: ${CAPTION}"

# Test with different step counts
for STEPS in 10 20 50; do
    echo ""
    echo "========================================"
    echo "Testing ${STEPS} steps..."
    echo "========================================"
    
    TEST_DIR="${OUTPUT_DIR}/test_${STEPS}steps"
    mkdir -p "${TEST_DIR}"
    
    python lora_experiment/scripts/lora_finetune.py \
        --config "${CONFIG_FILE}" \
        --video-path "${FULL_VIDEO_PATH}" \
        --caption "${CAPTION}" \
        --output-dir "${TEST_DIR}" \
        --num-steps ${STEPS} \
        --lr 1e-4 \
        --rank 8 \
        --seed 42 \
        2>&1 | tee "${TEST_DIR}/output.log"
    
    echo ""
    echo "Results for ${STEPS} steps:"
    cat "${TEST_DIR}/training_info.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  Model load time: {d.get(\"model_load_time_sec\", 0):.2f}s')
print(f'  Video encode time: {d.get(\"video_encode_time_sec\", 0):.2f}s')
print(f'  Pure train time: {d.get(\"pure_train_time_sec\", 0):.2f}s')
print(f'  Total time: {d.get(\"total_time_sec\", 0):.2f}s')
print(f'  Time per step: {d.get(\"pure_train_time_sec\", 0) / ${STEPS}:.3f}s')
"
    
    # Clean up weights to save space
    rm -f "${TEST_DIR}/lora_weights.pt"
done

echo ""
echo "========================================"
echo "TIMING SUMMARY"
echo "========================================"

python3 -c "
import json
from pathlib import Path

results = []
for steps in [10, 20, 50]:
    info_file = Path('${OUTPUT_DIR}') / f'test_{steps}steps' / 'training_info.json'
    if info_file.exists():
        with open(info_file) as f:
            d = json.load(f)
        results.append({
            'steps': steps,
            'model_load': d.get('model_load_time_sec', 0),
            'encode': d.get('video_encode_time_sec', 0),
            'train': d.get('pure_train_time_sec', 0),
            'total': d.get('total_time_sec', 0),
        })

if results:
    avg_model_load = sum(r['model_load'] for r in results) / len(results)
    avg_encode = sum(r['encode'] for r in results) / len(results)
    
    print(f'Average model load time: {avg_model_load:.2f}s (one-time per video)')
    print(f'Average video encode time: {avg_encode:.2f}s (one-time per video)')
    print()
    print('Training time scaling:')
    for r in results:
        time_per_step = r['train'] / r['steps']
        print(f'  {r[\"steps\"]} steps: {r[\"train\"]:.2f}s total, {time_per_step:.3f}s/step')
    
    # Estimate pure gradient step time
    if len(results) >= 2:
        # Linear regression to find time per step
        steps = [r['steps'] for r in results]
        times = [r['train'] for r in results]
        avg_time_per_step = sum(times) / sum(steps)
        print()
        print(f'Estimated time per gradient step: {avg_time_per_step:.3f}s')
        print()
        print('Estimated total times at different step counts:')
        for s in [20, 50, 100]:
            overhead = avg_model_load + avg_encode
            train_time = s * avg_time_per_step
            total = overhead + train_time
            print(f'  {s} steps: {overhead:.1f}s overhead + {train_time:.1f}s training = {total:.1f}s total')
"

echo ""
echo "Done!"

