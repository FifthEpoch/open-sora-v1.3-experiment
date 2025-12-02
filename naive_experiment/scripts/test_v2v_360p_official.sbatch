#!/bin/bash
#SBATCH --job-name=test_v2v_360p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=128G
#SBATCH --output=naive_experiment/scripts/slurm_log/test_v2v_360p_official_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/test_v2v_360p_official_%j.err

# Test V2V at 360p using EXACT official config
# This is to diagnose the RGB blocks issue

cd /scratch/wc3013/open-sora-v1.3-experiment

# --- Environment Setup ---
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Activate conda environment
conda activate /scratch/wc3013/conda-envs/opensora13

# Set environment variables
export HF_HOME=/scratch/wc3013/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/wc3013/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Testing V2V at 360p with EXACT Official Config ==="
echo "This uses the EXACT parameters from configs/opensora-v1-3/inference/v2v.py"
echo ""

# First, we need a conditioning video at 360p 9:16 (360x640)
# Let's create one from an existing UCF-101 video
COND_VIDEO="/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/results/test_cond_360p.mp4"

# Create conditioning video at 360p 9:16 from UCF-101
echo "Creating conditioning video at 360p 9:16..."
python -c "
import cv2
import numpy as np

# Read original video
cap = cv2.VideoCapture('/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4')

# Target: 360p 9:16 = (360, 640) in H,W = 640x360 in W,H for cv2
target_h, target_w = 360, 640
fps = 24
num_frames = 22  # Conditioning frames

# Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('${COND_VIDEO}', fourcc, fps, (target_w, target_h))

frame_count = 0
while cap.isOpened() and frame_count < num_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Center crop and resize to 360x640 (H,W)
    h, w = frame.shape[:2]
    
    # Calculate crop to match 9:16 aspect ratio
    target_aspect = target_w / target_h  # 640/360 = 1.78
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        # Too wide, crop width
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        frame = frame[:, start_x:start_x+new_w]
    else:
        # Too tall, crop height
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        frame = frame[start_y:start_y+new_h, :]
    
    # Resize to target
    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f'Created conditioning video with {frame_count} frames at {target_w}x{target_h}')
"

# Verify the conditioning video
echo ""
echo "Verifying conditioning video..."
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "${COND_VIDEO}"

# Run V2V test
echo ""
echo "Running V2V test..."
python naive_experiment/scripts/test_v2v_360p_official.py \
    --video-path "${COND_VIDEO}" \
    --prompt "A person applying eye makeup." \
    --output naive_experiment/results/test_v2v_360p_official.mp4

echo ""
echo "=== Test complete ==="
echo "Check naive_experiment/results/test_v2v_360p_official.mp4"
echo ""
echo "INTERPRETATION:"
echo "- If video looks good: 360p V2V works with official config!"
echo "- If video has RGB blocks: Issue is fundamental to 360p or environment."

