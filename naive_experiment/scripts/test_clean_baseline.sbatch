#!/bin/bash
#SBATCH --job-name=test_clean_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --time=01:00:00
#SBATCH --output=../results/debug/test_clean_baseline_%j.out
#SBATCH --error=../results/debug/test_clean_baseline_%j.err

# This script tests the CLEAN baseline configuration with NO tiling artifacts
# Key difference: use_tiled_conv3d=False to eliminate rectangular flashes

echo "=================================================="
echo "Testing CLEAN Baseline (No Tiling Artifacts)"
echo "Config: baseline_inference_clean.py"
echo "Key changes:"
echo "  - use_tiled_conv3d=False (no rectangular flashes)"
echo "  - condition_frame_length=7 (more context)"
echo "  - image_cfg_scale=5.0 (moderate guidance)"
echo "  - aspect_ratio=3:4 (landscape)"
echo "=================================================="
echo ""

# Change to repository root
cd ../..

# Setup environment
module purge
module load anaconda3/2025.06

# Activate conda environment
source activate opensora13

# Prioritize conda environment libraries to avoid GLIBCXX errors
export LD_LIBRARY_PATH=$CONDA_ENVS_PATH/opensora13/lib:$LD_LIBRARY_PATH

# Set cache directory
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface/transformers

# Print environment info
echo "Environment Info:"
echo "  Hostname: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  CUDA: $(nvcc --version | grep release)"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Create output directory
mkdir -p naive_experiment/results/debug

# Run test with CLEAN baseline config (no tiling artifacts)
echo "Running clean baseline test..."
python naive_experiment/scripts/test_conditioning_debug.py \
    --config naive_experiment/configs/baseline_inference_clean.py \
    --video-path env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 \
    --caption "A person applying eye makeup" \
    --output-path naive_experiment/results/debug/test_clean_baseline_output.mp4

echo ""
echo "=================================================="
echo "Test completed!"
echo "Output: naive_experiment/results/debug/test_clean_baseline_output.mp4"
echo "=================================================="

