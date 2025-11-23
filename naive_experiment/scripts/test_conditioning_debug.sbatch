#!/bin/bash
#SBATCH --job-name=test_conditioning_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=g40,gpu
#SBATCH --output=naive_experiment/results/debug/test_conditioning_%j.out
#SBATCH --error=naive_experiment/results/debug/test_conditioning_%j.err

# USAGE:
#   sbatch naive_experiment/scripts/test_conditioning_debug.sbatch
#
# This will test ONE video and output extensive debug information to verify
# that conditioning frames are being used correctly.

set -eo pipefail

echo "=========================================="
echo "Conditioning Debug Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Setup environment
SCRATCH_BASE=/scratch/${USER}
export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda-envs
export CONDA_PKGS_DIRS=${SCRATCH_BASE}/conda-pkgs
export PIP_CACHE_DIR=${SCRATCH_BASE}/py-cache/pip
export TORCH_HOME=${SCRATCH_BASE}/py-cache/torch
export TORCH_EXTENSIONS_DIR=${SCRATCH_BASE}/py-cache/torch/extensions
export PYTORCH_KERNEL_CACHE_PATH=${SCRATCH_BASE}/py-cache/models
export TRITON_CACHE_DIR=${SCRATCH_BASE}/py-cache/triton
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH_BASE}/py-cache/torch-inductor
export COLOSSALAI_CACHE_DIR=${SCRATCH_BASE}/py-cache/colossalai-jit
export HF_HOME=${SCRATCH_BASE}/hf-cache
export HF_HUB_CACHE=${SCRATCH_BASE}/hf-cache/hub
export HF_DATASETS_CACHE=${SCRATCH_BASE}/hf-cache/datasets
export TRANSFORMERS_CACHE=${SCRATCH_BASE}/hf-cache/transformers
export TMPDIR=${SCRATCH_BASE}/tmp
export TEMP=${SCRATCH_BASE}/tmp
export TMP=${SCRATCH_BASE}/tmp

# Create directories
mkdir -p naive_experiment/results/debug

# Load modules
echo "Loading modules..."
module purge
module load anaconda3/2025.06

# Activate conda environment
echo "Activating conda environment..."
set +u  # Temporarily disable unbound variable check for conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda activate opensora13
set -u  # Re-enable unbound variable check

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Get a test video from UCF-101
TEST_VIDEO="/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4"
TEST_CAPTION="A person applying eye makeup"
OUTPUT_VIDEO="naive_experiment/results/debug/test_conditioning_output.mp4"

if [ ! -f "$TEST_VIDEO" ]; then
    echo "ERROR: Test video not found: $TEST_VIDEO"
    echo "Please update TEST_VIDEO path in this script"
    exit 1
fi

echo "Test video: $TEST_VIDEO"
echo "Output video: $OUTPUT_VIDEO"

# Run debug test
echo "=========================================="
echo "Running conditioning debug test..."
echo "=========================================="

python naive_experiment/scripts/test_conditioning_debug.py \
    --config naive_experiment/configs/baseline_inference.py \
    --video-path "$TEST_VIDEO" \
    --caption "$TEST_CAPTION" \
    --output-path "$OUTPUT_VIDEO"

echo "=========================================="
echo "Test complete!"
echo "Output video: $OUTPUT_VIDEO"
echo "Conditioning video: naive_experiment/results/debug/debug_conditioning/conditioning.mp4"
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "NEXT STEPS:"
echo "1. Check the .out file for detailed debug information"
echo "2. Download both videos to compare visually:"
echo "   - $OUTPUT_VIDEO"
echo "   - naive_experiment/results/debug/debug_conditioning/conditioning.mp4"
echo "3. Verify the first 22 frames of output match the conditioning video"

