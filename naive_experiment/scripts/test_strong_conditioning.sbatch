#!/bin/bash
#SBATCH --job-name=test_strong_conditioning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --output=../results/debug/test_strong_%j.out
#SBATCH --error=../results/debug/test_strong_%j.err

# Test STRONG conditioning settings to see if we can improve video quality
# 
# USAGE:
#   cd naive_experiment/scripts
#   sbatch test_strong_conditioning.sbatch

set -eo pipefail

echo "=========================================="
echo "Strong Conditioning Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Change to repo root (2 levels up from scripts/)
cd ../..

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

# Fix library path conflicts - prioritize conda environment libraries over system anaconda
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Get a test video from UCF-101
TEST_VIDEO="env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4"
TEST_CAPTION="A person applying eye makeup"
OUTPUT_VIDEO="naive_experiment/results/debug/test_strong_conditioning_output.mp4"

if [ ! -f "$TEST_VIDEO" ]; then
    echo "ERROR: Test video not found: $TEST_VIDEO"
    exit 1
fi

echo "Test video: $TEST_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Config: naive_experiment/configs/baseline_inference_strong.py"

# Run debug test with STRONG conditioning config
echo "=========================================="
echo "Running STRONG conditioning test..."
echo "Settings:"
echo "  - condition_frame_length: 7 (more latent frames)"
echo "  - image_cfg_scale: 10.0 (2x stronger)"
echo "  - cfg_scale: 10.0 (stronger)"
echo "  - use_sdedit: False (no noise)"
echo "=========================================="

python naive_experiment/scripts/test_conditioning_debug.py \
    --config naive_experiment/configs/baseline_inference_strong.py \
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
echo "COMPARE THIS TO PREVIOUS TEST:"
echo "Previous: naive_experiment/results/debug/test_conditioning_output.mp4"
echo "Current:  $OUTPUT_VIDEO"
echo ""
echo "If quality is better, update baseline_inference.py with these settings!"

