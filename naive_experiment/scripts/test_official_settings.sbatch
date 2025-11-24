#!/bin/bash
#SBATCH --job-name=test_official_settings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --output=../results/debug/test_official_%j.out
#SBATCH --error=../results/debug/test_official_%j.err

# Test with OFFICIAL Open-Sora v1.3 video extending settings
# Reference: https://github.com/hpcaitech/Open-Sora/blob/opensora/v1.3/docs/commands.md
#
# Key differences from our previous settings:
# - image_cfg_scale: 2.0 (not 5.0!)
# - cfg_scale: 7.5 (not 8.5)
# - aspect_ratio: 9:16 portrait (not 4:3 landscape)
#
# USAGE:
#   cd naive_experiment/scripts
#   sbatch test_official_settings.sbatch

set -eo pipefail

echo "=========================================="
echo "Official Settings Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Change to repo root
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

mkdir -p naive_experiment/results/debug

# Load modules
echo "Loading modules..."
module purge
module load anaconda3/2025.06

# Activate conda environment
echo "Activating conda environment..."
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda activate opensora13
set -u

# Fix library path conflicts
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Test video
TEST_VIDEO="env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4"
TEST_CAPTION="A person applying eye makeup"
OUTPUT_VIDEO="naive_experiment/results/debug/test_official_settings_output.mp4"

if [ ! -f "$TEST_VIDEO" ]; then
    echo "ERROR: Test video not found: $TEST_VIDEO"
    exit 1
fi

echo "Test video: $TEST_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Config: naive_experiment/configs/baseline_inference_official.py"

# Run test with OFFICIAL settings
echo "=========================================="
echo "Running with OFFICIAL Open-Sora v1.3 settings:"
echo "  - image_cfg_scale: 2.0 (was 5.0)"
echo "  - cfg_scale: 7.5 (was 8.5)"
echo "  - aspect_ratio: 9:16 portrait (was 4:3 landscape)"
echo "  - condition_frame_length: 5 (latent)"
echo "  - use_sdedit: True"
echo "=========================================="

python naive_experiment/scripts/test_conditioning_debug.py \
    --config naive_experiment/configs/baseline_inference_official.py \
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
echo "COMPARE ALL THREE TESTS:"
echo "1. Baseline:  naive_experiment/results/debug/test_conditioning_output.mp4"
echo "   - image_cfg_scale: 5.0, aspect: 4:3"
echo "2. Strong:    naive_experiment/results/debug/test_strong_conditioning_output.mp4"
echo "   - image_cfg_scale: 10.0, aspect: 4:3"
echo "3. Official:  $OUTPUT_VIDEO"
echo "   - image_cfg_scale: 2.0, aspect: 9:16 (PORTRAIT)"
echo ""
echo "If Official settings work best, use those for experiments!"

