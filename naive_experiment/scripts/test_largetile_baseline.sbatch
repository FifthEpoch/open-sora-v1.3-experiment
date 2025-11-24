#!/bin/bash
#SBATCH --job-name=test_largetile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --time=01:00:00
#SBATCH --output=../results/debug/test_largetile_%j.out
#SBATCH --error=../results/debug/test_largetile_%j.err

# This script tests LARGE TILE baseline (tile_size=16 vs 4)
# Should dramatically reduce artifacts while using normal memory (64GB)
# Alternative to no-tiling (which needs 128GB)

echo "=================================================="
echo "Testing LARGE TILE Baseline (tile_size=16)"
echo "Config: baseline_inference_largetile.py"
echo "Key changes:"
echo "  - tile_size=16 (vs 4 in old configs)"
echo "  - condition_frame_length=7 (more context)"
echo "  - image_cfg_scale=5.0 (moderate guidance)"
echo "  - aspect_ratio=3:4 (landscape)"
echo "  - Memory: 64GB (normal, vs 128GB for no-tiling)"
echo "=================================================="
echo ""

# Change to repository root
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

# Create necessary directories
mkdir -p ${CONDA_ENVS_PATH} ${CONDA_PKGS_DIRS} ${PIP_CACHE_DIR}
mkdir -p ${TORCH_HOME} ${TORCH_EXTENSIONS_DIR} ${TRITON_CACHE_DIR}
mkdir -p ${HF_HOME} ${HF_HUB_CACHE} ${HF_DATASETS_CACHE} ${TMPDIR}

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
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Create output directory
mkdir -p naive_experiment/results/debug

# Run test with LARGE TILE config
echo "Running large tile baseline test..."
python naive_experiment/scripts/test_conditioning_debug.py \
    --config naive_experiment/configs/baseline_inference_largetile.py \
    --video-path env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 \
    --caption "A person applying eye makeup" \
    --output-path naive_experiment/results/debug/test_largetile_output.mp4

echo ""
echo "=================================================="
echo "Test completed!"
echo "Output: naive_experiment/results/debug/test_largetile_output.mp4"
echo "=================================================="

