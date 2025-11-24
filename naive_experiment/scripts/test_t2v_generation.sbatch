#!/bin/bash
#SBATCH --job-name=test_t2v
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --time=01:00:00
#SBATCH --output=../results/debug/test_t2v_%j.out
#SBATCH --error=../results/debug/test_t2v_%j.err

# Test pure text-to-video generation (no conditioning)
# Generate videos from UCF-101 category prompts to establish baseline quality

echo "=================================================="
echo "TEXT-TO-VIDEO GENERATION TEST"
echo "Testing model's baseline capability without conditioning"
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

# Prioritize conda environment libraries
export LD_LIBRARY_PATH=$CONDA_ENVS_PATH/opensora13/lib:$LD_LIBRARY_PATH

# Print environment info
echo "Environment Info:"
echo "  Hostname: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Create output directory
mkdir -p naive_experiment/results/debug/t2v_samples

# Test prompts from UCF-101 categories
echo "=================================================="
echo "Generating videos from text prompts..."
echo "=================================================="
echo ""

# Prompt 1: ApplyEyeMakeup (same as conditioning test)
echo "1/5: Generating 'A person applying eye makeup'..."
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_generation_test.py \
    --prompt "A person applying eye makeup" \
    --output-path naive_experiment/results/debug/t2v_samples/apply_eye_makeup.mp4 \
    --seed 42

echo ""
echo "---"
echo ""

# Prompt 2: Basketball
echo "2/5: Generating 'A person playing basketball'..."
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_generation_test.py \
    --prompt "A person playing basketball" \
    --output-path naive_experiment/results/debug/t2v_samples/basketball.mp4 \
    --seed 43

echo ""
echo "---"
echo ""

# Prompt 3: Cycling
echo "3/5: Generating 'A person riding a bicycle'..."
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_generation_test.py \
    --prompt "A person riding a bicycle" \
    --output-path naive_experiment/results/debug/t2v_samples/cycling.mp4 \
    --seed 44

echo ""
echo "---"
echo ""

# Prompt 4: Dancing
echo "4/5: Generating 'A person dancing'..."
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_generation_test.py \
    --prompt "A person dancing" \
    --output-path naive_experiment/results/debug/t2v_samples/dancing.mp4 \
    --seed 45

echo ""
echo "---"
echo ""

# Prompt 5: Swimming
echo "5/5: Generating 'A person swimming'..."
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_generation_test.py \
    --prompt "A person swimming" \
    --output-path naive_experiment/results/debug/t2v_samples/swimming.mp4 \
    --seed 46

echo ""
echo "=================================================="
echo "ALL GENERATIONS COMPLETE"
echo "=================================================="
echo "Output directory: naive_experiment/results/debug/t2v_samples/"
echo "Files generated:"
echo "  - apply_eye_makeup.mp4"
echo "  - basketball.mp4"
echo "  - cycling.mp4"
echo "  - dancing.mp4"
echo "  - swimming.mp4"
echo "=================================================="

