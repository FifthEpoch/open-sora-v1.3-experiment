#!/bin/bash
#SBATCH --job-name=test_t2v_360p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --output=naive_experiment/scripts/slurm_log/test_t2v_360p_exact_official_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/test_t2v_360p_exact_official_%j.err

# ===================================================================
# Test T2V at 360p using EXACT official config
# This is a quick diagnostic test before running the full experiment.
# ===================================================================

set -euo pipefail

# Prevent user site-packages from interfering
export PYTHONNOUSERSITE=1

PROJECT_ROOT="/scratch/wc3013/open-sora-v1.3-experiment"
SCRATCH_BASE="/scratch/wc3013"

cd "${PROJECT_ROOT}"

echo "========================================"
echo "Test T2V at 360p with EXACT Official Config"
echo "========================================"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

# --- Environment Setup ---
module purge
module load anaconda3/2025.06

# Source conda initialization
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Configure conda directories
export CONDA_ENVS_DIRS="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true

# Activate conda environment
echo "Activating conda environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"
if [ $? -ne 0 ] || [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: Failed to activate conda environment" >&2
    exit 1
fi
echo "✓ Environment activated: $CONDA_PREFIX"

# Ensure conda env's bin is first in PATH
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# Verify Python
PYTHON_PATH=$(which python)
echo "Using Python: ${PYTHON_PATH}"
python --version

# Set environment variables
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Verify PyTorch
echo ""
echo "Verifying PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Testing T2V at 360p with EXACT Official Config ==="
echo ""
echo "This uses the EXACT parameters from configs/opensora-v1-3/inference/t2v.py"
echo "Key settings:"
echo "  - tile_size=4 (official default)"
echo "  - use_oscillation_guidance=True"
echo "  - use_flaw_fix=True"
echo ""

# Create output directory
mkdir -p naive_experiment/results

# Run T2V test
python naive_experiment/scripts/test_t2v_360p_exact_official.py \
    --prompt "A person applying eye makeup in a bathroom mirror." \
    --output naive_experiment/results/test_t2v_360p_exact_official.mp4

echo ""
echo "=== Test complete ==="
echo "Check naive_experiment/results/test_t2v_360p_exact_official.mp4"
echo ""
echo "INTERPRETATION:"
echo "- If video looks good: 360p T2V works with official config!"
echo "- If video has RGB blocks: Issue is fundamental to 360p."
