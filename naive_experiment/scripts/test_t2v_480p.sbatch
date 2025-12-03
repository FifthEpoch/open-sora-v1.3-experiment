#!/bin/bash
#SBATCH --job-name=test_480p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --output=naive_experiment/scripts/slurm_log/test_t2v_480p_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/test_t2v_480p_%j.err

# ===================================================================
# Test T2V at 480p 9:16 - middle ground between 360p and 720p
# ===================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="/scratch/wc3013/open-sora-v1.3-experiment"
SCRATCH_BASE="/scratch/wc3013"

cd "${PROJECT_ROOT}"

echo "========================================"
echo "Test T2V at 480p 9:16"
echo "========================================"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

# --- Environment Setup ---
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

export CONDA_ENVS_DIRS="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true

echo "Activating conda environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"
if [ $? -ne 0 ] || [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: Failed to activate conda environment" >&2
    exit 1
fi
echo "✓ Environment activated: $CONDA_PREFIX"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== Testing 480p 9:16 ==="
echo "480p (480, 854) is between 360p (broken) and 720p (works but OOMs)"
echo "If 480p works, we can use it for the V2V experiment!"
echo ""

mkdir -p naive_experiment/results

python naive_experiment/scripts/test_t2v_480p.py \
    --prompt "A person applying eye makeup in a bathroom mirror." \
    --output naive_experiment/results/test_t2v_480p.mp4

echo ""
echo "=== Test complete ==="
echo "Check naive_experiment/results/test_t2v_480p.mp4"
echo ""
echo "INTERPRETATION:"
echo "- If video looks good: 480p works! Use it for the experiment."
echo "- If video has RGB blocks: Issue affects all resolutions < 720p"

