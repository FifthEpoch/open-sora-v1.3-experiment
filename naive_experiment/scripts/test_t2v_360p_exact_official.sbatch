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
export PYTHONPATH="/scratch/wc3013/open-sora-v1.3-experiment:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

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

