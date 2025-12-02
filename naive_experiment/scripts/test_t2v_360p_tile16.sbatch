#!/bin/bash
#SBATCH --job-name=test_tile16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=128G
#SBATCH --output=naive_experiment/scripts/slurm_log/test_t2v_360p_tile16_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/test_t2v_360p_tile16_%j.err

# Test T2V at 360p with tile_size=16 (same as working 720p config)
# This tests if tile_size=4 was causing the RGB blocks

cd /scratch/wc3013/open-sora-v1.3-experiment

# Activate environment
source /scratch/wc3013/conda-envs/opensora13/bin/activate

# Set environment variables
export HF_HOME=/scratch/wc3013/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/wc3013/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Testing T2V at 360p with tile_size=16 ==="
echo ""
echo "WORKING 720p config used:"
echo "  - tile_size=16"
echo "  - num_sampling_steps=60"
echo "  - cfg_scale=10.0"
echo "  - use_oscillation_guidance=True"
echo "  - use_flaw_fix=True"
echo ""
echo "FAILING 360p tests used:"
echo "  - tile_size=4  ← SUSPECT!"
echo "  - num_sampling_steps=30"
echo "  - cfg_scale=7.5"
echo "  - use_oscillation_guidance=False"
echo "  - use_flaw_fix=False"
echo ""

# Run T2V test with tile_size=16
python naive_experiment/scripts/test_t2v_360p_tile16.py \
    --prompt "A person applying eye makeup in front of a mirror." \
    --output naive_experiment/results/test_t2v_360p_tile16.mp4

echo ""
echo "=== Test complete ==="
echo "Check naive_experiment/results/test_t2v_360p_tile16.mp4"
echo ""
echo "INTERPRETATION:"
echo "- If video looks good: tile_size=4 was the problem! Use tile_size=16."
echo "- If video still has RGB blocks: Issue is elsewhere (not tile_size)."

