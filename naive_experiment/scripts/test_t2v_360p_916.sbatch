#!/bin/bash
#SBATCH --job-name=test_t2v_916
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --output=naive_experiment/scripts/slurm_log/test_t2v_360p_916_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/test_t2v_360p_916_%j.err

# Test T2V at 360p 9:16 - the EXACT official resolution
# This is (360, 640) which is perfectly divisible by 8

cd /scratch/wc3013/open-sora-v1.3-experiment

# Activate environment
source /scratch/wc3013/conda-envs/opensora13/bin/activate

# Set environment variables
export HF_HOME=/scratch/wc3013/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/wc3013/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Testing T2V at 360p 9:16 (OFFICIAL RESOLUTION) ==="
echo "Resolution: 360p with 9:16 aspect ratio"
echo "Dimensions: (360, 640) - BOTH divisible by 8"
echo ""

# Run T2V test
python naive_experiment/scripts/test_t2v_360p_916.py \
    --prompt "A person applying eye makeup in front of a mirror." \
    --output naive_experiment/results/test_t2v_360p_916.mp4

echo ""
echo "=== Test complete ==="
echo "Check naive_experiment/results/test_t2v_360p_916.mp4"
echo ""
echo "INTERPRETATION:"
echo "- If video looks good: 360p 9:16 works! We should use this aspect ratio."
echo "- If video has RGB blocks: There's a deeper issue with the model/environment."

