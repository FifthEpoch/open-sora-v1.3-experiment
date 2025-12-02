#!/bin/bash
#SBATCH --job-name=test_t2v_360p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --output=naive_experiment/scripts/slurm_log/test_t2v_360p_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/test_t2v_360p_%j.err

# Test T2V at 360p_16d to verify the model works at this resolution
# If T2V works but V2V doesn't, the issue is in conditioning handling

cd /scratch/wc3013/open-sora-v1.3-experiment

# Activate environment
source /scratch/wc3013/conda-envs/opensora13/bin/activate

# Set environment variables
export HF_HOME=/scratch/wc3013/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/wc3013/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Testing T2V at 360p_16d ==="
echo "This will verify if the model/VAE works at 360p_16d resolution"
echo ""

# Run T2V test
python naive_experiment/scripts/test_t2v_360p.py \
    --prompt "A person applying eye makeup in front of a mirror." \
    --output naive_experiment/results/test_t2v_360p.mp4 \
    --resolution 360p_16d \
    --aspect-ratio "3:4"

echo ""
echo "=== Test complete ==="
echo "Check naive_experiment/results/test_t2v_360p.mp4"
echo ""
echo "INTERPRETATION:"
echo "- If video looks good: Issue is in V2V conditioning, not model/VAE"
echo "- If video has RGB blocks: Issue is in model/VAE at 360p_16d"

