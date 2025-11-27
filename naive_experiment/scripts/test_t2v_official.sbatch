#!/bin/bash
#SBATCH --job-name=test_t2v_official
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --output=naive_experiment/scripts/results/debug/test_t2v_official_%j.out
#SBATCH --error=naive_experiment/scripts/results/debug/test_t2v_official_%j.err

set -euo pipefail

echo "=================================================="
echo "Text-to-Video Generation Test (Official Settings)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# Setup environment
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"

cd "$PROJECT_ROOT"

# Load environment
module purge
module load anaconda3/2025.06
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

# Set library path to prioritize conda environment
export LD_LIBRARY_PATH="${SCRATCH_BASE}/conda-envs/opensora13/lib:${LD_LIBRARY_PATH:-}"

# Check optimizations availability
echo "=================================================="
echo "CHECKING OPTIMIZATION AVAILABILITY"
echo "=================================================="
python naive_experiment/scripts/check_optimizations.py
echo ""

# Decide which config to use based on what's available
echo "=================================================="
echo "DECIDING CONFIGURATION"
echo "=================================================="

if python -c "import flash_attn, apex; from apex.normalization import FusedLayerNorm" 2>/dev/null; then
    CONFIG="naive_experiment/configs/t2v_generation_official.py"
    echo "✓ Using OFFICIAL config (flash-attn + apex available)"
elif python -c "import flash_attn" 2>/dev/null; then
    CONFIG="naive_experiment/configs/t2v_generation_noopt.py"
    echo "⚠ Using NO-OPT config (only flash-attn available, but safer to disable both)"
else
    CONFIG="naive_experiment/configs/t2v_generation_noopt.py"
    echo "⚠ Using NO-OPT config (optimizations not available)"
fi

echo "Config file: $CONFIG"
echo ""

# Create output directory
mkdir -p naive_experiment/results/debug/t2v_official

echo "=================================================="
echo "GENERATING VIDEOS"
echo "=================================================="
echo ""

# Test with 3 prompts
PROMPTS=(
    "A person applying eye makeup"
    "A person playing basketball, dribbling and shooting a ball"
    "A person cycling on a bicycle down a road"
)

NAMES=(
    "apply_eye_makeup"
    "basketball"
    "cycling"
)

for i in ${!PROMPTS[@]}; do
    PROMPT="${PROMPTS[$i]}"
    NAME="${NAMES[$i]}"
    echo "$(($i + 1))/${#PROMPTS[@]}: Generating '${PROMPT}'..."
    
    python naive_experiment/scripts/test_t2v_generation.py \
        --config "$CONFIG" \
        --prompt "$PROMPT" \
        --output-path "naive_experiment/results/debug/t2v_official/${NAME}.mp4" \
        --seed $((42 + $i))
    
    echo ""
    echo "---"
    echo ""
done

echo "=================================================="
echo "GENERATION COMPLETE"
echo "=================================================="
echo "Videos saved to: naive_experiment/results/debug/t2v_official/"
echo "End time: $(date)"
echo "=================================================="

