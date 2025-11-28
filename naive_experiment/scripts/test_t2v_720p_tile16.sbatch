#!/bin/bash
#SBATCH --job-name=test_t2v_720p_tile16
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_720p_tile16_%j.out
#SBATCH --error=/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_720p_tile16_%j.err

set -euo pipefail

echo "=================================================="
echo "Text-to-Video Generation Test (720p tile_size=16)"
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
echo "Loading modules..."
module purge
module load anaconda3/2025.06

# Source conda properly
echo "Sourcing conda.sh..."
set +u
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"
set -u

echo "✓ Conda environment activated"

# Set library path to prioritize conda environment
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check optimizations availability
echo "=================================================="
echo "CHECKING OPTIMIZATION AVAILABILITY"
echo "=================================================="
python naive_experiment/scripts/check_optimizations.py
echo ""

echo "=================================================="
echo "CONFIGURATION"
echo "=================================================="
CONFIG="naive_experiment/configs/t2v_generation_720p_tile16.py"
echo "Using 720p with MAXIMUM tile_size=16"
echo "Config file: $CONFIG"
echo "Resolution: 720p"
echo "VAE Tiling: tile_size=16 (fewest tile boundaries, minimal artifacts)"
echo "Expected: Fastest generation with minimal tiling artifacts"
echo ""

# Create output directory
OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/debug/t2v_720p_tile16"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "GENERATING VIDEOS AT 720P (tile_size=16)"
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
        --output-path "${OUTPUT_DIR}/${NAME}.mp4" \
        --seed $((42 + $i))
    
    echo ""
    echo "---"
    echo ""
done

echo "=================================================="
echo "GENERATION COMPLETE"
echo "=================================================="
echo "Videos saved to: ${OUTPUT_DIR}/"
echo "Resolution: 720p"
echo "VAE tile_size: 16"
echo "End time: $(date)"
echo "=================================================="

