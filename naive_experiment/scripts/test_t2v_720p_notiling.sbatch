#!/bin/bash
#SBATCH --job-name=test_t2v_720p_notiling
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_720p_notiling_%j.out
#SBATCH --error=/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_720p_notiling_%j.err

set -euo pipefail

echo "=================================================="
echo "Text-to-Video Generation Test (720p NO TILING)"
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
CONFIG="naive_experiment/configs/t2v_generation_720p_notiling.py"
echo "Using 720p NO TILING configuration"
echo "Config file: $CONFIG"
echo "Resolution: 720p (960x1280)"
echo "VAE Tiling: DISABLED - should eliminate rectangular artifacts"
echo "Memory: 384GB (high memory needed for no-tiling VAE)"
echo "Expected: Artifact-free, highest quality output"
echo ""

# Create output directory
OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/debug/t2v_720p_notiling"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "GENERATING VIDEOS AT 720P (NO TILING)"
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
    echo "NOTE: VAE decoding will be slower but artifact-free"
    
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
echo "Resolution: 720p (960x1280)"
echo "VAE Tiling: DISABLED"
echo "Quality: Should be artifact-free!"
echo "End time: $(date)"
echo "=================================================="

