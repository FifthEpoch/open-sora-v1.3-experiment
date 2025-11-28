#!/bin/bash
#SBATCH --job-name=test_t2v_quality_sweep
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_quality_sweep_%j.out
#SBATCH --error=/scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/debug/test_t2v_quality_sweep_%j.err

set -euo pipefail

echo "=================================================="
echo "Text-to-Video Quality Sweep Test"
echo "Testing: tile_size=16 AND quality-focused parameters"
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

# Single test prompt for quick comparison
PROMPT="A person applying eye makeup"
SEED=42

echo "=================================================="
echo "TEST 1: tile_size=16 (Aggressive Tiling)"
echo "=================================================="
echo "Config: t2v_generation_720p_tile16.py"
echo "Expected: Minimal tiling artifacts, fastest"
echo ""

mkdir -p "${PROJECT_ROOT}/naive_experiment/results/debug/t2v_tile16"
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_generation_720p_tile16.py \
    --prompt "$PROMPT" \
    --output-path "${PROJECT_ROOT}/naive_experiment/results/debug/t2v_tile16/apply_eye_makeup.mp4" \
    --seed $SEED

echo ""
echo "---"
echo ""

echo "=================================================="
echo "TEST 2: Quality-Focused Parameters"
echo "=================================================="
echo "Config: t2v_generation_720p_quality.py"
echo "Changes: num_sampling_steps=50, cfg_scale=9.0, aes=7.0, flow=5.0"
echo "Expected: Better structural coherence and prompt adherence"
echo ""

mkdir -p "${PROJECT_ROOT}/naive_experiment/results/debug/t2v_quality"
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_generation_720p_quality.py \
    --prompt "$PROMPT" \
    --output-path "${PROJECT_ROOT}/naive_experiment/results/debug/t2v_quality/apply_eye_makeup.mp4" \
    --seed $SEED

echo ""
echo "---"
echo ""

echo "=================================================="
echo "SWEEP COMPLETE"
echo "=================================================="
echo ""
echo "Compare these outputs:"
echo "1. tile_size=16:     ${PROJECT_ROOT}/naive_experiment/results/debug/t2v_tile16/apply_eye_makeup.mp4"
echo "2. Quality params:   ${PROJECT_ROOT}/naive_experiment/results/debug/t2v_quality/apply_eye_makeup.mp4"
echo ""
echo "Previous tests for reference:"
echo "- tile_size=4:  naive_experiment/results/debug/t2v_720p/apply_eye_makeup.mp4"
echo "- tile_size=8:  naive_experiment/results/debug/t2v_720p_largetile/apply_eye_makeup.mp4"
echo ""
echo "If tiling changes don't matter → tiling is not the quality bottleneck"
echo "If quality params help → focus on scheduler/conditioning improvements"
echo ""
echo "End time: $(date)"
echo "=================================================="

