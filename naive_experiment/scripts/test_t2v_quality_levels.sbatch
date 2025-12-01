#!/bin/bash
#SBATCH --job-name=test_t2v_quality_levels
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=06:00:00

set -euo pipefail

# ============================================================
# Text-to-Video Quality Levels Test
# Tests 3 quality configurations: Balanced, Enhanced, Ultra
# ============================================================

PROJECT_ROOT="/scratch/wc3013/open-sora-v1.3-experiment"
cd "$PROJECT_ROOT"

# Setup logging
LOG_DIR="${PROJECT_ROOT}/naive_experiment/scripts/results/debug"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Text-to-Video Quality Levels Test"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "============================================================"
echo ""

# Load environment
echo "Loading modules..."
module purge
module load anaconda3/2025.06

echo "Sourcing conda.sh..."
set +u
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate "/scratch/wc3013/conda-envs/opensora13"
set -u

echo "✓ Conda environment activated"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Test prompt (same as quality sweep)
PROMPT="A person applying eye makeup"
SEED=42

# ============================================================
# TEST 1: Balanced (Fast, Good Quality)
# ============================================================
echo "============================================================"
echo "TEST 1: BALANCED Configuration"
echo "============================================================"
echo "Config: t2v_720p_balanced.py"
echo "Parameters:"
echo "  - num_sampling_steps: 40"
echo "  - cfg_scale: 8.5"
echo "  - aes: 6.8"
echo "  - flow: 5.5"
echo "Expected: Good quality, reasonable speed (~15-20 min)"
echo ""

OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/debug/t2v_balanced"
mkdir -p "$OUTPUT_DIR"

START_TIME=$(date +%s)
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_720p_balanced.py \
    --prompt "$PROMPT" \
    --output-path "${OUTPUT_DIR}/apply_eye_makeup.mp4" \
    --seed $SEED
END_TIME=$(date +%s)
BALANCED_TIME=$((END_TIME - START_TIME))

echo "✓ Balanced config completed in ${BALANCED_TIME}s (~$((BALANCED_TIME / 60)) minutes)"
echo ""
echo "---"
echo ""

# ============================================================
# TEST 2: Enhanced (Better Quality)
# ============================================================
echo "============================================================"
echo "TEST 2: ENHANCED Configuration"
echo "============================================================"
echo "Config: t2v_720p_enhanced.py"
echo "Parameters:"
echo "  - num_sampling_steps: 60"
echo "  - cfg_scale: 10.0"
echo "  - aes: 7.0"
echo "  - flow: 6.0"
echo "Expected: Better quality, slower (~25-35 min)"
echo ""

OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/debug/t2v_enhanced"
mkdir -p "$OUTPUT_DIR"

START_TIME=$(date +%s)
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_720p_enhanced.py \
    --prompt "$PROMPT" \
    --output-path "${OUTPUT_DIR}/apply_eye_makeup.mp4" \
    --seed $SEED
END_TIME=$(date +%s)
ENHANCED_TIME=$((END_TIME - START_TIME))

echo "✓ Enhanced config completed in ${ENHANCED_TIME}s (~$((ENHANCED_TIME / 60)) minutes)"
echo ""
echo "---"
echo ""

# ============================================================
# TEST 3: Ultra (Maximum Quality)
# ============================================================
echo "============================================================"
echo "TEST 3: ULTRA Configuration"
echo "============================================================"
echo "Config: t2v_720p_ultra.py"
echo "Parameters:"
echo "  - num_sampling_steps: 100"
echo "  - cfg_scale: 12.0"
echo "  - aes: 7.5"
echo "  - flow: 7.0"
echo "Expected: Best quality, slowest (~45-60 min)"
echo ""

OUTPUT_DIR="${PROJECT_ROOT}/naive_experiment/results/debug/t2v_ultra"
mkdir -p "$OUTPUT_DIR"

START_TIME=$(date +%s)
python naive_experiment/scripts/test_t2v_generation.py \
    --config naive_experiment/configs/t2v_720p_ultra.py \
    --prompt "$PROMPT" \
    --output-path "${OUTPUT_DIR}/apply_eye_makeup.mp4" \
    --seed $SEED
END_TIME=$(date +%s)
ULTRA_TIME=$((END_TIME - START_TIME))

echo "✓ Ultra config completed in ${ULTRA_TIME}s (~$((ULTRA_TIME / 60)) minutes)"
echo ""
echo "---"
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================"
echo "QUALITY LEVELS TEST COMPLETE"
echo "============================================================"
echo ""
echo "Generated videos:"
echo "1. Balanced:  ${PROJECT_ROOT}/naive_experiment/results/debug/t2v_balanced/apply_eye_makeup.mp4"
echo "2. Enhanced:  ${PROJECT_ROOT}/naive_experiment/results/debug/t2v_enhanced/apply_eye_makeup.mp4"
echo "3. Ultra:     ${PROJECT_ROOT}/naive_experiment/results/debug/t2v_ultra/apply_eye_makeup.mp4"
echo ""
echo "Generation times:"
echo "  Balanced: ${BALANCED_TIME}s (~$((BALANCED_TIME / 60)) min)"
echo "  Enhanced: ${ENHANCED_TIME}s (~$((ENHANCED_TIME / 60)) min)"
echo "  Ultra:    ${ULTRA_TIME}s (~$((ULTRA_TIME / 60)) min)"
echo ""
echo "Next steps:"
echo "  1. Download and compare videos side-by-side"
echo "  2. Assess quality vs. speed trade-off"
echo "  3. Choose best config for full experiment"
echo ""
echo "End time: $(date)"
echo "============================================================"

