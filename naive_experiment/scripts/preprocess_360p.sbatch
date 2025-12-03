#!/bin/bash
#SBATCH --job-name=preprocess_360p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=naive_experiment/scripts/slurm_log/preprocess_360p_%j.out
#SBATCH --error=naive_experiment/scripts/slurm_log/preprocess_360p_%j.err

# ===================================================================
# Preprocess UCF-101 videos at 360p 9:16 (360×640)
# Run this BEFORE run_experiment.sbatch to ensure correct resolution
# ===================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="/scratch/wc3013/open-sora-v1.3-experiment"
SCRATCH_BASE="/scratch/wc3013"

cd "${PROJECT_ROOT}"

echo "========================================"
echo "Preprocess UCF-101 at 360p"
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
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python --version

# --- Preprocessing ---
DATA_DIR="${PROJECT_ROOT}/env_setup/download_ucf101"
PROCESSED_DIR="${DATA_DIR}/ucf101_processed"
METADATA_CSV="${DATA_DIR}/ucf101_metadata.csv"

echo ""
echo "========================================"
echo "Step 1: Clean up old processed data"
echo "========================================"

if [ -d "${PROCESSED_DIR}" ]; then
    echo "Removing old processed directory: ${PROCESSED_DIR}"
    rm -rf "${PROCESSED_DIR}"
    echo "✓ Removed"
else
    echo "No existing processed directory found"
fi

if [ -f "${METADATA_CSV}" ]; then
    echo "Removing old metadata CSV: ${METADATA_CSV}"
    rm -f "${METADATA_CSV}"
    echo "✓ Removed"
else
    echo "No existing metadata CSV found"
fi

echo ""
echo "========================================"
echo "Step 2: Verify original UCF-101 data exists"
echo "========================================"

ORIGINAL_DIR="${DATA_DIR}/ucf101_org"
if [ -d "${ORIGINAL_DIR}" ]; then
    NUM_ORIGINAL=$(find "${ORIGINAL_DIR}" -name "*.avi" | wc -l)
    echo "✓ Found ${NUM_ORIGINAL} original .avi files in ${ORIGINAL_DIR}"
else
    echo "ERROR: Original UCF-101 data not found at ${ORIGINAL_DIR}" >&2
    echo "Please download UCF-101 first using download_ucf101.py" >&2
    exit 1
fi

echo ""
echo "========================================"
echo "Step 3: Preprocess videos at 360p 9:16"
echo "========================================"
echo "Target resolution: 360×640 (360p 9:16)"
echo "Target frames: 49"
echo "Conditioning frames: 22"
echo ""

cd "${DATA_DIR}"

python preprocess_ucf101.py \
    --input-dir ucf101_org \
    --output-dir ucf101_processed \
    --height 360 \
    --width 640 \
    --frames 49 \
    --conditioning-frames 22 \
    --skip-cleanup

echo ""
echo "========================================"
echo "Step 4: Verify preprocessing results"
echo "========================================"

if [ -f "${METADATA_CSV}" ]; then
    echo "✓ Metadata CSV created: ${METADATA_CSV}"
    
    # Count videos
    NUM_PROCESSED=$(wc -l < "${METADATA_CSV}")
    NUM_PROCESSED=$((NUM_PROCESSED - 1))  # Subtract header
    echo "✓ Processed ${NUM_PROCESSED} videos"
    
    # Show sample
    echo ""
    echo "Sample metadata (first 3 rows):"
    head -4 "${METADATA_CSV}"
    
    # Verify dimensions
    echo ""
    echo "Verifying dimensions..."
    SAMPLE_HEIGHT=$(head -2 "${METADATA_CSV}" | tail -1 | cut -d',' -f4)
    SAMPLE_WIDTH=$(head -2 "${METADATA_CSV}" | tail -1 | cut -d',' -f5)
    echo "Sample video dimensions: height=${SAMPLE_HEIGHT}, width=${SAMPLE_WIDTH}"
    
    if [ "${SAMPLE_HEIGHT}" = "360" ] && [ "${SAMPLE_WIDTH}" = "640" ]; then
        echo "✓ Dimensions are correct (360×640)"
    else
        echo "WARNING: Unexpected dimensions! Expected 360×640"
    fi
    
    # Verify with ffprobe on first video
    echo ""
    echo "Verifying first video with ffprobe..."
    FIRST_VIDEO=$(head -2 "${METADATA_CSV}" | tail -1 | cut -d',' -f1)
    if [ -f "${FIRST_VIDEO}" ]; then
        ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "${FIRST_VIDEO}"
    else
        # Try relative path
        FIRST_VIDEO="${DATA_DIR}/${FIRST_VIDEO}"
        if [ -f "${FIRST_VIDEO}" ]; then
            ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "${FIRST_VIDEO}"
        else
            echo "Could not find video file for verification"
        fi
    fi
else
    echo "ERROR: Metadata CSV was not created!" >&2
    exit 1
fi

echo ""
echo "========================================"
echo "Preprocessing complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Next step: Run the experiment with:"
echo "  sbatch naive_experiment/scripts/run_experiment.sbatch"

