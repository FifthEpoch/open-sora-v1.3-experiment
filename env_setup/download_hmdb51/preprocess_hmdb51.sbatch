#!/bin/bash
#SBATCH -J preprocess_hmdb51
#SBATCH -p cpu
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

echo "========================================"
echo "HMDB51 Preprocessing Job"
echo "========================================"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

# --- Environment Setup ---
# Auto-load scratch environment variables to avoid hitting /home quotas
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../00_set_scratch_env.sh" ]; then
    echo "Loading scratch environment configuration..."
    source "${SCRIPT_DIR}/../00_set_scratch_env.sh"
else
    echo "ERROR: ${SCRIPT_DIR}/../00_set_scratch_env.sh not found!"
    exit 1
fi

# Load conda module
module purge
module load anaconda3/2024.02

# Activate conda environment in scratch location
echo "Activating conda environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

# Verify environment
python --version
python -c "import av; import numpy; import pandas; print('✓ Required packages available')"

# Set OMP threads to match requested CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Run Preprocessing ---
echo "========================================"
echo "Starting HMDB51 preprocessing..."
echo "========================================"

cd "${SCRIPT_DIR}"

# Run with --skip-cleanup to avoid interactive prompt
python preprocess_hmdb51.py --skip-cleanup

EXIT_CODE=$?

echo "========================================"
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE

