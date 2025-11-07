#!/bin/bash
# Fix bitsandbytes compatibility issue with PyTorch 2.2.2
# Run this once on the cluster to upgrade bitsandbytes in your conda environment

set -euo pipefail

echo "================================================"
echo "Fixing bitsandbytes compatibility issue"
echo "================================================"

# Set scratch base
SCRATCH_BASE="/scratch/wc3013"

# Load conda
module purge
module load anaconda3/2024.02
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Configure conda paths
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --prepend pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# Activate environment
echo "Activating opensora13 environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

# Check current version
echo ""
echo "Current bitsandbytes version:"
pip show bitsandbytes | grep Version || echo "Not installed"

# Upgrade bitsandbytes
echo ""
echo "Upgrading bitsandbytes to version compatible with PyTorch 2.2.2..."
pip install --upgrade 'bitsandbytes>=0.43.0' --no-cache-dir

# Verify new version
echo ""
echo "New bitsandbytes version:"
pip show bitsandbytes | grep Version

# Test import
echo ""
echo "Testing bitsandbytes import..."
python -c "import bitsandbytes; print('✓ bitsandbytes import successful')" || echo "✗ Import failed"

echo ""
echo "================================================"
echo "Fix complete!"
echo "================================================"
echo ""
echo "You can now run your experiment with:"
echo "  cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts"
echo "  sbatch run_experiment.sbatch"

