#!/bin/bash
#SBATCH -J fix_bitsandbytes
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_fix_bitsandbytes.out
#SBATCH --error=slurm_fix_bitsandbytes.err

set -euo pipefail

echo "================================================"
echo "Fixing bitsandbytes compatibility issue"
echo "================================================"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================"

# Set scratch base
SCRATCH_BASE="/scratch/wc3013"

# Create necessary directories
mkdir -p "${SCRATCH_BASE}"/{conda-envs,conda-pkgs,py-cache/{pip,torch/extensions},tmp}

# Set environment variables
export CONDA_ENVS_DIRS="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/py-cache/pip"
export TMPDIR="${SCRATCH_BASE}/tmp"
export TMP="${SCRATCH_BASE}/tmp"
export TEMP="${SCRATCH_BASE}/tmp"

# Load conda module
module purge
module load anaconda3/2024.02

# Source conda initialization script
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh

# Deactivate any existing environment
conda deactivate 2>/dev/null || true

# Configure conda to recognize custom environment location
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --prepend pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# Activate environment
echo ""
echo "Activating opensora13 environment..."
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

# Verify GPU is available
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check current versions
echo ""
echo "================================================"
echo "Current package versions:"
echo "================================================"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || echo "NumPy not found"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch not found"
python -c "import xformers; print(f'xformers: {xformers.__version__}')" || echo "xformers not found"
pip show bitsandbytes | grep Version || echo "bitsandbytes not found"

# CRITICAL: Downgrade NumPy FIRST (PyTorch 2.2.2 compiled against NumPy 1.x)
echo ""
echo "================================================"
echo "Step 1: Downgrading NumPy to 1.x (compatible with PyTorch 2.2.2)..."
echo "================================================"
pip install 'numpy<2' --no-cache-dir --force-reinstall

# Reinstall PyTorch 2.2.2 (in case it was upgraded)
echo ""
echo "================================================"
echo "Step 2: Ensuring PyTorch 2.2.2 with CUDA 12.1..."
echo "================================================"
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# Install correct bitsandbytes version WITHOUT upgrading dependencies
echo ""
echo "================================================"
echo "Step 3: Installing bitsandbytes 0.43.3 (without upgrading deps)..."
echo "================================================"
pip install 'bitsandbytes==0.43.3' --no-cache-dir --force-reinstall --no-deps

# Verify new versions
echo ""
echo "================================================"
echo "New package versions:"
echo "================================================"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import xformers; print(f'xformers: {xformers.__version__}')"
pip show bitsandbytes | grep Version

# Test imports on GPU node
echo ""
echo "================================================"
echo "Testing package imports on GPU node..."
echo "================================================"
python -c "import numpy; print(f'✓ NumPy import successful')"
python -c "import cv2; print(f'✓ OpenCV import successful')"
python -c "import torch; print(f'✓ PyTorch import successful')"
python -c "import xformers; print(f'✓ xformers import successful')"
python -c "import bitsandbytes; print(f'✓ bitsandbytes import successful')"

# Test CUDA availability
echo ""
echo "Testing CUDA availability:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "================================================"
echo "Fix complete!"
echo "================================================"
echo "End time: $(date)"
echo ""
echo "You can now run your experiment with:"
echo "  cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts"
echo "  sbatch run_experiment.sbatch"

