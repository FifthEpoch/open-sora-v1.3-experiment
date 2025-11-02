#!/bin/bash
#SBATCH -J build_kernels
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "Node: $(hostname)"
nvidia-smi || true

# --- env ---
# Setup scratch environment variables FIRST
# NOTE: Ensure this path exists before running
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_set_scratch_env.sh"

# Load modules
module purge
# (Optional) load a modern GCC if your default is very old:
# module load gcc/12
module load anaconda3/2024.02

# Activate conda environment in scratch location
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"

python -c "import torch;print('Torch:', torch.__version__,'CUDA:', torch.version.cuda)"

# Use system CUDA installation
if [ -d "/usr/local/cuda-12.2" ]; then
    export CUDA_HOME="/usr/local/cuda-12.2"
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME="/usr/local/cuda-12"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
else
    echo "ERROR: Could not find system CUDA installation"
    exit 1
fi

echo "Using system CUDA: $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version

# Hopper (H100/H200) arch; include others if you may run on A100 too:
export TORCH_CUDA_ARCH_LIST="90"     # or "80;90" if you also use A100 nodes
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

# Explicitly disable pip cache to avoid cross-device link errors
export PIP_NO_CACHE_DIR=1

# Verify environment variables
echo "Build environment:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  TMPDIR: $TMPDIR"
echo "  PIP_NO_CACHE_DIR: $PIP_NO_CACHE_DIR"

# Build wheels in scratch directory
WHEEL_DIR="${SCRATCH_BASE}/wheels/cu121_sm90"
mkdir -p "${WHEEL_DIR}"

# ---- FLASH-ATTN ----
# Build from source to avoid cross-device link issues with pip cache
cd /scratch/wc3013/tmp
rm -rf flash-attention
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && git checkout v2.5.8

# Force all temporary directories to be on scratch filesystem
export TMPDIR=/scratch/wc3013/tmp
export TMP=/scratch/wc3013/tmp
export TEMP=/scratch/wc3013/tmp
unset PIP_NO_CACHE_DIR  # Use PIP_CACHE_DIR instead which is on scratch

pip wheel -v --no-build-isolation --no-deps -w "$WHEEL_DIR" .
cd "$SCRIPT_DIR"

# ---- APEX ----
# Build both C++ and CUDA extensions (fused layernorm, FusedAdam, etc.)
# NOTE: Apex build will likely fail due to CUDA 12.1/12.2 mismatch, but we build anyway
# If it fails, we simply skip it and use enable_layernorm_kernel=False in configs
# Force TMPDIR on scratch before building
export TMPDIR=/scratch/wc3013/tmp
export TMP=/scratch/wc3013/tmp
export TEMP=/scratch/wc3013/tmp

echo "Attempting to build apex (may fail due to CUDA version mismatch)..."
pip wheel -v --no-cache-dir --no-build-isolation --no-deps -w "$WHEEL_DIR" \
  "git+https://github.com/NVIDIA/apex.git#egg=apex" \
  --config-settings="--build-option=--cpp_ext" \
  --config-settings="--build-option=--cuda_ext" || echo "Apex build failed - this is OK, we'll use standard LayerNorm"

# Install the wheels we just produced (optional here; you can install later in your run env)
pip install "$WHEEL_DIR"/flash_attn-*.whl

# Install apex if it was built successfully
if ls "$WHEEL_DIR"/apex-*.whl 1> /dev/null 2>&1; then
    pip install "$WHEEL_DIR"/apex-*.whl
    echo "Apex installed successfully"
else
    echo "WARNING: Apex not available. Use enable_layernorm_kernel=False in configs"
fi

# Smoke tests
python - <<'PY'
import torch
print("CUDA visible:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0))
# Flash-Attn checks
try:
    import flash_attn
    from flash_attn.bert_padding import index_first_axis
    print("flash-attn import OK")
except Exception as e:
    print("flash-attn import FAILED:", e)
# Apex checks
try:
    import apex
    from apex.normalization import FusedLayerNorm
    from apex.optimizers import FusedAdam
    print("apex import OK")
except Exception as e:
    print("apex import FAILED:", e)
PY

echo "Built wheels are in $WHEEL_DIR"

# ---- OTHER OPEN-SORA v1.3 REQUIREMENTS ----
# NOTE: Update this path to your actual repo location
cd /path/to/your/repo/Open-Sora-1.3
pip install -v -e .
# (optional) TensorNVMe for fast disk I/O; fall back if io_uring not available
DISABLE_URING=1 pip install -v --no-build-isolation git+https://github.com/hpcaitech/TensorNVMe.git || true