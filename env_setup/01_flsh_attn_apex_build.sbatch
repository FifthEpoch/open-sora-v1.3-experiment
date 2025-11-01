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
module purge
# (Optional) load a modern GCC if your default is very old:
# module load gcc/12

source ~/.bashrc

# Setup scratch environment variables
# NOTE: Ensure this path exists before running
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/05_setup_scratch_env.sh"

conda activate opensora13

python -c "import torch;print('Torch:', torch.__version__,'CUDA:', torch.version.cuda)"

# Bring an NVCC that matches our cu12 runtime (no reliance on system CUDA-13)
# This drops nvcc into site-packages under nvidia/cuda_nvcc.
pip install -U "nvidia-cuda-nvcc-cu12==12.1.105"

# Point CUDA_HOME & PATH/LD_LIBRARY_PATH to that toolkit
export CUDA_HOME="$(python - <<'PY'
import os, nvidia.cuda_nvcc as m
from pathlib import Path
p = Path(m.__file__).resolve().parent  # .../site-packages/nvidia/cuda_nvcc
print(str(p.parent))                   # .../site-packages/nvidia
PY
)"
export PATH="$CUDA_HOME/cuda_nvcc/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/cuda_nvcc/lib64:${LD_LIBRARY_PATH:-}"

# Hopper (H100/H200) arch; include others if you may run on A100 too:
export TORCH_CUDA_ARCH_LIST="90"     # or "80;90" if you also use A100 nodes
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

# Build wheels in scratch directory
mkdir -p "${SCRATCH_DIR}/wheels/cu121_sm90"
WHEEL_DIR="${SCRATCH_DIR}/wheels/cu121_sm90"

# ---- FLASH-ATTN ----
# Version family known to work with torch 2.2.x + cu12.x on Hopper.
# --no-build-isolation ensures it sees the same torch/py build env.
pip wheel -v --no-build-isolation --no-deps -w "$WHEEL_DIR" \
  "flash-attn==2.5.8"

# (Alternative if you prefer source pin)
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention && git checkout v2.5.8 && pip wheel -v --no-build-isolation --no-deps -w "$WHEEL_DIR" .

# ---- APEX ----
# Build both C++ and CUDA extensions (fused layernorm, FusedAdam, etc.)
pip wheel -v --no-build-isolation --no-deps -w "$WHEEL_DIR" \
  "git+https://github.com/NVIDIA/apex.git#egg=apex" \
  --config-settings="--build-option=--cpp_ext" \
  --config-settings="--build-option=--cuda_ext"

# Install the wheels we just produced (optional here; you can install later in your run env)
pip install "$WHEEL_DIR"/flash_attn-*.whl
pip install "$WHEEL_DIR"/apex-*.whl

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