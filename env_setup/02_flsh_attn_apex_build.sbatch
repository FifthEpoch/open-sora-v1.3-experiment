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

# Bring an NVCC that matches our cu12 runtime (no reliance on system CUDA-13)
# This drops nvcc into site-packages under nvidia/cuda_nvcc.
pip install -U "nvidia-cuda-nvcc-cu12==12.1.105"

# Point CUDA_HOME & PATH/LD_LIBRARY_PATH to that toolkit
# Debug: Print the actual structure
python - <<'PY'
import nvidia.cuda_nvcc as m
from pathlib import Path
p = Path(m.__file__).resolve()
print(f"Module file: {p}")
print(f"Module dir: {p.parent}")
print(f"Parent: {p.parent.parent}")
print(f"\nContents of parent directory:")
for item in p.parent.parent.iterdir():
    print(f"  {item.name}")
PY

export CUDA_HOME="$(python - <<'PY'
import os, nvidia.cuda_nvcc as m
from pathlib import Path
p = Path(m.__file__).resolve().parent  # .../site-packages/nvidia/cuda_nvcc
print(str(p.parent))                   # .../site-packages/nvidia
PY
)"
echo "CUDA_HOME (parent) set to: $CUDA_HOME"

# Check if nvcc is directly in the module dir or in a subdirectory
if [ -f "$CUDA_HOME/cuda_nvcc/bin/nvcc" ]; then
    export CUDA_HOME="$CUDA_HOME/cuda_nvcc"
    echo "Using nvcc from: $CUDA_HOME/cuda_nvcc/bin/nvcc"
elif [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "Using nvcc from: $CUDA_HOME/bin/nvcc"
elif [ -f "$(python -c 'import nvidia.cuda_nvcc as m; from pathlib import Path; print(Path(m.__file__).resolve().parent / "bin" / "nvcc")')" ]; then
    export CUDA_HOME="$(python -c 'import nvidia.cuda_nvcc as m; from pathlib import Path; print(Path(m.__file__).resolve().parent)')"
    echo "Using nvcc from module dir: $CUDA_HOME/bin/nvcc"
else
    echo "ERROR: Could not find nvcc binary!"
    echo "Searching for nvcc recursively..."
    python - <<'PY'
import nvidia.cuda_nvcc as m
from pathlib import Path
p = Path(m.__file__).resolve().parent
print(f"Searching from: {p}")
print(f"Searching for nvcc in all subdirectories...")
try:
    for item in p.rglob("nvcc"):
        if item.is_file():
            print(f"Found nvcc: {item}")
            print(f"  Parent dir: {item.parent}")
            print(f"  Grandparent dir: {item.parent.parent}")
    # If nothing found, list the directory structure
    print("\nDirectory structure:")
    for item in sorted(p.rglob("*")):
        if item.is_dir():
            print(f"  DIR:  {item}")
        elif item.is_file():
            print(f"  FILE: {item}")
except Exception as e:
    print(f"Error during search: {e}")
PY
    exit 1
fi

echo "CUDA_HOME set to: $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Verify nvcc is accessible
echo "Checking nvcc installation:"
ls -la "$CUDA_HOME/bin/nvcc" || echo "WARNING: nvcc not found at $CUDA_HOME/bin/nvcc"
nvcc --version || echo "WARNING: nvcc not in PATH"

# Hopper (H100/H200) arch; include others if you may run on A100 too:
export TORCH_CUDA_ARCH_LIST="90"     # or "80;90" if you also use A100 nodes
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

# Build wheels in scratch directory
WHEEL_DIR="${SCRATCH_BASE}/wheels/cu121_sm90"
mkdir -p "${WHEEL_DIR}"

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