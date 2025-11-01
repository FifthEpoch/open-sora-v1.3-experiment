#!/bin/bash

set -euo pipefail

# Auto-load scratch environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/00_set_scratch_env.sh" ]; then
    echo "Loading scratch environment configuration..."
    source "${SCRIPT_DIR}/00_set_scratch_env.sh"
else
    echo "Warning: ${SCRIPT_DIR}/00_set_scratch_env.sh not found!"
    exit 1
fi

module purge || true
conda activate "${SCRATCH_BASE}/conda-envs/opensora13"
conda install -y -c conda-forge ffmpeg>=6,<7