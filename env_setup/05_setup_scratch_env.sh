#!/bin/bash
# Environment Variables Setup for Cluster Scratch Storage
# This script redirects ALL installs, caches, and temp files to /scratch
# Run this BEFORE any installation or before activating your conda environment

# Base scratch directory - all Open-Sora installations go here
export SCRATCH_DIR="/scratch/wc3013/opensora_env"

# Create organized directory structure
mkdir -p "${SCRATCH_DIR}"/{conda_envs,conda_pkgs,pip_cache,torch_cache,transformers_cache,hf_cache,triton_cache,extensions_cache,inductor_cache,tmp,xdg_cache,wandb_cache,tensorboard}

# ===========================================================================
# PYTHON / CONDA / PIP CACHES
# ===========================================================================
export PIP_CACHE_DIR="${SCRATCH_DIR}/pip_cache"
export CONDA_ENVS_DIRS="${SCRATCH_DIR}/conda_envs"
export CONDA_PKGS_DIRS="${SCRATCH_DIR}/conda_pkgs"

# ===========================================================================
# PYTORCH CACHES
# ===========================================================================
export TORCH_HOME="${SCRATCH_DIR}/torch_cache"
export TORCH_EXTENSIONS_DIR="${SCRATCH_DIR}/extensions_cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/inductor_cache"
export TMPDIR="${SCRATCH_DIR}/tmp"

# ===========================================================================
# TRITON CACHE
# ===========================================================================
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton_cache"

# ===========================================================================
# COLOSSALAI CACHES
# ===========================================================================
export COLOSSALAI_EXTENSIONS_JIT_CACHE_PATH="${SCRATCH_DIR}/extensions_cache/colossalai"

# ===========================================================================
# HUGGING FACE CACHES
# ===========================================================================
export HF_HOME="${SCRATCH_DIR}/hf_cache/hf_home"
export HF_HUB_CACHE="${SCRATCH_DIR}/hf_cache/hub"
export HF_DATASETS_CACHE="${SCRATCH_DIR}/hf_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/transformers_cache"

# ===========================================================================
# XDG CACHE (general system cache)
# ===========================================================================
export XDG_CACHE_HOME="${SCRATCH_DIR}/xdg_cache"

# ===========================================================================
# ADDITIONAL CACHES (to be thorough)
# ===========================================================================
export WANDB_CACHE_DIR="${SCRATCH_DIR}/wandb_cache"
export TENSORBOARD_LOGDIR="${SCRATCH_DIR}/tensorboard"

# Create any missing cache directories
mkdir -p "${TORCH_EXTENSIONS_DIR}"
mkdir -p "${COLOSSALAI_EXTENSIONS_JIT_CACHE_PATH}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${TENSORBOARD_LOGDIR}"

# Configure conda to use scratch directories (if conda is available)
if command -v conda &> /dev/null; then
    conda config --add envs_dirs "${SCRATCH_DIR}/conda_envs" 2>/dev/null || true
    conda config --add pkgs_dirs "${SCRATCH_DIR}/conda_pkgs" 2>/dev/null || true
fi

echo "=================================================="
echo "Environment variables configured for /scratch"
echo "=================================================="
echo "Base directory: ${SCRATCH_DIR}"
echo "Conda environments: ${CONDA_ENVS_DIRS}"
echo "Conda packages: ${CONDA_PKGS_DIRS}"
echo "Torch cache: ${TORCH_HOME}"
echo "Transformers cache: ${TRANSFORMERS_CACHE}"
echo "HuggingFace cache: ${HF_HOME}"
echo "Pip cache: ${PIP_CACHE_DIR}"
echo "=================================================="

