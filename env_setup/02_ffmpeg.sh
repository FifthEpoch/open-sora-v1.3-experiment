#!/bin/bash
# NOTE: This script should be run after sourcing 05_setup_scratch_env.sh
# If you haven't sourced it, do this: source env_setup/05_setup_scratch_env.sh

module purge || true
conda activate opensora13
conda install -y -c conda-forge ffmpeg>=6,<7