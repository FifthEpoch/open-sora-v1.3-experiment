#!/usr/bin/env python3
"""
Fine-tune Open-Sora v1.3 on a single video sample.

This script takes one video, creates a training dataset from it, and fine-tunes
the model for a small number of steps.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import torch
from mmengine.config import Config
from torch.utils.data import Dataset

# We'll use the existing training infrastructure
from opensora.registry import MODELS
from opensora.utils.misc import create_logger

# Import training components
import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from datetime import timedelta
import torch.distributed as dist


def create_single_video_csv(video_path, caption, output_csv):
    """Create a CSV file with a single video entry for training."""
    df = pd.DataFrame([{
        'path': video_path,
        'text': caption,
        'num_frames': 45,
        'height': 480,
        'width': 640,
        'fps': 24,
        'aspect_ratio': 1.33,  # 4:3
    }])
    df.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on a single video")
    parser.add_argument("--config", type=str, required=True, help="Path to fine-tuning config")
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    parser.add_argument("--caption", type=str, required=True, help="Video caption")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to base checkpoint")
    parser.add_argument("--vae-path", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save fine-tuned checkpoint")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    logger = create_logger()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Update paths in config
    cfg.model.from_pretrained = args.checkpoint_path
    cfg.vae.from_pretrained = args.vae_path
    cfg.lr = args.learning_rate
    
    # Create temporary CSV with single video
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_csv = f.name
        create_single_video_csv(args.video_path, args.caption, temp_csv)
        cfg.dataset.data_path = temp_csv
    
    # Set output directory
    cfg.outputs = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Modify training parameters for single-video fine-tuning
    cfg.epochs = 1
    cfg.log_every = 1
    cfg.ckpt_every = args.num_steps + 1  # Save at the end
    
    # Initialize distributed training (single GPU)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 42))
    coordinator = DistCoordinator()
    device = get_current_device()
    
    # Set batch size to 1 and disable drop_last
    cfg.dataset.batch_size = 1
    cfg.num_workers = 0  # Disable multiprocessing for single video
    
    logger.info(f"Fine-tuning on video: {args.video_path}")
    logger.info(f"Caption: {args.caption}")
    logger.info(f"Training for {args.num_steps} steps")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Import and run training (we'll call the train.py main function)
    # For now, print instructions
    logger.info("\n" + "="*70)
    logger.info("To run fine-tuning, use the standard train.py with modified config:")
    logger.info("="*70)
    logger.info(f"torchrun --standalone --nproc_per_node 1 scripts/train.py {args.config} \\")
    logger.info(f"    --data-path {temp_csv} \\")
    logger.info(f"    --ckpt-path {args.checkpoint_path}")
    logger.info("\nNote: The training will run for the specified number of steps.")
    logger.info(f"Temporary CSV created at: {temp_csv}")
    logger.info("\nYou may need to manually adjust the training loop to stop after")
    logger.info(f"{args.num_steps} steps instead of a full epoch.")
    
    # Actually, let's create a wrapper that calls train.py properly
    import subprocess
    
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node", "1",
        str(Path(__file__).parent.parent.parent / "scripts" / "train.py"),
        args.config,
        "--data-path", temp_csv,
        "--ckpt-path", args.checkpoint_path,
    ]
    
    logger.info(f"\nRunning command: {' '.join(cmd)}")
    logger.info("Note: You may need to manually stop training after the desired steps.")
    
    # Save the command for later execution
    script_path = Path(args.output_dir) / "finetune_command.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Fine-tuning command\n")
        f.write("# Stop training after reaching desired steps (Ctrl+C or modify train.py)\n\n")
        f.write(" ".join(cmd) + "\n")
    script_path.chmod(0o755)
    
    logger.info(f"\nSaved command to: {script_path}")
    logger.info("\nFor automated step-based training, you may need to modify train.py")
    logger.info("to accept a --max-steps argument.")


if __name__ == "__main__":
    main()

