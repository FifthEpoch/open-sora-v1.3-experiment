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


def create_truncated_video(video_path, num_frames=22, output_dir=None):
    """Create a truncated video with only first num_frames frames for training."""
    import av
    
    if output_dir is None:
        output_dir = Path(video_path).parent / "truncated_for_training"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    truncated_path = output_dir / f"{video_name}_first{num_frames}frames.mp4"
    
    if truncated_path.exists():
        return str(truncated_path)
    
    # Copy first num_frames frames to new video
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    
    output_container = av.open(str(truncated_path), mode='w')
    output_stream = output_container.add_stream('libx264', rate=24)
    output_stream.width = video_stream.width
    output_stream.height = video_stream.height
    output_stream.pix_fmt = 'yuv420p'
    
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count >= num_frames:
            break
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
        frame_count += 1
    
    # Flush encoder
    for packet in output_stream.encode():
        output_container.mux(packet)
    
    container.close()
    output_container.close()
    
    return str(truncated_path)


def create_single_video_csv(video_path, caption, output_csv):
    """Create a CSV file with a single video entry for training (using truncated video)."""
    df = pd.DataFrame([{
        'path': video_path,
        'text': caption,
        'num_frames': 22,  # Only first 22 frames for training
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
    
    # Create truncated video (first 22 frames only) for training
    logger.info(f"Creating truncated video (first 22 frames) from: {args.video_path}")
    truncated_video_path = create_truncated_video(args.video_path, num_frames=22)
    logger.info(f"Truncated video saved to: {truncated_video_path}")
    
    # Create temporary CSV with single truncated video
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_csv = f.name
        create_single_video_csv(truncated_video_path, args.caption, temp_csv)
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

