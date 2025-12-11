#!/usr/bin/env python3
"""
LoRA fine-tuning script for Open-Sora STDiT3 model.

This script performs test-time adaptation using LoRA, which trains <1% of
the model parameters while achieving similar results to full fine-tuning.

Usage:
    python lora_finetune.py \
        --config lora_experiment/configs/lora_finetune.py \
        --video-path path/to/video.mp4 \
        --caption "description of video" \
        --output-dir lora_experiment/results \
        --num-steps 20
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mmengine.config import Config
from tqdm import tqdm

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import load_checkpoint

# Import LoRA utilities
sys.path.insert(0, str(PROJECT_ROOT / "lora_experiment"))
from lora_layers import (
    inject_lora_into_stdit3,
    get_lora_parameters,
    count_lora_parameters,
    save_lora_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Open-Sora")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--video-path", type=str, required=True, help="Path to input video")
    parser.add_argument("--caption", type=str, required=True, help="Video caption")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank (overrides config)")
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha (overrides config)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_video_for_training(video_path: str, vae, num_frames: int, image_size: tuple, device: str, dtype):
    """Load and encode video for training."""
    import av
    import numpy as np
    from torchvision import transforms
    
    # Read video frames
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        if len(frames) >= num_frames:
            break
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
    container.close()
    
    if len(frames) < num_frames:
        # Pad with last frame if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    # Convert to tensor
    frames = np.stack(frames, axis=0)  # (T, H, W, C)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()  # (C, T, H, W)
    frames = frames / 255.0
    
    # Resize to target size
    H, W = image_size
    frames = F.interpolate(frames.unsqueeze(0), size=(frames.shape[1], H, W), mode="trilinear", align_corners=False)
    frames = frames.squeeze(0)
    
    # Normalize to [-1, 1]
    frames = frames * 2 - 1
    
    # Add batch dimension and encode with VAE
    frames = frames.unsqueeze(0).to(device, dtype)  # (1, C, T, H, W)
    
    with torch.no_grad():
        latents = vae.encode(frames)
    
    return latents


def train_lora(
    model,
    vae,
    text_encoder,
    scheduler,
    latents,
    caption,
    num_steps: int,
    lr: float,
    device: str,
    dtype,
    image_size: tuple,
    num_frames: int,
):
    """Train LoRA weights on a single video."""
    
    # Get LoRA parameters only
    lora_params = get_lora_parameters(model)
    if not lora_params:
        raise ValueError("No LoRA parameters found. Make sure LoRA was injected into the model.")
    
    # Create optimizer for LoRA params only
    optimizer = AdamW(lora_params, lr=lr, betas=(0.9, 0.999), eps=1e-15)
    
    # Encode text - need to tokenize first, then encode
    with torch.no_grad():
        model_args = {}
        # Tokenize the caption
        tokens = text_encoder.tokenize_fn(caption)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        # Encode to get embeddings
        encoded = text_encoder.encode(input_ids, attention_mask)
        model_args["y"] = encoded["y"]
        model_args["mask"] = encoded["mask"]
    
    # Get latent size
    B, C, T, H, W = latents.shape
    
    # Training loop
    model.train()
    losses = []
    
    # Get the inner scheduler for add_noise (RFLOW wraps RFlowScheduler)
    inner_scheduler = scheduler.scheduler if hasattr(scheduler, 'scheduler') else scheduler
    num_timesteps = scheduler.num_timesteps
    
    pbar = tqdm(range(num_steps), desc="LoRA Training")
    for step in pbar:
        optimizer.zero_grad()
        
        # Sample random timestep (0 to num_timesteps-1)
        t = torch.randint(0, num_timesteps, (B,), device=device)
        
        # Add noise using rectified flow interpolation
        noise = torch.randn_like(latents)
        noisy_latents = inner_scheduler.add_noise(latents, noise, t)
        
        # Get model prediction
        model_args["x"] = noisy_latents
        model_args["timestep"] = t
        model_args["height"] = torch.tensor([image_size[0]], device=device)
        model_args["width"] = torch.tensor([image_size[1]], device=device)
        model_args["fps"] = torch.tensor([24], device=device)
        
        # Forward pass - model outputs [velocity, sigma] concatenated
        pred = model(**model_args)
        velocity_pred = pred.chunk(2, dim=1)[0]  # Take velocity part
        
        # For rectified flow, target is velocity: v = x_start - noise
        target = latents - noise
        
        loss = F.mse_loss(velocity_pred.float(), target.float())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        
        # Optimizer step
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return losses


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override config with command line args
    if args.lr is not None:
        cfg.lr = args.lr
    if args.rank is not None:
        cfg.lora.rank = args.rank
    if args.alpha is not None:
        cfg.lora.alpha = args.alpha
    
    device = args.device
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16
    
    print(f"{'='*60}")
    print(f"LoRA Fine-tuning")
    print(f"{'='*60}")
    print(f"Video: {args.video_path}")
    print(f"Caption: {args.caption}")
    print(f"Steps: {args.num_steps}")
    print(f"Learning rate: {cfg.lr}")
    print(f"LoRA rank: {cfg.lora.rank}")
    print(f"LoRA alpha: {cfg.lora.alpha}")
    print(f"{'='*60}")
    
    # Build model components
    print("\nLoading model components...")
    model_load_start = time.time()
    
    # Text encoder
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    
    # VAE
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    # Get sizes
    image_size = get_image_size(cfg.get("resolution", "360p"), cfg.get("aspect_ratio", "9:16"))
    num_frames = get_num_frames(cfg.get("num_frames", 49))
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    
    # Build STDiT3 model (don't move to device yet - inject LoRA first)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=False,
    )
    
    # Inject LoRA (before moving to device)
    target_mlp = cfg.lora.get("target_mlp", False)
    print(f"\nInjecting LoRA layers (target_mlp={target_mlp})...")
    lora_modules = inject_lora_into_stdit3(
        model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.get("dropout", 0.0),
        target_modules=cfg.lora.get("target_modules", ["qkv", "proj"]),
        target_blocks=cfg.lora.get("target_blocks", "all"),
        target_mlp=target_mlp,
    )
    
    # NOW move model (with LoRA layers) to device
    model = model.to(device, dtype)
    
    # Count parameters
    param_counts = count_lora_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  LoRA trainable: {param_counts['lora']:,}")
    print(f"  Trainable %: {param_counts['trainable_pct']:.4f}%")
    
    # Build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # Link embedder
    text_encoder.y_embedder = model.y_embedder
    
    model_load_time = time.time() - model_load_start
    print(f"\nModel loading time: {model_load_time:.2f} seconds")
    
    # Load video and encode
    print("\nLoading and encoding video...")
    encode_start = time.time()
    latents = load_video_for_training(
        args.video_path, vae, num_frames, image_size, device, dtype
    )
    print(f"Latent shape: {latents.shape}")
    
    # Free VAE memory
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    
    encode_time = time.time() - encode_start
    print(f"Video encoding time: {encode_time:.2f} seconds")
    
    # Train LoRA
    print("\nStarting LoRA training...")
    train_start_time = time.time()
    
    losses = train_lora(
        model=model,
        vae=None,  # Not needed during training
        text_encoder=text_encoder,
        scheduler=scheduler,
        latents=latents,
        caption=args.caption,
        num_steps=args.num_steps,
        lr=cfg.lr,
        device=device,
        dtype=dtype,
        image_size=image_size,
        num_frames=num_frames,
    )
    
    pure_train_time = time.time() - train_start_time
    total_time = model_load_time + encode_time + pure_train_time
    
    print(f"\n{'='*60}")
    print(f"Timing Breakdown:")
    print(f"  Model loading + LoRA injection: {model_load_time:.2f}s")
    print(f"  Video encoding: {encode_time:.2f}s")
    print(f"  Pure training (gradient steps): {pure_train_time:.2f}s")
    print(f"  Total: {total_time:.2f}s")
    print(f"{'='*60}")
    print(f"Average loss: {sum(losses) / len(losses):.4f}")
    
    # Save LoRA weights
    os.makedirs(args.output_dir, exist_ok=True)
    lora_path = os.path.join(args.output_dir, "lora_weights.pt")
    save_lora_weights(model, lora_path)
    print(f"\nLoRA weights saved to: {lora_path}")
    
    # Save training info with detailed timing breakdown
    info = {
        "video_path": args.video_path,
        "caption": args.caption,
        "num_steps": args.num_steps,
        "lr": cfg.lr,
        "lora_rank": cfg.lora.rank,
        "lora_alpha": cfg.lora.alpha,
        # Detailed timing breakdown
        "model_load_time_sec": model_load_time,
        "video_encode_time_sec": encode_time,
        "pure_train_time_sec": pure_train_time,  # Just gradient steps
        "total_time_sec": total_time,
        # Legacy field for backwards compatibility
        "train_time": pure_train_time,
        # Loss info
        "final_loss": losses[-1],
        "avg_loss": sum(losses) / len(losses),
        "losses": losses,
    }
    
    import json
    info_path = os.path.join(args.output_dir, "training_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Training info saved to: {info_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

