#!/usr/bin/env python3
"""
Text-to-Video Inference Script

Generates videos from text prompts only (no conditioning frames).
Uses the same captions from UCF101 videos as the LoRA experiments.
"""

import argparse
import os
import sys
import time
import torch
import imageio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mmengine.config import Config
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-Video Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output-path", type=str, required=True, help="Output video path")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of frames to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = args.device
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("Text-to-Video Inference")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output_path}")
    print(f"Frames: {args.num_frames}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override num_frames if specified
    if args.num_frames:
        cfg.num_frames = args.num_frames
    
    # Build components
    print("\nLoading model components...")
    load_start = time.time()
    
    # Text encoder
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    print(f"  ✓ Text encoder loaded")
    
    # VAE
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    print(f"  ✓ VAE loaded")
    
    # Get sizes
    image_size = get_image_size(cfg.resolution, cfg.aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    
    print(f"\nVideo settings:")
    print(f"  Resolution: {cfg.resolution} ({image_size[1]}x{image_size[0]})")
    print(f"  Frames: {num_frames}")
    print(f"  Latent size: {latent_size}")
    
    # STDiT model
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=False,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder
    print(f"  ✓ STDiT model loaded")
    
    # Scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    print(f"  ✓ Scheduler loaded")
    
    load_time = time.time() - load_start
    print(f"\nModel loading time: {load_time:.1f}s")
    
    # Initialize random latents
    print("\nInitializing latents...")
    z = torch.randn(
        1, vae.out_channels, *latent_size,
        device=device, dtype=dtype
    )
    
    # Prepare model arguments for T2V (batch size = 1)
    model_args = {
        "height": torch.tensor([image_size[0]], device=device),
        "width": torch.tensor([image_size[1]], device=device),
        "fps": torch.tensor([cfg.fps], device=device),
        "num_frames": torch.tensor([num_frames], device=device),
    }
    
    # Generate video
    print(f"\nGenerating video from prompt: '{args.prompt}'")
    gen_start = time.time()
    
    # Prompts must be a list for scheduler.sample()
    prompts = [args.prompt]
    
    with torch.no_grad():
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            prompts=prompts,
            device=device,
            additional_args=model_args,
            progress=True,
            # No conditioning for T2V
            z_cond=None,
            z_cond_mask=None,
            mask_index=None,
        )
    
    gen_time = time.time() - gen_start
    print(f"Generation time: {gen_time:.1f}s")
    
    # Decode
    print("\nDecoding latents to video...")
    decode_start = time.time()
    
    with torch.no_grad():
        video = vae.decode(samples.to(dtype))
    
    decode_time = time.time() - decode_start
    print(f"Decode time: {decode_time:.1f}s")
    
    # Post-process
    video = video.squeeze(0)  # (C, T, H, W)
    video = (video + 1) / 2  # [-1, 1] -> [0, 1]
    video = video.clamp(0, 1)
    video = video.permute(1, 2, 3, 0)  # (T, H, W, C)
    video = (video * 255).to(torch.uint8).cpu().numpy()
    
    # Save video
    print(f"\nSaving video to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    imageio.mimwrite(args.output_path, video, fps=cfg.fps, codec='libx264', quality=8)
    
    # Summary
    total_time = time.time() - load_start
    print("\n" + "=" * 60)
    print("T2V Inference Complete")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"  - Model loading: {load_time:.1f}s")
    print(f"  - Generation: {gen_time:.1f}s")
    print(f"  - Decoding: {decode_time:.1f}s")
    print(f"Output: {args.output_path}")
    print("=" * 60)
    
    return {
        "total_time": total_time,
        "load_time": load_time,
        "gen_time": gen_time,
        "decode_time": decode_time,
    }


if __name__ == "__main__":
    main()

