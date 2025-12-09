#!/usr/bin/env python3
"""
LoRA inference script for Open-Sora STDiT3 model.

Loads a trained LoRA checkpoint and generates video continuation.

Usage:
    python lora_inference.py \
        --config lora_experiment/configs/lora_finetune.py \
        --lora-weights lora_experiment/results/lora_weights.pt \
        --video-path path/to/input_video.mp4 \
        --caption "description of video" \
        --output-path output.mp4
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torchvision

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mmengine.config import Config

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module

# Import LoRA utilities
sys.path.insert(0, str(PROJECT_ROOT / "lora_experiment"))
from lora_layers import (
    inject_lora_into_stdit3,
    load_lora_weights,
    count_lora_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA inference for Open-Sora")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--lora-weights", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--video-path", type=str, required=True, help="Path to conditioning video")
    parser.add_argument("--caption", type=str, required=True, help="Video caption")
    parser.add_argument("--output-path", type=str, required=True, help="Output video path")
    parser.add_argument("--condition-frames", type=int, default=22, help="Number of conditioning frames")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def load_conditioning_video(video_path: str, num_frames: int, image_size: tuple, device: str, dtype):
    """Load conditioning frames from video."""
    import av
    import numpy as np
    import torch.nn.functional as F
    
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        if len(frames) >= num_frames:
            break
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
    container.close()
    
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    frames = np.stack(frames, axis=0)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    frames = frames / 255.0
    
    H, W = image_size
    frames = F.interpolate(frames.unsqueeze(0), size=(frames.shape[1], H, W), mode="trilinear", align_corners=False)
    frames = frames.squeeze(0)
    
    frames = frames * 2 - 1
    frames = frames.unsqueeze(0).to(device, dtype)
    
    return frames


def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    device = args.device
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16
    
    print(f"{'='*60}")
    print(f"LoRA Inference")
    print(f"{'='*60}")
    print(f"LoRA weights: {args.lora_weights}")
    print(f"Video: {args.video_path}")
    print(f"Caption: {args.caption}")
    print(f"{'='*60}")
    
    # Build model components
    print("\nLoading model components...")
    
    # Text encoder
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    
    # VAE
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    # Get sizes
    resolution = cfg.get("resolution", "360p")
    aspect_ratio = cfg.get("aspect_ratio", "9:16")
    image_size = get_image_size(resolution, aspect_ratio)
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
    
    # Inject LoRA BEFORE moving to device (with same config as training)
    print("\nInjecting LoRA layers...")
    inject_lora_into_stdit3(
        model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=0.0,  # No dropout during inference
        target_modules=cfg.lora.get("target_modules", ["qkv", "proj"]),
        target_blocks=cfg.lora.get("target_blocks", "all"),
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_weights}...")
    load_lora_weights(model, args.lora_weights)
    
    # NOW move model (with LoRA) to device
    model = model.to(device, dtype).eval()
    
    # Count parameters
    param_counts = count_lora_parameters(model)
    print(f"LoRA parameters loaded: {param_counts['lora']:,}")
    
    # Build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # Link embedder
    text_encoder.y_embedder = model.y_embedder
    
    # Load conditioning video
    print(f"\nLoading conditioning frames from {args.video_path}...")
    cond_frames = load_conditioning_video(
        args.video_path, args.condition_frames, image_size, device, dtype
    )
    
    # Encode conditioning frames
    with torch.no_grad():
        cond_latents = vae.encode(cond_frames)
    
    # Prepare for generation
    latent_size_hw = vae.get_latent_size(input_size)
    T_latent = latent_size_hw[0]
    cond_T_latent = cond_latents.shape[2]
    
    # Create mask for conditioning
    mask_index = list(range(cond_T_latent))
    
    # Prepare model inputs
    print("\nPreparing generation...")
    
    # Initialize latents with noise
    noise_shape = (1, vae.out_channels, T_latent, latent_size_hw[1], latent_size_hw[2])
    latents = torch.randn(noise_shape, device=device, dtype=dtype)
    
    # Copy conditioning latents
    latents[:, :, :cond_T_latent] = cond_latents
    
    # Create conditioning tensors for z_cond and z_cond_mask
    z_cond = torch.zeros_like(latents)
    z_cond[:, :, :cond_T_latent] = cond_latents
    z_cond_mask = torch.zeros_like(latents)
    z_cond_mask[:, :, :cond_T_latent] = 1.0
    
    # Generate
    print("Generating video...")
    
    # Additional model args (height, width, num_frames, fps) - all required by scheduler
    additional_args = {
        "height": torch.tensor([image_size[0]], device=device),
        "width": torch.tensor([image_size[1]], device=device),
        "num_frames": torch.tensor([T_latent * 4 + 1], device=device),  # Convert latent frames to pixel frames
        "fps": torch.tensor([24], device=device),
    }
    
    # Sample using scheduler - RFLOW.sample() takes specific args
    with torch.no_grad():
        samples = scheduler.sample(
            model=model,
            text_encoder=text_encoder,
            z=latents,
            prompts=[args.caption],  # List of prompts
            device=device,
            additional_args=additional_args,
            guidance_scale=7.5,
            image_cfg_scale=2.0,
            progress=True,
            z_cond=z_cond,
            z_cond_mask=z_cond_mask,
            mask_index=mask_index,
            use_sdedit=True,
            use_oscillation_guidance_for_text=True,
            use_oscillation_guidance_for_image=True,
        )
    
    # Decode
    print("Decoding video...")
    with torch.no_grad():
        video = vae.decode(samples.to(dtype))
    
    # Post-process and save
    video = video.squeeze(0)  # (C, T, H, W)
    video = (video + 1) / 2  # [-1, 1] -> [0, 1]
    video = video.clamp(0, 1)
    video = video.permute(1, 2, 3, 0)  # (T, H, W, C)
    video = (video * 255).to(torch.uint8).cpu()
    
    # Save video
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torchvision.io.write_video(args.output_path, video, fps=24)
    
    print(f"\nVideo saved to: {args.output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

