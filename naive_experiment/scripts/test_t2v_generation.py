#!/usr/bin/env python3
"""
Test script for pure text-to-video generation (no conditioning)
Tests the model's baseline capability to generate videos from text prompts only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add repository root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from mmengine.config import Config

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.datasets.utils import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import prepare_multi_resolution_info

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Test pure text-to-video generation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save output video')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        cfg.seed = args.seed
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
    
    logger.info("=" * 50)
    logger.info("TEXT-TO-VIDEO GENERATION TEST")
    logger.info("=" * 50)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Resolution: {cfg.resolution}")
    logger.info(f"Aspect ratio: {cfg.aspect_ratio}")
    logger.info(f"Num frames: {cfg.num_frames}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info("")
    
    # Build model components
    logger.info("Building model components...")
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    image_size = get_image_size(cfg.resolution, cfg.aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    
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
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    logger.info(f"Image size: {image_size}")
    logger.info(f"Latent size: {latent_size}")
    logger.info("")
    
    # Prepare multi-resolution info (additional model args)
    model_kwargs = prepare_multi_resolution_info(
        cfg.multi_resolution,
        1,  # batch_size
        image_size,
        num_frames,
        cfg.fps,
        device,
        dtype,
    )
    
    # Generate from pure noise (no conditioning)
    logger.info("Generating video from text prompt...")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"CFG scale: {cfg.scheduler.cfg_scale}")
    
    with torch.no_grad():
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        
        # Start from random noise
        z = torch.randn(1, vae.out_channels, *latent_size, device=device, dtype=dtype)
        
        # Run diffusion sampling (scheduler will encode text internally)
        samples = scheduler.sample(
            model,
            text_encoder,  # Pass text_encoder, not pre-encoded text
            z,
            [args.prompt],  # Prompts as list of strings
            device,
            additional_args=model_kwargs,
        )
        samples = samples[0]  # [1, C, T, H, W]
    
    logger.info(f"Generated latent shape: {samples.shape}")
    logger.info("")
    
    # Decode to video
    logger.info("Decoding latent to video...")
    with torch.no_grad():
        video = vae.decode(samples.to(dtype)).squeeze(0)  # [C, T, H, W]
    
    logger.info(f"Decoded video shape: {video.shape}")
    logger.info("")
    
    # Save video
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving video to: {output_path}")
    save_sample(
        video,
        save_path=str(output_path),
        fps=cfg.fps,
        normalize=True,
        value_range=(-1, 1),
    )
    
    logger.info("=" * 50)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Output: {output_path}")
    logger.info("")


if __name__ == "__main__":
    main()

