#!/usr/bin/env python3
"""
Debug script to verify conditioning is working correctly.
This will generate ONE test video with extensive debug output.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import torch
from mmengine.config import Config

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.datasets.utils import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import (
    collect_references_batch,
    prep_ref_and_mask,
)
from opensora.utils.misc import create_logger, to_torch_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--caption", type=str, default="video")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype("bf16" if device == "cuda" else "fp32")
    logger = create_logger()
    
    # Load config
    cfg = Config.fromfile(args.config)
    logger.info(f"=== CONFIGURATION ===")
    logger.info(f"cond_type: {cfg.cond_type}")
    logger.info(f"condition_frame_length: {cfg.get('condition_frame_length', 'NOT SET')}")
    logger.info(f"use_sdedit: {cfg.get('use_sdedit', False)}")
    logger.info(f"use_oscillation_guidance_for_image: {cfg.get('use_oscillation_guidance_for_image', False)}")
    logger.info(f"image_cfg_scale: {cfg.get('image_cfg_scale', 'NOT SET')}")
    logger.info(f"num_frames: {cfg.num_frames}")
    
    # Build components
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
    
    logger.info(f"=== VIDEO SIZES ===")
    logger.info(f"image_size: {image_size}")
    logger.info(f"num_frames: {num_frames}")
    logger.info(f"latent_size: {latent_size}")
    
    # Create conditioning video (first 22 frames IN PIXEL SPACE)
    # Note: condition_frame_length in config is LATENT space (5), but we extract PIXEL frames (22)
    condition_frames_pixel = 22  # Extract 22 pixel frames
    condition_frames_latent = cfg.get('condition_frame_length', 5)  # 5 latent frames expected
    logger.info(f"\n=== CONDITIONING SETUP ===")
    logger.info(f"Extracting first {condition_frames_pixel} PIXEL frames from {args.video_path}")
    logger.info(f"Config condition_frame_length (latent): {condition_frames_latent}")
    
    import av
    container = av.open(str(args.video_path))
    output_dir = Path(args.output_path).parent / "debug_conditioning"
    output_dir.mkdir(parents=True, exist_ok=True)
    cond_video_path = output_dir / "conditioning.mp4"
    
    output_container = av.open(str(cond_video_path), mode='w')
    video_stream = container.streams.video[0]
    output_stream = output_container.add_stream('libx264', rate=24)
    output_stream.width = video_stream.width
    output_stream.height = video_stream.height
    output_stream.pix_fmt = 'yuv420p'
    
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count >= condition_frames_pixel:
            break
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
        frame_count += 1
    
    for packet in output_stream.encode():
        output_container.mux(packet)
    container.close()
    output_container.close()
    
    logger.info(f"Conditioning video saved: {cond_video_path}")
    logger.info(f"Conditioning frames extracted: {frame_count}")
    
    # Prepare prompt with mask_strategy (uses PIXEL frames for mask_strategy)
    mask_strategy = f"0,0,0,0,{condition_frames_pixel},0.0"
    prompt = f'{args.caption}.{{"reference_path": "{cond_video_path}", "mask_strategy": "{mask_strategy}"}}'
    logger.info(f"\n=== PROMPT ===")
    logger.info(f"Full prompt: {prompt}")
    
    # Extract reference
    import re
    import json
    parts = re.split(r"(?=[{])", prompt)
    reference_path = json.loads(parts[1])["reference_path"]
    logger.info(f"Reference path extracted: {reference_path}")
    
    # Collect references
    logger.info("\n=== COLLECTING REFERENCES ===")
    refs = collect_references_batch([reference_path], vae, image_size)
    logger.info(f"Number of reference batches: {len(refs)}")
    logger.info(f"Reference shape: {refs[0][0].shape if refs and refs[0] else 'EMPTY!'}")
    
    # Prepare reference and mask
    target_shape = (1, vae.out_channels, *latent_size)
    logger.info(f"\n=== PREPARING CONDITIONING ===")
    logger.info(f"target_shape: {target_shape}")
    
    ref, mask_index = prep_ref_and_mask(
        cfg.cond_type,
        condition_frames_latent,  # Use LATENT space frame count (5)
        refs,
        target_shape,
        loop=1,
        device=device,
        dtype=dtype,
    )
    
    logger.info(f"ref shape: {ref.shape}")
    logger.info(f"mask_index: {mask_index}")
    logger.info(f"mask_index length: {len(mask_index)}")
    logger.info(f"Expected mask_index length (latent): {condition_frames_latent}")
    logger.info(f"Mask index matches expected: {len(mask_index) == condition_frames_latent}")
    
    # Check if ref actually contains the conditioning frames
    ref_nonzero = (ref != 0).any(dim=(0, 1, 3, 4))  # Check which temporal indices have non-zero values
    logger.info(f"ref non-zero temporal indices: {torch.where(ref_nonzero)[0].tolist()}")
    logger.info(f"ref non-zero count: {ref_nonzero.sum().item()} / {latent_size[0]}")
    logger.info(f"Expected non-zero count: {condition_frames_latent}")
    
    # Create conditioning mask
    x_cond_mask = torch.zeros(target_shape, device=device).to(dtype)
    if len(mask_index) > 0:
        x_cond_mask[:, :, mask_index, :, :] = 1.0
    
    logger.info(f"x_cond_mask shape: {x_cond_mask.shape}")
    logger.info(f"x_cond_mask non-zero frames: {(x_cond_mask.sum(dim=(0,1,3,4)) > 0).sum().item()}")
    
    # Generate
    logger.info("\n=== RUNNING GENERATION ===")
    logger.info(f"use_sdedit: {cfg.get('use_sdedit', False)}")
    logger.info(f"image_cfg_scale: {cfg.get('image_cfg_scale', 5.0)}")
    
    with torch.no_grad():
        torch.manual_seed(cfg.seed)
        z = torch.randn(target_shape, device=device, dtype=dtype)
        
        batch_size = 1
        model_kwargs = {
            "height": torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(batch_size),
            "width": torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(batch_size),
            "num_frames": torch.tensor([num_frames], device=device, dtype=dtype).repeat(batch_size),
            "ar": torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(batch_size),
            "fps": torch.tensor([cfg.fps], device=device, dtype=dtype).repeat(batch_size),
        }
        
        logger.info("Starting scheduler.sample()...")
        samples = scheduler.sample(
            model,
            text_encoder,
            z,
            [args.caption],  # Clean caption without JSON
            device,
            additional_args=model_kwargs,
            progress=True,
            mask=None,
            mask_index=mask_index,
            image_cfg_scale=cfg.get("image_cfg_scale", 5.0),
            neg_prompts=None,
            z_cond=ref,
            z_cond_mask=x_cond_mask,
            use_sdedit=cfg.get("use_sdedit", False),
            use_oscillation_guidance_for_text=cfg.get("use_oscillation_guidance_for_text", False),
            use_oscillation_guidance_for_image=cfg.get("use_oscillation_guidance_for_image", False),
        )
        logger.info(f"samples shape: {samples.shape}")
    
    # Decode
    logger.info("\n=== DECODING ===")
    with torch.no_grad():
        full_video = vae.decode(samples.to(dtype)).squeeze(0)
    logger.info(f"decoded video shape: {full_video.shape}")
    
    # Resize if needed
    target_h, target_w = 480, 640
    if full_video.shape[2:] != (target_h, target_w):
        import torch.nn.functional as F
        logger.info(f"Resizing from {full_video.shape[2:]} to ({target_h}, {target_w})")
        full_video = F.interpolate(
            full_video,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
    
    # Save
    logger.info(f"\n=== SAVING ===")
    output_path = Path(args.output_path).resolve()
    logger.info(f"Saving to: {output_path}")
    save_sample(
        full_video,
        str(output_path),
        fps=cfg.fps,
        write_video_backend="pyav",
    )
    
    logger.info("\n=== VERIFICATION CHECKS ===")
    logger.info(f"✓ Config has cond_type='v2v_head': {cfg.cond_type == 'v2v_head'}")
    logger.info(f"✓ mask_index is correct length: {len(mask_index) == condition_frames_latent}")
    logger.info(f"✓ ref contains conditioning frames: {ref_nonzero.sum().item() >= condition_frames_latent}")
    logger.info(f"✓ x_cond_mask is set: {(x_cond_mask.sum() > 0).item()}")
    logger.info(f"✓ use_sdedit is enabled: {cfg.get('use_sdedit', False)}")
    logger.info(f"✓ image_cfg_scale is set: {cfg.get('image_cfg_scale', 'NOT SET') != 'NOT SET'}")
    
    logger.info("\n=== DONE ===")
    logger.info(f"Output video: {output_path}")
    logger.info(f"Conditioning video: {cond_video_path}")


if __name__ == "__main__":
    main()

