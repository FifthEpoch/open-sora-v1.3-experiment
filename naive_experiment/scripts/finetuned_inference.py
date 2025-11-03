#!/usr/bin/env python3
"""
Generate fine-tuned video continuation outputs (O_f) using per-video fine-tuned weights.

Similar to baseline_inference.py but uses fine-tuned checkpoint.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import torch
from mmengine.config import Config
from tqdm import tqdm

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import (
    collect_references_batch,
    prep_ref_and_mask,
    save_sample,
)
from opensora.utils.misc import create_logger, to_torch_dtype
from opensora.utils.config_utils import parse_configs

# Duplicate helper functions to avoid import issues
def load_model_and_components(config_path, checkpoint_path, vae_path=None, device=None, dtype=None):
    """Load model, VAE, text encoder, and scheduler."""
    from mmengine.config import Config
    from opensora.datasets.aspect import get_image_size, get_num_frames
    from opensora.registry import MODELS, SCHEDULERS, build_module
    
    cfg = Config.fromfile(config_path)
    cfg.model.from_pretrained = checkpoint_path  # Fine-tuned checkpoint is always required
    # Update VAE path only if provided
    if vae_path is not None:
        cfg.vae.from_pretrained = vae_path
    
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
    
    return model, vae, text_encoder, scheduler, cfg, image_size, num_frames


def split_video_for_conditioning(video_path, condition_frames=8, output_dir=None):
    """Split video into conditioning frames and save as reference."""
    import av
    from pathlib import Path
    
    if output_dir is None:
        output_dir = Path(video_path).parent / "conditioning"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    cond_video_path = output_dir / f"{video_name}_cond_{condition_frames}frames.mp4"
    
    if cond_video_path.exists():
        return str(cond_video_path)
    
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    
    output_container = av.open(str(cond_video_path), mode='w')
    output_stream = output_container.add_stream('libx264', rate=24)
    output_stream.width = video_stream.width
    output_stream.height = video_stream.height
    output_stream.pix_fmt = 'yuv420p'
    
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count >= condition_frames:
            break
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
        frame_count += 1
    
    for packet in output_stream.encode():
        output_container.mux(packet)
    
    container.close()
    output_container.close()
    
    return str(cond_video_path)


def generate_continuation(
    model, vae, text_encoder, scheduler, cfg,
    video_path, caption, condition_frames, image_size, num_frames,
    device, dtype, save_dir, video_idx
):
    """Generate continuation for a single video."""
    # Split video and create conditioning clip
    cond_video_path = split_video_for_conditioning(
        video_path, condition_frames=condition_frames,
        output_dir=Path(save_dir) / "conditioning"
    )
    
    # Prepare prompt with reference
    prompt = f'{caption}.{{"reference_path": "{cond_video_path}"}}'
    
    # Extract reference path from prompt
    import re
    import json
    parts = re.split(r"(?=[{])", prompt)
    assert len(parts) == 2, f"Invalid prompt: {prompt}"
    reference_path = json.loads(parts[1])["reference_path"]
    
    # Collect references
    refs = collect_references_batch([reference_path], vae, image_size)
    
    # Prepare latent size
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    target_shape = (1, vae.out_channels, *latent_size)
    
    # Prepare reference and mask
    ref, mask_index = prep_ref_and_mask(
        cfg.cond_type,
        condition_frames,
        refs,
        target_shape,
        loop=1,
        device=device,
        dtype=dtype,
    )
    
    # Encode text
    text_emb = text_encoder.encode(text_encoder.tokenize_fn([caption]))['y']
    
    # Generate
    with torch.no_grad():
        # Prepare timesteps (use same seed as baseline for fair comparison)
        torch.manual_seed(cfg.seed)
        z = torch.randn(target_shape, device=device, dtype=dtype)
        
        # Prepare mask for conditioning
        x_cond_mask = torch.zeros(target_shape, device=device).to(dtype)
        if len(mask_index) > 0:
            x_cond_mask[:, :, mask_index, :, :] = 1.0
        
        # Run scheduler
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            z_cond=ref,
            z_cond_mask=x_cond_mask,
            prompts=[caption],
            device=device,
            additional_args=None,
            progress=False,
            mask=None,
            mask_index=mask_index,
            image_cfg_scale=None,
            neg_prompts=None,
            use_sdedit=cfg.get("use_sdedit", False),
            use_oscillation_guidance_for_text=cfg.get("use_oscillation_guidance_for_text", False),
            use_oscillation_guidance_for_image=cfg.get("use_oscillation_guidance_for_image", False),
        )
    
    # Decode
    with torch.no_grad():
        samples = vae.decode(samples.to(dtype))
    
    # Save
    video_name = Path(video_path).stem
    output_path = Path(save_dir) / f"finetuned_{video_idx:04d}_{video_name}.mp4"
    save_sample(
        samples,
        str(output_path),
        fps=cfg.fps,
        loop=cfg.get("loop", 1),
    )
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuned video continuations")
    parser.add_argument("--config", type=str, required=True, help="Path to fine-tuned inference config")
    parser.add_argument("--finetuned-checkpoint", type=str, required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--vae-path", type=str, default=None, help="Open-Sora VAE checkpoint path or HuggingFace ID (optional, uses config default if not provided)")
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    parser.add_argument("--caption", type=str, required=True, help="Video caption")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save output")
    parser.add_argument("--condition-frames", type=int, default=8, help="Number of conditioning frames")
    parser.add_argument("--video-idx", type=int, default=0, help="Video index for naming")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype("bf16" if device == "cuda" else "fp32")
    logger = create_logger()
    
    logger.info("Loading fine-tuned model and components...")
    model, vae, text_encoder, scheduler, cfg, image_size, num_frames = load_model_and_components(
        args.config, args.finetuned_checkpoint, args.vae_path, device, dtype
    )
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate continuation
    try:
        output_path = generate_continuation(
            model, vae, text_encoder, scheduler, cfg,
            args.video_path, args.caption, args.condition_frames,
            image_size, num_frames, device, dtype,
            save_dir, args.video_idx
        )
        logger.info(f"Generated continuation saved to: {output_path}")
        print(output_path)  # Print for script capture
    except Exception as e:
        logger.error(f"Error generating continuation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

