#!/usr/bin/env python3
"""
Generate baseline video continuation outputs (O_b) for all UCF-101 videos.

This script loads the vanilla Open-Sora v1.3 checkpoint once and generates
continuations for all videos in the dataset.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import Open-Sora modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import torch
from mmengine.config import Config
from tqdm import tqdm

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.datasets.utils import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    collect_references_batch,
    prep_ref_and_mask,
)
from opensora.utils.misc import create_logger, to_torch_dtype


def load_model_and_components(config_path, checkpoint_path=None, vae_path=None, device=None, dtype=None):
    """Load model, VAE, text encoder, and scheduler."""
    cfg = Config.fromfile(config_path)
    
    # Update checkpoint paths only if provided (otherwise use config defaults)
    if checkpoint_path is not None:
        cfg.model.from_pretrained = checkpoint_path
    if vae_path is not None:
        cfg.vae.from_pretrained = vae_path
    
    # Build components
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    # Prepare video size
    image_size = get_image_size(cfg.resolution, cfg.aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    
    # Build diffusion model
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
    
    # Build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    return model, vae, text_encoder, scheduler, cfg, image_size, num_frames


def split_video_for_conditioning(video_path, condition_frames=22, output_dir=None):
    """
    Split video into conditioning frames (first 22) and save as reference.
    Returns path to the conditioning video clip.
    """
    import av
    
    # Create output directory for conditioning videos
    if output_dir is None:
        output_dir = Path(video_path).parent / "conditioning"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output path
    video_name = Path(video_path).stem
    cond_video_path = output_dir / f"{video_name}_cond_{condition_frames}frames.mp4"
    
    if cond_video_path.exists():
        return str(cond_video_path)
    
    # Read input video
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    
    # Create output container
    output_container = av.open(str(cond_video_path), mode='w')
    output_stream = output_container.add_stream('libx264', rate=24)
    output_stream.width = video_stream.width
    output_stream.height = video_stream.height
    output_stream.pix_fmt = 'yuv420p'
    
    # Write first condition_frames frames
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count >= condition_frames:
            break
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
        frame_count += 1
    
    # Flush encoder
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
    
    # Generate
    with torch.no_grad():
        # Prepare timesteps
        torch.manual_seed(cfg.seed)
        z = torch.randn(target_shape, device=device, dtype=dtype)
        
        # Prepare mask for conditioning
        x_cond_mask = torch.zeros(target_shape, device=device).to(dtype)
        if len(mask_index) > 0:
            x_cond_mask[:, :, mask_index, :, :] = 1.0
        
        # Run scheduler
        # Model kwargs for RF scheduler time sampler - must match prepare_multi_resolution_info() format
        # Reference: opensora/utils/inference_utils.py:prepare_multi_resolution_info() for STDiT2
        batch_size = 1
        model_kwargs = {
            "height": torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(batch_size),
            "width": torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(batch_size),
            "num_frames": torch.tensor([num_frames], device=device, dtype=dtype).repeat(batch_size),
            # Optional keys used by some configurations
            "ar": torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(batch_size),
            "fps": torch.tensor([cfg.fps], device=device, dtype=dtype).repeat(batch_size),
        }
        samples = scheduler.sample(
            model,
            text_encoder,
            z,
            [caption],
            device,
            additional_args=model_kwargs,
            progress=False,
            mask=None,  # Not using mask_strategy
            mask_index=mask_index,
            image_cfg_scale=None,
            neg_prompts=None,
            z_cond=ref,
            z_cond_mask=x_cond_mask,
            use_sdedit=cfg.get("use_sdedit", False),
            use_oscillation_guidance_for_text=cfg.get("use_oscillation_guidance_for_text", False),
            use_oscillation_guidance_for_image=cfg.get("use_oscillation_guidance_for_image", False),
        )
    
    # Decode
    with torch.no_grad():
        samples = vae.decode(samples.to(dtype)).squeeze(0)  # [B,C,T,H,W] -> [C,T,H,W]
    
    # Save
    video_name = Path(video_path).stem
    output_path = Path(save_dir) / f"baseline_{video_idx:04d}_{video_name}.mp4"
    # Ensure output_path is absolute
    output_path = output_path.resolve()
    save_sample(
        samples,
        str(output_path),
        fps=cfg.fps,
        write_video_backend="pyav",
    )
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline video continuations")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline inference config")
    parser.add_argument("--data-csv", type=str, required=True, help="Path to UCF-101 metadata CSV")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Open-Sora STDiT checkpoint path or HuggingFace ID (optional, uses config default if not provided)")
    parser.add_argument("--vae-path", type=str, default=None, help="Open-Sora VAE checkpoint path or HuggingFace ID (optional, uses config default if not provided)")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--condition-frames", type=int, default=22, help="Number of conditioning frames")
    parser.add_argument("--num-videos", type=int, default=None, help="Number of videos to process (None = all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype("bf16" if device == "cuda" else "fp32")
    logger = create_logger()
    
    logger.info("Loading model and components...")
    model, vae, text_encoder, scheduler, cfg, image_size, num_frames = load_model_and_components(
        args.config, args.checkpoint_path, args.vae_path, device, dtype
    )
    
    # Load dataset
    df = pd.read_csv(args.data_csv)
    if args.num_videos is not None:
        df = df.head(args.num_videos)
    
    logger.info(f"Processing {len(df)} videos...")
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate continuations
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating baselines"):
        video_path = row['path']
        caption = row.get('text', row.get('caption', 'video'))
        
        # Make video path absolute if relative
        if not os.path.isabs(video_path):
            video_path = os.path.join(Path(args.data_csv).parent, video_path)
        
        try:
            output_path = generate_continuation(
                model, vae, text_encoder, scheduler, cfg,
                video_path, caption, args.condition_frames,
                image_size, num_frames, device, dtype,
                save_dir, idx
            )
            results.append({
                'video_idx': idx,
                'original_path': video_path,
                'baseline_output': output_path,
                'caption': caption,
            })
        except Exception as e:
            logger.error(f"Error processing video {idx} ({video_path}): {e}")
            results.append({
                'video_idx': idx,
                'original_path': video_path,
                'baseline_output': None,
                'error': str(e),
            })
    
    # Save results manifest
    results_df = pd.DataFrame(results)
    manifest_path = save_dir / "baseline_manifest.csv"
    results_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved manifest to {manifest_path}")
    logger.info(f"Successfully generated {sum(1 for r in results if r.get('baseline_output'))} baselines")


if __name__ == "__main__":
    main()

