#!/usr/bin/env python3
"""
Test V2V at 360p using EXACT official config from configs/opensora-v1-3/inference/v2v.py
This is to diagnose the RGB blocks issue.
"""

import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import (
    collect_references_batch,
    prepare_multi_resolution_info,
    prep_ref_and_mask,
)
from opensora.utils.misc import to_torch_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True, help="Path to conditioning video")
    parser.add_argument("--prompt", type=str, default="A person in a video.")
    parser.add_argument("--output", type=str, default="naive_experiment/results/test_v2v_360p_official.mp4")
    args = parser.parse_args()
    
    device = "cuda"
    dtype = to_torch_dtype("bf16")
    
    # EXACT official config from v2v.py
    resolution = "360p"
    aspect_ratio = "9:16"
    num_frames = 49  # Reduced from 113 for memory
    fps = 24
    condition_frame_length = 5  # latent frames
    
    image_size = get_image_size(resolution, aspect_ratio)
    
    print(f"=== V2V Test with EXACT Official Config ===")
    print(f"Resolution: {resolution}")
    print(f"Aspect ratio: {aspect_ratio}")
    print(f"Image size (H, W): {image_size}")
    print(f"Num frames: {num_frames}")
    print(f"Condition frame length (latent): {condition_frame_length}")
    print(f"Video path: {args.video_path}")
    print(f"Prompt: {args.prompt}")
    
    # Build models - EXACT official config
    print("\nBuilding models...")
    
    # VAE - EXACT from v2v.py
    vae = build_module(
        dict(
            type="OpenSoraVAE_V1_3",
            from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
            z_channels=16,
            micro_batch_size=1,
            micro_batch_size_2d=4,
            micro_frame_size=17,
            use_tiled_conv3d=True,
            tile_size=4,  # Official uses 4
            normalization="video",
            temporal_overlap=True,
            force_huggingface=True,
        ),
        MODELS
    ).to(device, dtype).eval()
    
    # Text encoder
    text_encoder = build_module(
        dict(
            type="t5",
            from_pretrained="google/t5-v1_1-xxl",
            model_max_length=300,
        ),
        MODELS,
        device=device
    )
    
    # Get latent size
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    print(f"Latent size: {latent_size}")
    
    # Model - EXACT from v2v.py (except enable_layernorm_kernel which needs apex)
    model = build_module(
        dict(
            type="STDiT3-XL/2",
            from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
            qk_norm=True,
            enable_flash_attn=True,
            enable_layernorm_kernel=False,  # Would be True with apex
            kernel_size=(8, 8, -1),
            use_spatial_rope=True,
            class_dropout_prob=0.0,
            force_huggingface=True,
        ),
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=False,
    ).to(device, dtype).eval()
    
    text_encoder.y_embedder = model.y_embedder
    
    # Scheduler - EXACT from v2v.py
    scheduler = build_module(
        dict(
            type="rflow",
            use_timestep_transform=True,
            num_sampling_steps=30,
            cfg_scale=7.5,
            scale_image_weight=True,
            initial_image_scale=1.0,
        ),
        SCHEDULERS
    )
    
    # Collect references (encode conditioning video with VAE)
    print("\nEncoding conditioning video...")
    refs = collect_references_batch([args.video_path], vae, image_size)
    
    # Prepare reference and mask
    target_shape = (1, vae.out_channels, *latent_size)
    ref, mask_index = prep_ref_and_mask(
        "v2v_head",  # cond_type from v2v.py uses i2v_head, but v2v_head is for video continuation
        condition_frame_length,
        refs,
        target_shape,
        loop=1,
        device=device,
        dtype=dtype,
    )
    
    print(f"Reference shape: {ref.shape if ref is not None else None}")
    print(f"Mask index: {mask_index}")
    
    # Prepare model args
    model_args = prepare_multi_resolution_info(
        "STDiT2", 1, image_size, num_frames, fps, device, dtype
    )
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        torch.manual_seed(42)
        z = torch.randn(1, vae.out_channels, *latent_size, device=device, dtype=dtype)
        
        # Prepare conditioning mask
        x_cond_mask = torch.zeros(1, vae.out_channels, *latent_size, device=device).to(dtype)
        if len(mask_index) > 0:
            x_cond_mask[:, :, mask_index, :, :] = 1.0
        
        # V2V generation with conditioning
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            prompts=[args.prompt],
            device=device,
            additional_args=model_args,
            progress=True,
            mask=None,
            mask_index=mask_index,
            image_cfg_scale=5.0,  # From v2v.py
            neg_prompts=None,
            z_cond=ref,
            z_cond_mask=x_cond_mask,
            use_sdedit=True,  # From v2v.py
            use_oscillation_guidance_for_text=True,  # From v2v.py
            use_oscillation_guidance_for_image=True,  # From v2v.py
        )
        
        print(f"Samples shape: {samples.shape}")
        
        # Decode
        print("Decoding...")
        video = vae.decode(samples.to(dtype)).squeeze(0)
        print(f"Video shape: {video.shape}")
        
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_sample(video, args.output, fps=fps, write_video_backend="pyav")
        print(f"\nSaved to: {args.output}")
    
    print("\n=== Done ===")
    print("This uses EXACT official v2v.py config parameters.")
    print("If this still has RGB blocks, the issue is NOT in our config.")


if __name__ == "__main__":
    main()

