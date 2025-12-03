#!/usr/bin/env python3
"""
Test T2V at 360p with VAE tiling DISABLED.

This is to diagnose if the RGB blocks are caused by the tiled conv3d implementation.
If this works, the issue is in tiled_conv3d. If it still fails, the issue is elsewhere.
"""

import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import to_torch_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A person applying eye makeup in a bathroom mirror.")
    parser.add_argument("--output", type=str, default="naive_experiment/results/test_t2v_360p_no_tiling.mp4")
    args = parser.parse_args()
    
    device = "cuda"
    dtype = to_torch_dtype("bf16")
    
    resolution = "360p"
    aspect_ratio = "9:16"
    num_frames = 49
    fps = 24
    
    image_size = get_image_size(resolution, aspect_ratio)
    
    print(f"=== T2V Test at 360p with TILING DISABLED ===")
    print(f"Resolution: {resolution}")
    print(f"Aspect ratio: {aspect_ratio}")
    print(f"Image size (H, W): {image_size}")
    print(f"Num frames: {num_frames}")
    print(f"Prompt: {args.prompt}")
    print("")
    print("KEY CHANGE: use_tiled_conv3d=False")
    print("This will use more memory but should avoid any tiling artifacts.")
    
    # Build models
    print("\nBuilding models...")
    
    # VAE - DISABLE TILING
    vae = build_module(
        dict(
            type="OpenSoraVAE_V1_3",
            from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
            z_channels=16,
            micro_batch_size=1,
            micro_batch_size_2d=4,
            micro_frame_size=17,
            use_tiled_conv3d=False,  # DISABLED!
            # tile_size not needed when tiling is disabled
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
    
    # Model
    model = build_module(
        dict(
            type="STDiT3-XL/2",
            from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
            qk_norm=True,
            enable_flash_attn=True,
            enable_layernorm_kernel=False,
            kernel_size=(8, 8, -1),
            use_spatial_rope=True,
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
    
    # Scheduler
    scheduler = build_module(
        dict(
            type="rflow",
            use_timestep_transform=True,
            num_sampling_steps=30,
            cfg_scale=7.5,
            use_oscillation_guidance=True,
            use_flaw_fix=True,
        ),
        SCHEDULERS
    )
    
    # Prepare model args
    model_args = prepare_multi_resolution_info(
        "STDiT2", 1, image_size, num_frames, fps, device, dtype
    )
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        torch.manual_seed(42)
        z = torch.randn(1, vae.out_channels, *latent_size, device=device, dtype=dtype)
        
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            prompts=[args.prompt],
            device=device,
            additional_args=model_args,
            progress=True,
            mask=None,
        )
        
        print(f"Samples shape: {samples.shape}")
        
        # Decode
        print("Decoding (without tiling - may use more memory)...")
        video = vae.decode(samples.to(dtype)).squeeze(0)
        print(f"Video shape: {video.shape}")
        
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_sample(video, args.output, fps=fps, write_video_backend="pyav")
        print(f"\nSaved to: {args.output}")
    
    print("\n=== Done ===")
    print("This test DISABLED tiled conv3d.")
    print("")
    print("INTERPRETATION:")
    print("- If video looks good: The issue is in TiledConv3d implementation at 360p")
    print("- If video still has RGB blocks: The issue is NOT tiling-related")


if __name__ == "__main__":
    main()

