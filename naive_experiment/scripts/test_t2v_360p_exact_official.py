#!/usr/bin/env python3
"""
Test T2V at 360p using EXACT official config from configs/opensora-v1-3/inference/t2v.py
This is to verify that 360p T2V works before running the full experiment.

Key differences from previous 360p tests:
- Uses tile_size=4 (official default, NOT 16)
- Uses exact official scheduler params
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
    parser.add_argument("--output", type=str, default="naive_experiment/results/test_t2v_360p_exact_official.mp4")
    args = parser.parse_args()
    
    device = "cuda"
    dtype = to_torch_dtype("bf16")
    
    # EXACT official config from t2v.py (with resolution changed to 360p)
    resolution = "360p"
    aspect_ratio = "9:16"
    num_frames = 49  # Reduced from 113 for memory
    fps = 24
    
    image_size = get_image_size(resolution, aspect_ratio)
    
    print(f"=== T2V Test with EXACT Official Config (360p) ===")
    print(f"Resolution: {resolution}")
    print(f"Aspect ratio: {aspect_ratio}")
    print(f"Image size (H, W): {image_size}")
    print(f"Num frames: {num_frames}")
    print(f"Prompt: {args.prompt}")
    
    # Build models - EXACT official config from t2v.py
    print("\nBuilding models...")
    
    # VAE - EXACT from t2v.py
    vae = build_module(
        dict(
            type="OpenSoraVAE_V1_3",
            from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
            z_channels=16,
            micro_batch_size=1,
            micro_batch_size_2d=4,
            micro_frame_size=17,
            use_tiled_conv3d=True,
            tile_size=4,  # CRITICAL: Official uses 4, NOT 16!
            normalization="video",
            temporal_overlap=True,
            force_huggingface=True,
        ),
        MODELS
    ).to(device, dtype).eval()
    
    # Text encoder - EXACT from t2v.py
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
    
    # Model - EXACT from t2v.py (except enable_layernorm_kernel which needs apex)
    model = build_module(
        dict(
            type="STDiT3-XL/2",
            from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
            qk_norm=True,
            enable_flash_attn=True,
            enable_layernorm_kernel=False,  # Would be True with apex
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
    
    # Scheduler - EXACT from t2v.py
    scheduler = build_module(
        dict(
            type="rflow",
            use_timestep_transform=True,
            num_sampling_steps=30,
            cfg_scale=7.5,
            use_oscillation_guidance=True,  # From official t2v.py
            use_flaw_fix=True,  # From official t2v.py
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
        
        # Pure T2V - no conditioning
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
        print("Decoding...")
        video = vae.decode(samples.to(dtype)).squeeze(0)
        print(f"Video shape: {video.shape}")
        
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_sample(video, args.output, fps=fps, write_video_backend="pyav")
        print(f"\nSaved to: {args.output}")
    
    print("\n=== Done ===")
    print("This uses EXACT official t2v.py config parameters at 360p.")
    print("Key settings:")
    print("  - tile_size=4 (official default)")
    print("  - use_oscillation_guidance=True")
    print("  - use_flaw_fix=True")
    print("")
    print("If this still has RGB blocks, the issue is NOT in our config.")
    print("If this works, we can proceed with the full experiment at 360p.")


if __name__ == "__main__":
    main()

