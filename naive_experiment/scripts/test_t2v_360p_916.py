#!/usr/bin/env python3
"""
Test T2V generation at 360p with 9:16 aspect ratio (official config).
This is the EXACT resolution from the official v2v.py config.
Resolution: 360p, Aspect: 9:16 → (360, 640) - both divisible by 8!
"""

import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmengine.config import Config
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import to_torch_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A person applying eye makeup in front of a mirror.")
    parser.add_argument("--output", type=str, default="naive_experiment/results/test_t2v_360p_916.mp4")
    args = parser.parse_args()
    
    device = "cuda"
    dtype = to_torch_dtype("bf16")
    
    # Use EXACT official config: 360p with 9:16
    # This gives (360, 640) which is perfectly divisible by 8
    resolution = "360p"
    aspect_ratio = "9:16"
    
    image_size = get_image_size(resolution, aspect_ratio)
    num_frames = 49
    fps = 24
    
    print(f"=== T2V Test with OFFICIAL Resolution ===")
    print(f"Resolution: {resolution}")
    print(f"Aspect ratio: {aspect_ratio}")
    print(f"Image size (H, W): {image_size}")
    print(f"H % 8 = {image_size[0] % 8}, W % 8 = {image_size[1] % 8}")
    print(f"Num frames: {num_frames}")
    print(f"Prompt: {args.prompt}")
    
    # Build models
    print("\nBuilding models...")
    
    # VAE - EXACT official config from v2v.py
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
    
    # Model - EXACT official config
    model = build_module(
        dict(
            type="STDiT3-XL/2",
            from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
            qk_norm=True,
            enable_flash_attn=True,
            enable_layernorm_kernel=False,  # We don't have apex
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
    
    # Scheduler - EXACT official config from v2v.py
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
            mask_index=[],  # Empty - no conditioning
            image_cfg_scale=None,
            neg_prompts=None,
            z_cond=None,
            z_cond_mask=None,
            use_sdedit=False,
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
    print("This uses the EXACT official resolution (360p 9:16 = 360×640)")
    print("If this works, we need to use 9:16 aspect ratio instead of 3:4")


if __name__ == "__main__":
    main()

