#!/usr/bin/env python3
"""
Test T2V generation at 360p_16d to verify the model works at this resolution.
If T2V works but V2V doesn't, the issue is in conditioning handling.
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
    parser.add_argument("--output", type=str, default="naive_experiment/results/test_t2v_360p.mp4")
    parser.add_argument("--resolution", type=str, default="360p_16d")
    parser.add_argument("--aspect-ratio", type=str, default="3:4")
    args = parser.parse_args()
    
    device = "cuda"
    dtype = to_torch_dtype("bf16")
    
    # Get image size
    image_size = get_image_size(args.resolution, args.aspect_ratio)
    num_frames = 49
    fps = 24
    
    print(f"=== T2V Test at {args.resolution} ===")
    print(f"Image size (H, W): {image_size}")
    print(f"Num frames: {num_frames}")
    print(f"Prompt: {args.prompt}")
    
    # Build models
    print("Building models...")
    
    # VAE - same config as baseline_inference.py
    vae = build_module(
        dict(
            type="OpenSoraVAE_V1_3",
            from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
            z_channels=16,
            micro_batch_size=1,
            micro_batch_size_2d=4,
            micro_frame_size=17,
            use_tiled_conv3d=True,
            tile_size=4,  # Same as baseline
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
    
    # Scheduler - same as baseline
    scheduler = build_module(
        dict(
            type="rflow",
            use_timestep_transform=True,
            num_sampling_steps=30,
            cfg_scale=7.5,
            use_oscillation_guidance=False,
            use_flaw_fix=False,
        ),
        SCHEDULERS
    )
    
    # Prepare model args
    model_args = prepare_multi_resolution_info(
        "STDiT2", 1, image_size, num_frames, fps, device, dtype
    )
    
    # Generate
    print("Generating...")
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
        print(f"Saved to: {args.output}")
    
    print("=== Done ===")
    print(f"If this video looks good, the issue is in V2V conditioning, not the model/VAE.")
    print(f"If this video also has RGB blocks, the issue is in the model/VAE at 360p_16d.")


if __name__ == "__main__":
    main()

