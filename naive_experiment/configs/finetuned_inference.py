# Configuration for inference with fine-tuned model
# This generates O_f outputs using the video-specific fine-tuned weights

num_frames = 49  # Total frames (22 conditioning + 27 continuation) - Open-Sora bucket size
condition_frame_length = 5  # Number of conditioning frames IN LATENT SPACE (5 latent ≈ 16 pixel frames)
resolution = "360p_16d"  # Use 16-aligned variant for VAE compatibility (VAE requires div by 8)
aspect_ratio = "3:4"  # LANDSCAPE (H:W = 3:4 = 0.75) → 360p_16d: (416, 544), 720p: (832, 1110)
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/finetuned"
multi_resolution = "STDiT2"
seed = 42  # Use same seed as baseline for fair comparison
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"  # Video-to-video continuation from head
use_sdedit = True  # Enable SDEdit for smoother temporal transitions (Option B)
use_oscillation_guidance_for_text = True  # Better text alignment (Option B)
use_oscillation_guidance_for_image = True  # Better conditioning frame adherence (Option B)

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="path/to/finetuned/checkpoint",  # Will be replaced by script
    qk_norm=True,
    enable_flash_attn=True,  # ✓ ENABLED - flash-attn 2.5.8 installed successfully
    enable_layernorm_kernel=False,  # ✗ DISABLED - apex not available (expected)
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    class_dropout_prob=0.0,
    force_huggingface=False,  # Load self-trained checkpoint
)

vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,  # Normal for 480p (44% of 720p pixels)
    micro_frame_size=17,  # Normal for 480p
    use_tiled_conv3d=True,
    tile_size=4,  # Conservative tiling for safety
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)

text_encoder = dict(
    type="t5",
    from_pretrained="google/t5-v1_1-xxl",
    model_max_length=300,
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,  # Reduced for memory efficiency (60 steps → 137GB OOM)
    cfg_scale=7.5,  # Balanced prompt adherence
    use_oscillation_guidance=False,  # Disabled to save memory
    use_flaw_fix=False,  # Disabled to save memory
    scale_image_weight=True,
    initial_image_scale=1.0,
)

# Balanced conditioning parameters (memory-efficient)
image_cfg_scale = 2.0  # Image guidance scale for conditioning frames
aes = 6.5  # Good aesthetic quality
flow = None  # Motion strength (None to save memory)

