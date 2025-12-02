# Configuration for inference with fine-tuned model
# This generates O_f outputs using the video-specific fine-tuned weights
#
# CRITICAL: Must use 720p - 360p has RGB flashing blocks bug!
# Tested: 360p with various tile_size, aspect ratios all fail.
# Only 720p produces clean output.
#
# MEMORY OPTIMIZATION:
# - T2V at 720p works (no conditioning frames to encode)
# - V2V at 720p OOMs due to VAE encode of conditioning frames
# - Solution: Reduce conditioning frames from 22 to 8 (2 latent frames)
# - Using 9:16 (720, 1280) - both divisible by 8 ✓

num_frames = 25  # 8 conditioning + 17 generated (pixel space)
condition_frame_length = 2  # 2 latent frames ≈ 8 pixel frames (REDUCED!)
resolution = "720p"  # MUST use 720p - 360p has RGB blocks bug
aspect_ratio = "9:16"  # PORTRAIT - uses (720, 1280), both div by 8 ✓
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/finetuned"
multi_resolution = "STDiT2"
seed = 42  # Use same seed as baseline for fair comparison
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"  # Video-to-video continuation from head
use_sdedit = True  # Enable SDEdit for smoother temporal transitions
use_oscillation_guidance_for_text = True  # Better text alignment
use_oscillation_guidance_for_image = True  # Better conditioning frame adherence

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="path/to/finetuned/checkpoint",  # Will be replaced by script
    qk_norm=True,
    enable_flash_attn=True,  # ✓ ENABLED - flash-attn installed
    enable_layernorm_kernel=False,  # ✗ DISABLED - apex not available
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
    micro_batch_size_2d=4,  # Same as working T2V config
    micro_frame_size=17,  # Same as working T2V config
    use_tiled_conv3d=True,
    tile_size=16,  # Same as working T2V config
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
    num_sampling_steps=30,  # Keep reduced for memory (60 caused OOM)
    cfg_scale=7.5,  # Balanced prompt adherence
    use_oscillation_guidance=False,  # Disabled to save memory
    use_flaw_fix=False,  # Disabled to save memory
    scale_image_weight=True,
    initial_image_scale=1.0,
)

# Balanced conditioning parameters
image_cfg_scale = 2.0  # Image guidance scale for conditioning frames
aes = 6.5  # Good aesthetic quality
flow = None  # Motion strength (None to save memory)

