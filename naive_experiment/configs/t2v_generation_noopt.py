# Configuration for text-to-video generation WITHOUT flash-attn/apex optimizations
# This config should work even if optimizations aren't installed, but will be slower

num_frames = 49
resolution = "480p"
aspect_ratio = "3:4"  # LANDSCAPE
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/t2v_noopt"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

# Model configuration - optimizations DISABLED
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=False,  # Disabled - doesn't require flash-attn package
    enable_layernorm_kernel=False,  # Disabled - doesn't require apex package
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    force_huggingface=True,
)

# VAE configuration - smaller tiles for better quality
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,  # Use official tile_size=4 for better quality
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)

# Text encoder
text_encoder = dict(
    type="t5",
    from_pretrained="google/t5-v1_1-xxl",
    model_max_length=300,
)

# Scheduler configuration - matching official
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,  # Official uses 30
    cfg_scale=7.5,
    use_oscillation_guidance=True,
    use_flaw_fix=True,  # CRITICAL for quality
)

aes = None
flow = None

