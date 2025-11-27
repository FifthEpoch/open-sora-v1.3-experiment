# Configuration for text-to-video generation matching official Open-Sora v1.3 settings
# Based on configs/opensora-v1-3/inference/t2v.py

# Reduced from official 113 frames to fit our setup
num_frames = 49  # Official uses 113, but we use 49 to match UCF-101 preprocessing
resolution = "480p"  # Official uses 720p, we use 480p for memory constraints
aspect_ratio = "3:4"  # LANDSCAPE to match UCF-101 (official uses 9:16 portrait)
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/t2v_official"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

# Model configuration - EXACTLY matching official
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=True,  # ✓ ENABLED - flash-attn 2.5.8 installed successfully
    enable_layernorm_kernel=False,  # ✗ DISABLED - apex not available (expected)
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    force_huggingface=True,
)

# VAE configuration - matching official
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,  # Official uses 4 (smaller tiles = better quality, more memory)
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)

# Text encoder - matching official
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
    use_oscillation_guidance=True,  # Official flag
    use_flaw_fix=True,  # CRITICAL - Official uses this for quality
)

# Conditioning scores (not in official config, but commonly used)
# aes: aesthetic score (4.0-7.0 scale, higher = better quality)
# flow: motion strength (0.0-10.0 scale, higher = more motion)
aes = 6.5  # "very good" aesthetic quality
flow = None  # Let model decide motion naturally

