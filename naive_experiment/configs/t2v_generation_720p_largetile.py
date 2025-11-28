# Configuration for text-to-video generation at 720p with larger tiles
# Option B-revised: Use larger tile_size to reduce artifacts while staying within memory
# Balance between quality and memory usage

# 720p with proper 16-aligned dimensions
num_frames = 49
resolution = "720p"
aspect_ratio = "3:4"  # LANDSCAPE (H:W) - 960x1280, perfectly 16-aligned
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/t2v_720p_largetile"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

# Model configuration - with flash-attn
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=True,  # ✓ ENABLED - flash-attn 2.5.8 installed
    enable_layernorm_kernel=False,  # ✗ DISABLED - apex not available
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    force_huggingface=True,
)

# VAE configuration - larger tiles for better memory/quality balance
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=8,  # INCREASED from 4 - reduces artifacts while staying in memory
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
    num_sampling_steps=30,
    cfg_scale=7.5,
    use_oscillation_guidance=True,
    use_flaw_fix=True,
)

# Aesthetic conditioning for better quality
aes = 6.5  # "very good" aesthetic quality
flow = None  # Let model decide motion naturally

