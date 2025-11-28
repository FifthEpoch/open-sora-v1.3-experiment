# Balanced configuration for 720p T2V generation
# Good quality with reasonable generation time

num_frames = 49
resolution = "720p"
aspect_ratio = "3:4"  # LANDSCAPE (H:W) - 960x1280
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/t2v_720p_balanced"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

# Model configuration
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    kernel_size=(8, 8, -1),
    use_spatial_rope=True,
    force_huggingface=True,
)

# VAE configuration
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=16,
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

# Balanced scheduler - moderate improvements without excessive time
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=40,  # Moderate - faster than ultra
    cfg_scale=8.5,  # Moderate prompt adherence
    use_oscillation_guidance=True,
    use_flaw_fix=True,
)

# Moderate conditioning
aes = 6.8  # Good aesthetic quality
flow = 5.5  # Moderate motion strength

