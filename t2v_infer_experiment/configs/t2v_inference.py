# Configuration for Text-to-Video inference at 360p
# No conditioning frames - pure T2V generation

resolution = "360p"
aspect_ratio = "9:16"
num_frames = 49  # Same as LoRA experiments (22 cond + 27 generated = 49 total)
fps = 24

# Model - use 360p specific model
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4-360p",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    kernel_size=(8, 8, -1),
    use_spatial_rope=True,
    force_huggingface=True,
)

# VAE
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
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

# Scheduler for T2V (no SDEdit, no image conditioning)
scheduler = dict(
    type="rflow",
    num_sampling_steps=50,
    cfg_scale=7.5,
    use_discrete_timesteps=False,
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# No conditioning for T2V
cond_type = None  # Pure T2V, no conditioning

