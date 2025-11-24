# Configuration for baseline video continuation inference with STRONG conditioning
# Testing hypothesis: weak conditioning is causing poor quality

num_frames = 49  # Total frames (22 conditioning + 27 continuation) - Open-Sora bucket size
condition_frame_length = 7  # INCREASED: More latent frames (7 ≈ 22 pixel frames)
resolution = "480p"
aspect_ratio = "4:3"  # 640x480
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/baselines_strong"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"  # Video-to-video continuation from head
use_sdedit = False  # CHANGED: Disable SDEdit to avoid adding noise to conditioning frames
use_oscillation_guidance_for_text = True  # Better text alignment
use_oscillation_guidance_for_image = True  # Better conditioning frame adherence

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=False,  # Disabled - flash-attn not available in current environment
    enable_layernorm_kernel=False,  # Disabled - apex not available in current environment
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    class_dropout_prob=0.0,
    force_huggingface=True,
)
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
text_encoder = dict(
    type="t5",
    from_pretrained="google/t5-v1_1-xxl",  # Will auto-download
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=50,
    cfg_scale = 10.0,  # INCREASED from 8.5 for much stronger prompt adherence
    scale_image_weight=True,
    initial_image_scale=1.0,
)

# STRONG conditioning settings
image_cfg_scale = 10.0  # INCREASED from 5.0 - much stronger image guidance for conditioning
aes = 7.0  # Aesthetic score conditioning
flow = None  # Motion score (optional)

