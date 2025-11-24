# Configuration for pure text-to-video generation (no conditioning)
# Testing baseline model capabilities without video continuation

num_frames = 49  # Generate 49 frames from scratch
resolution = "480p"
aspect_ratio = "3:4"  # LANDSCAPE to match UCF-101
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/t2v_generation"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

# NO CONDITIONING - pure text-to-video generation
cond_type = None  # No video conditioning
use_sdedit = False  # Not applicable without conditioning
use_oscillation_guidance_for_text = True  # Better text alignment
use_oscillation_guidance_for_image = False  # Not applicable

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
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
    tile_size=16,  # Use default tile size
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
    num_sampling_steps=50,
    cfg_scale=7.5,  # Standard text guidance
    scale_image_weight=False,  # No image conditioning
    initial_image_scale=1.0,
)

# Text-only generation parameters
image_cfg_scale = None  # Not applicable (no image conditioning)
aes = 7.0  # Aesthetic score conditioning
flow = None  # Motion score (optional)

