# Configuration for CLEAN baseline video continuation (no tiling artifacts)
# Key change: use_tiled_conv3d=False to eliminate rectangular flashes

num_frames = 49  # Total frames (22 conditioning + 27 continuation)
condition_frame_length = 7  # Increased from 5 for more conditioning context (~27 pixel frames)
resolution = "480p"
aspect_ratio = "3:4"  # LANDSCAPE (H:W = 3:4 = 0.75) to match UCF-101's landscape format
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/baselines_clean"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"  # Video-to-video continuation from head
use_sdedit = True  # CRITICAL: Preserves conditioning frames pixel-perfect
use_oscillation_guidance_for_text = True  # Better text alignment
use_oscillation_guidance_for_image = True  # Better conditioning frame adherence

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=False,  # Disabled - flash-attn not available
    enable_layernorm_kernel=False,  # Disabled - apex not available
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
    use_tiled_conv3d=False,  # KEY FIX: Disabled to prevent rectangular artifacts/flashes
    tile_size=16,  # Not used when tiling disabled, but kept for reference
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
    cfg_scale=8.5,  # Text prompt guidance strength
    scale_image_weight=True,
    initial_image_scale=1.0,
)

# Conditioning parameters
image_cfg_scale = 5.0  # Moderate image guidance (balanced between 2.0 official and 10.0 strong)
aes = 7.0  # Aesthetic score conditioning
flow = None  # Motion score (optional)

