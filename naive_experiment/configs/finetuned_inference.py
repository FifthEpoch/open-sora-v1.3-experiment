# Configuration for inference with fine-tuned model
# This generates O_f outputs using the video-specific fine-tuned weights

num_frames = 45  # Total frames (22 conditioning + 23 continuation)
condition_frame_length = 22  # Number of conditioning frames
resolution = "480p"
aspect_ratio = "4:3"  # 640x480
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/finetuned"
multi_resolution = "STDiT2"
seed = 42  # Use same seed as baseline for fair comparison
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"  # Video-to-video continuation from head
use_sdedit = False  # Same as baseline
use_oscillation_guidance_for_text = False
use_oscillation_guidance_for_image = False

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="path/to/finetuned/checkpoint",  # Will be replaced by script
    qk_norm=True,
    enable_flash_attn=False,  # Disabled - flash-attn not available in current environment
    enable_layernorm_kernel=False,  # Disabled - apex not available in current environment
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
    from_pretrained="google/t5-v1_1-xxl",
    model_max_length=300,
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.5,
    scale_image_weight=True,
    initial_image_scale=1.0,
)

