# Configuration for baseline video continuation inference
# This generates O_b outputs using vanilla Open-Sora v1.3 without fine-tuning
#
# Using 360p 9:16 (360x640) - EXACT official v2v.py config
# This resolution is officially supported and uses much less memory than 720p.
#
# Key settings from official v2v.py:
# - resolution = "360p", aspect_ratio = "9:16" → (360, 640)
# - tile_size = 4 (official default)
# - scale_image_weight = True, initial_image_scale = 1.0
# - image_cfg_scale = 5.0

num_frames = 49  # Official uses 113, reduced for memory
condition_frame_length = 5  # 5 latent frames (official default)
resolution = "360p"  # Official resolution
aspect_ratio = "9:16"  # Official aspect ratio → (360, 640)
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/baselines"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"  # Video-to-video continuation from head
use_sdedit = True  # From official v2v.py
use_oscillation_guidance_for_text = True  # From official v2v.py
use_oscillation_guidance_for_image = True  # From official v2v.py

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,  # Would be True with apex
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
    micro_batch_size_2d=4,  # Official default
    micro_frame_size=17,  # Official default
    use_tiled_conv3d=True,
    tile_size=4,  # Official default (NOT 16!)
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
    num_sampling_steps=30,  # Official default
    cfg_scale=7.5,  # Official default
    scale_image_weight=True,  # From official v2v.py
    initial_image_scale=1.0,  # From official v2v.py
)

# From official v2v.py
image_cfg_scale = 5.0  # Official default for V2V
aes = 7.0  # Official default
flow = None  # Official default
