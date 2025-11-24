# Configuration matching OFFICIAL Open-Sora v1.3 video extending example
# Reference: https://github.com/hpcaitech/Open-Sora/blob/opensora/v1.3/docs/commands.md
#
# Official command:
# python scripts/inference_i2v.py configs/opensora-v1-3/inference/v2v.py \
#   --num-frames 97 --resolution 720p --aspect-ratio "9:16" --cond-type v2v_head --use-sdedit True \
#   --use-oscillation-guidance-for-image True --image-cfg-scale 2.0 \
#   --use-oscillation-guidance-for-text True --cfg-scale 7.5 \
#   --prompt 'A car driving on the ocean.{"reference_path": "https://cdn.openai.com/tmp/s/interp/d0.mp4"}'

num_frames = 49  # Using 49 instead of 97 due to UCF-101 constraints
condition_frame_length = 5  # Latent space frame count
resolution = "480p"  # Using 480p instead of 720p for faster testing
aspect_ratio = "9:16"  # PORTRAIT like official example (not 4:3!)
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/baselines_official"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"  
use_sdedit = True  # Match official
use_oscillation_guidance_for_text = True  # Match official
use_oscillation_guidance_for_image = True  # Match official

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    kernel_size=(8, 8, -1),
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
    from_pretrained="google/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=50,
    cfg_scale=7.5,  # MATCH OFFICIAL (was 8.5)
    scale_image_weight=True,
    initial_image_scale=1.0,
)

# MATCH OFFICIAL SETTINGS
image_cfg_scale = 2.0  # OFFICIAL VALUE (was 5.0!)
aes = 7.0
flow = None

