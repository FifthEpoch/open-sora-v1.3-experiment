# Configuration for fine-tuning on a single video sample
# This trains the model on just one 45-frame video for a small number of steps

# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

# Single video bucket: 480p, 45 frames total
# During training: uses v2v_head masking (first 22 frames conditioned)
# During inference: uses condition_frame_length=8 (first 8 frames conditioned, generates 37)
bucket_config = {
    "480p": {
        45: (1, 0),  # batch_size=1, no range for search
    },
}

# Masking strategy for v2v continuation
mask_types = {
    "v2v_head": 1,  # Always use v2v_head for continuation from first frames
}

# Disable condition dropping to ensure we always use the conditioning frames
drop_condition = {
    "cond": 0.0,
    "text": 0.0,
    "null": 0.0,
    "keep": 1.0,
}

grad_checkpoint = True

# Acceleration settings
num_workers = 0  # Important: disable multiprocessing for single-sample dataset
num_bucket_build_workers = 1
dtype = "bf16"
plugin = "zero2"  # Use ZeRO-2 for memory efficiency

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="path/to/OpenSora-STDiT-v4",  # Will be replaced by script
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    class_dropout_prob=0.0,
    force_huggingface=True,
)

vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="path/to/OpenSora-VAE-v1.3",  # Will be replaced by script
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
    sample_method="logit-normal",
    use_timestep_transform=True,
    drop_condition=drop_condition,
)

# Log settings
seed = 42
outputs = "naive_experiment/results/finetuned_checkpoints"
wandb = False
epochs = 1  # Single epoch
log_every = 1  # Log every step since we have few steps
ckpt_every = 50  # Save checkpoint every 50 steps

# Optimization settings - CRITICAL for single-video training
lr = 1e-5  # Lower learning rate to prevent overfitting
warmup_steps = 0  # No warmup for such short training
use_cosine_scheduler = False  # Use constant LR for simplicity
grad_clip = 1.0
adam_eps = 1e-15
ema_decay = 0.99
accumulation_steps = 1  # No accumulation needed with batch_size=1

# Training will run for a small number of steps (e.g., 10-50)
# This is controlled by the script, not the config

