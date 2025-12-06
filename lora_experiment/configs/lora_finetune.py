# Configuration for LoRA fine-tuning on a single video
#
# LoRA allows training <1% of parameters while achieving similar results
# to full fine-tuning, with much faster training and lower memory usage.

# ============================================================================
# LoRA Configuration
# ============================================================================
lora = dict(
    rank=8,                    # Low-rank dimension (smaller = fewer params, faster)
    alpha=16,                  # Scaling factor (typically 2x rank)
    dropout=0.0,               # Dropout for LoRA layers
    target_modules=["qkv", "proj"],  # Which modules to apply LoRA to
    target_blocks="all",       # "all", "spatial", or "temporal"
)

# ============================================================================
# Dataset settings (same as naive experiment)
# ============================================================================
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

bucket_config = {
    "360p": {
        22: (1, 1),
    },
}

mask_types = {
    "v2v_head": 1,
}

drop_condition = {
    "cond": 0.0,
    "text": 0.0,
    "null": 0.0,
    "keep": 1.0,
}

# ============================================================================
# Model Configuration
# ============================================================================
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4-360p",
    qk_norm=True,
    enable_flash_attn=True,
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
    sample_method="logit-normal",
    use_timestep_transform=True,
    drop_condition=drop_condition,
)

# ============================================================================
# Training settings
# ============================================================================
dtype = "bf16"
grad_checkpoint = True
num_workers = 0
num_bucket_build_workers = 1

# Optimization - LoRA requires different settings than full fine-tuning
# Higher LR is typically used for LoRA since we're training fewer parameters
lr = 1e-4                      # Higher LR for LoRA (vs 5e-5 for full FT)
warmup_steps = 0
use_cosine_scheduler = False
grad_clip = 1.0
adam_eps = 1e-15
accumulation_steps = 1

# Logging
seed = 42
outputs = "lora_experiment/results/lora_checkpoints"
wandb = False
epochs = 1
log_every = 1
ckpt_every = 50

# Training will run for 20-50 steps (faster than full fine-tuning)
# Controlled by the script via --finetune-steps

