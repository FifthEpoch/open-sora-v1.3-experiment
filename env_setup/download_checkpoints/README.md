# Downloading Open-Sora Checkpoints

Open-Sora requires pre-trained model checkpoints for inference. The default config files reference checkpoints stored in developer directories that won't exist on your system.

## Quick Start

Download all required checkpoints:

```bash
python download_checkpoints.py --output-dir /path/to/your/checkpoints
```

## Available Models

The script downloads the following checkpoints from [Hugging Face](https://huggingface.co):

1. **OpenSora-STDiT-v4** (~1GB)
   - STDiT3-XL/2 model weights for video diffusion
   - Repository: [hpcai-tech/OpenSora-STDiT-v4](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v4)

2. **OpenSora-VAE-v1.3** (~330MB)
   - VAE model for video encoding/decoding
   - Repository: [hpcai-tech/OpenSora-VAE-v1.3](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.3)

3. **T5-v1.1-XXL** (~20GB) ⚠️ Optional
   - T5 XXL text encoder for text embeddings
   - Repository: [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl)
   - Note: This is very large and will auto-download if not pre-downloaded

## Usage

### Basic Usage

Download essential checkpoints to a directory (STDiT + VAE, **not** T5):

```bash
python download_checkpoints.py --output-dir ./checkpoints
```

> **Note**: By default, the script downloads only STDiT and VAE models (~1.3GB total). The T5 model (~20GB) is not included by default because it's very large and will auto-download when needed. Use `--model all` to download everything.

### Download Specific Models

Download only the STDiT model:

```bash
python download_checkpoints.py --output-dir ./checkpoints --model stdit
```

Download only the VAE model:

```bash
python download_checkpoints.py --output-dir ./checkpoints --model vae
```

Download the T5 text encoder (very large ~20GB):

```bash
python download_checkpoints.py --output-dir ./checkpoints --model t5
```

Download ALL models including T5:

```bash
python download_checkpoints.py --output-dir ./checkpoints --model all
```

### Force Re-download

Re-download even if checkpoints already exist:

```bash
python download_checkpoints.py --output-dir ./checkpoints --force-redownload
```

### Auto-Update Config Files

Download checkpoints and automatically update a config file:

```bash
python download_checkpoints.py \
    --output-dir /scratch/user/checkpoints \
    --update-config configs/opensora-v1-3/inference/v2v.py
```

This will update the `from_pretrained` paths in your config file with the downloaded checkpoint locations.

## How Checkpoint Loading Works

When `force_huggingface=True` is set in the config (which it is in `v2v.py`), the inference script will:

1. **First**: Check if the local path exists
2. **If not found**: Automatically download from Hugging Face Hub using the repository ID
3. **If found**: Load from the local path

This means you have two options:

### Option 1: Let it auto-download (Easier)

The code will automatically download from HuggingFace if the path doesn't exist:

```python
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",  # Will auto-download
    force_huggingface=True,
)
```

### Option 2: Pre-download checkpoints (Recommended for clusters)

Download checkpoints to your cluster storage, then point to them:

```python
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/scratch/user/checkpoints/OpenSora-STDiT-v4",
    force_huggingface=True,
)
```

## Storage Considerations

**Recommended for HPC clusters:**

1. **Download to shared storage**: Put checkpoints in a shared filesystem that all nodes can access
2. **Use scratch directory**: Download to `/scratch/$USER/checkpoints` to avoid quota issues
3. **Symlink from project**: Create symlinks from your project directory to the shared checkpoints

Example workflow:

```bash
# Download to scratch space
python download_checkpoints.py --output-dir /scratch/$USER/checkpoints

# Create symlink in project
ln -s /scratch/$USER/checkpoints/OpenSora-STDiT-v4 ./checkpoints/OpenSora-STDiT-v4
ln -s /scratch/$USER/checkpoints/OpenSora-VAE-v1.3 ./checkpoints/OpenSora-VAE-v1.3

# Update config to use relative paths
# from_pretrained="checkpoints/OpenSora-STDiT-v4"
```

## Requirements

- Python 3.7+
- `huggingface_hub` library

Install requirements:

```bash
pip install huggingface_hub
```

Or install from project requirements:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### "huggingface_hub is not installed"

Install it:

```bash
pip install huggingface_hub
```

### "Permission denied" or "Disk quota exceeded"

- Download to a location with more space (like `/scratch` or `/tmp`)
- Check available disk space: `df -h`

### "Network timeout" or "Connection error"

- Check your internet connection
- If behind a firewall, you may need to configure proxy settings
- Try downloading models individually: `--model stdit`

### "Out of memory during download"

This shouldn't happen during download (only during inference). If it does:

```bash
# Download one model at a time
python download_checkpoints.py --output-dir ./checkpoints --model stdit
python download_checkpoints.py --output-dir ./checkpoints --model vae
```

## After Downloading

Once downloaded, your checkpoints will be in:

```
your-output-dir/
├── OpenSora-STDiT-v4/          # STDiT model weights
│   ├── config.json
│   ├── model_index.json
│   ├── model.safetensors
│   └── ...
└── OpenSora-VAE-v1.3/          # VAE model weights
    ├── config.json
    ├── config.json.encoder
    ├── config.json.decoder
    ├── diffusion_pytorch_model.safetensors
    ├── diffusion_pytorch_model.encoder.safetensors
    └── ...
```

## Updating Config Files Manually

If you don't use `--update-config`, you can manually edit your config files:

**Before:**
```python
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/path/to/some/local/directory/OpenSora-STDiT-v4",
    ...
)
```

**After:**
```python
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/your/download/path/OpenSora-STDiT-v4",
    ...
)
```

## References

- [Hugging Face Model Hub](https://huggingface.co/hpcai-tech)
- [Open-Sora Repository](https://github.com/hpcaitech/Open-Sora)
- [OpenSora-STDiT-v4 Model Card](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v4)

