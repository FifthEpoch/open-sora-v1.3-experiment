#!/usr/bin/env python3
"""
Download Open-Sora checkpoints from Hugging Face Hub.

This script downloads pre-trained models for Open-Sora inference:
- OpenSora-STDiT-v4 (STDiT3-XL/2 model weights) - ~1GB
- OpenSora-VAE-v1.3 (VAE model weights) - ~330MB
- T5-v1.1-XXL (text encoder) - ~20GB [optional]

Usage:
    # Download essential models (STDiT + VAE)
    python download_checkpoints.py --output-dir /path/to/checkpoints
    
    # Download specific models
    python download_checkpoints.py --output-dir /path/to/checkpoints --model stdit
    python download_checkpoints.py --output-dir /path/to/checkpoints --model all
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    exit(1)


CHECKPOINT_MAP = {
    "stdit": {
        "repo_id": "hpcai-tech/OpenSora-STDiT-v4",
        "description": "STDiT3-XL/2 model weights",
        "expected_size": "~1GB",
    },
    "vae": {
        "repo_id": "hpcai-tech/OpenSora-VAE-v1.3",
        "description": "VAE model weights",
        "expected_size": "~330MB",
    },
    "t5": {
        "repo_id": "google/t5-v1_1-xxl",
        "description": "T5 XXL text encoder",
        "expected_size": "~20GB",
    },
}


def format_bytes(bytes_size):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_directory_size(path):
    """Calculate total size of a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def download_checkpoint(model_name: str, output_dir: Path, force_redownload: bool = False):
    """
    Download a checkpoint from Hugging Face Hub.
    
    Args:
        model_name: Name of the model ('stdit' or 'vae')
        output_dir: Directory to save the checkpoint
        force_redownload: Force re-download even if checkpoint exists
        
    Returns:
        Path to the downloaded checkpoint directory
    """
    if model_name not in CHECKPOINT_MAP:
        raise ValueError(f"Unknown model: {model_name}. Must be one of: {list(CHECKPOINT_MAP.keys())}")
    
    checkpoint_info = CHECKPOINT_MAP[model_name]
    repo_id = checkpoint_info["repo_id"]
    model_dir = output_dir / repo_id.split("/")[1]
    
    # Check if already downloaded
    if model_dir.exists() and not force_redownload:
        print(f"\n✓ {model_name.upper()} checkpoint already exists at:")
        print(f"  {model_dir}")
        print(f"  Size: {format_bytes(get_directory_size(model_dir))}")
        print("  (Use --force-redownload to re-download)")
        return model_dir
    
    # Download the checkpoint
    print(f"\n{'='*70}")
    print(f"Downloading {model_name.upper()} checkpoint")
    print(f"{'='*70}")
    print(f"Repository: {repo_id}")
    print(f"Description: {checkpoint_info['description']}")
    print(f"Expected size: {checkpoint_info['expected_size']}")
    print(f"Destination: {model_dir}")
    print("\nThis may take a while depending on your internet connection...")
    
    try:
        # Use snapshot_download to get the entire repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume interrupted downloads
        )
        
        actual_size = get_directory_size(model_dir)
        print(f"\n✓ {model_name.upper()} checkpoint downloaded successfully!")
        print(f"  Location: {model_dir}")
        print(f"  Size: {format_bytes(actual_size)}")
        
        return Path(downloaded_path)
        
    except Exception as e:
        print(f"\n✗ Error downloading {model_name.upper()} checkpoint: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Make sure you have enough disk space")
        print("  3. Try using --force-redownload to clear cached files")
        raise


def update_config_file(config_path: Path, stdit_path: Path = None, vae_path: Path = None, t5_path: Path = None):
    """
    Update a config file to use the downloaded checkpoints.
    
    Args:
        config_path: Path to the config file to update
        stdit_path: Path to the STDiT checkpoint
        vae_path: Path to the VAE checkpoint
    """
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return
    
    print(f"\n{'='*70}")
    print("Updating config file")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    
    # Read the config file
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace the paths
    original_content = content
    if stdit_path:
        content = content.replace(
            'from_pretrained="/home/guoxinying/open_source_video_ocean_V1/OpenSora-STDiT-v4"',
            f'from_pretrained="{stdit_path}"'
        )
    if vae_path:
        content = content.replace(
            'from_pretrained="/home/guoxinying/open_source_video_ocean_V1/OpenSora-VAE-v1.3"',
            f'from_pretrained="{vae_path}"'
        )
    if t5_path:
        content = content.replace(
            'from_pretrained="/mnt/jfs-hdd/sora/checkpoints/pretrained_models/t5-v1_1-xxl"',
            f'from_pretrained="{t5_path}"'
        )
    
    if content != original_content:
        # Backup the original file
        backup_path = config_path.with_suffix('.py.backup')
        shutil.copy2(config_path, backup_path)
        print(f"  Created backup: {backup_path}")
        
        # Write the updated content
        with open(config_path, 'w') as f:
            f.write(content)
        
        print(f"  Updated {config_path}")
        print("\n✓ Config file updated successfully!")
    else:
        print("  No changes needed (paths already correct)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Open-Sora checkpoints from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all checkpoints to a directory
    python download_checkpoints.py --output-dir /scratch/user/checkpoints
    
    # Download only STDiT model
    python download_checkpoints.py --output-dir ./checkpoints --model stdit
    
    # Force re-download all models
    python download_checkpoints.py --output-dir ./checkpoints --force-redownload
    
    # Download and update a specific config file
    python download_checkpoints.py --output-dir ./checkpoints --update-config configs/opensora-v1-3/inference/v2v.py
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where checkpoints will be downloaded"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["stdit", "vae", "t5", "all"],
        default=None,
        help="Download only a specific model: stdit, vae, t5, or all (default: stdit+vae only, not t5)"
    )
    
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download even if checkpoint already exists"
    )
    
    parser.add_argument(
        "--update-config",
        type=str,
        default=None,
        help="Update this config file with new checkpoint paths"
    )
    
    args = parser.parse_args()
    
    # Parse output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    print("="*70)
    print("Open-Sora Checkpoint Downloader")
    print("="*70)
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to download
    if args.model == "all":
        models_to_download = ["stdit", "vae", "t5"]
    elif args.model:
        models_to_download = [args.model]
    else:
        # Default: download stdit and vae, but not t5 (too large for many users)
        models_to_download = ["stdit", "vae"]
    
    # Download checkpoints
    downloaded_paths = {}
    for model_name in models_to_download:
        try:
            model_path = download_checkpoint(model_name, output_dir, args.force_redownload)
            downloaded_paths[model_name] = model_path
        except Exception as e:
            print(f"\n✗ Failed to download {model_name}: {e}")
            return 1
    
    # Update config file if requested
    if args.update_config:
        config_path = Path(args.update_config).expanduser().resolve()
        update_config_file(
            config_path,
            stdit_path=downloaded_paths.get("stdit"),
            vae_path=downloaded_paths.get("vae"),
            t5_path=downloaded_paths.get("t5")
        )
    
    # Print summary
    print("\n" + "="*70)
    print("Download Complete!")
    print("="*70)
    print("\nDownloaded checkpoints:")
    for model_name, model_path in downloaded_paths.items():
        checkpoint_info = CHECKPOINT_MAP[model_name]
        print(f"  {model_name.upper()}: {model_path}")
        print(f"    {checkpoint_info['description']}")
    
    print("\nTo use these checkpoints in your config files, update the 'from_pretrained' paths:")
    print("\nFor STDiT model:")
    print(f"  from_pretrained=\"{downloaded_paths.get('stdit', '')}\"")
    print("\nFor VAE model:")
    print(f"  from_pretrained=\"{downloaded_paths.get('vae', '')}\"")
    if "t5" in downloaded_paths:
        print("\nFor T5 model:")
        print(f"  from_pretrained=\"{downloaded_paths.get('t5', '')}\"")
    
    print("\nYou can also use the --update-config flag to automatically update a config file.")
    
    return 0


if __name__ == "__main__":
    exit(main())

# Example usage:
# python download_checkpoints.py --output-dir ./checkpoints \
#   --update-config configs/opensora-v1-3/inference/v2v.py

