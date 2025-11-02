#!/usr/bin/env python3
"""
Download HMDB51 dataset using Hugging Face Datasets (no RAR extraction needed!).

This script downloads HMDB51 from Hugging Face Datasets which provides it
pre-extracted, avoiding the need for unrar or other RAR extraction tools.

Dataset: https://huggingface.co/datasets/divm/hmdb51
"""

import os
import sys
import shutil
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not found!")
    print("Please install it with: pip install datasets")
    sys.exit(1)


def main():
    # Set working directory to script location
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print("=" * 60)
    print("HMDB51 Dataset Download (Hugging Face)")
    print("=" * 60)
    
    # Check if already downloaded
    hmdb51_org = script_dir / "hmdb51_org"
    if hmdb51_org.exists():
        print(f"\nFound existing directory: {hmdb51_org}")
        response = input("Re-download? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(hmdb51_org)
        else:
            print("Using existing directory.")
            return
    
    # Download from Hugging Face
    print("\nDownloading HMDB51 from Hugging Face Datasets...")
    print("This may take a while (~2GB to download)...")
    
    try:
        # Load dataset - this will download if not cached
        ds = load_dataset("divm/hmdb51", split="all", download_mode="force_redownload")
        
        # The dataset structure from HF may be different, we need to reorganize it
        # to match the expected structure (action_class/video.avi)
        print("\nReorganizing dataset structure...")
        
        # Create output directory
        hmdb51_org.mkdir(exist_ok=True)
        
        # Process each video
        num_videos = 0
        for example in ds:
            video_path = example['video']
            label = example['label']
            action_name = ds.features['label'].int2str(label)
            
            # Create action class directory
            action_dir = hmdb51_org / action_name
            action_dir.mkdir(exist_ok=True)
            
            # Copy video to action directory
            video_filename = Path(video_path).name
            dest_path = action_dir / video_filename
            shutil.copy2(video_path, dest_path)
            num_videos += 1
            
            if num_videos % 100 == 0:
                print(f"Processed {num_videos} videos...")
        
        print(f"\nDownloaded {num_videos} videos from Hugging Face")
        
    except Exception as e:
        print(f"\nError downloading from Hugging Face: {e}")
        print("Falling back to manual download instructions...")
        sys.exit(1)
    
    # Generate captions
    print("\nGenerating captions...")
    generate_captions(str(script_dir))
    
    print("\n" + "=" * 60)
    print("HMDB51 dataset preparation complete!")
    print("=" * 60)
    print("\nDataset structure:")
    print("  env_setup/download_hmdb51/")
    print("    hmdb51_org/           # HMDB51 videos by action class")
    print("    captions.txt          # Video-caption pairs")
    print("\nNext step: Run preprocess_hmdb51.py to prepare for training")


def generate_captions(base_dir: str):
    """
    Generate simple captions for HMDB51 videos based on their folder names.
    Each video gets a caption based on the action category it belongs to.
    """
    base_path = Path(base_dir)
    hmdb51_org = base_path / 'hmdb51_org'
    
    if not hmdb51_org.exists():
        print(f"Warning: {hmdb51_org} does not exist, skipping caption generation.")
        return
    
    captions = []
    
    # Iterate through all class folders
    for class_folder in sorted(hmdb51_org.iterdir()):
        if not class_folder.is_dir():
            continue
        
        action_name = class_folder.name
        
        # Find all video files in this class
        for video_file in sorted(class_folder.glob('*.avi')):
            video_name = video_file.stem
            caption = f"{action_name}"
            captions.append((video_file, caption))
    
    # Write captions to a file
    captions_file = base_path / 'captions.txt'
    with open(captions_file, 'w') as f:
        for video_file, caption in captions:
            # Use relative path from hmdb51 directory
            rel_path = video_file.relative_to(base_path)
            f.write(f"{rel_path}\t{caption}\n")
    
    print(f"Generated captions for {len(captions)} videos in {captions_file}")


if __name__ == "__main__":
    main()

