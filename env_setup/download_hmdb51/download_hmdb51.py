#!/usr/bin/env python3
"""
Download and extract the HMDB51 action recognition dataset.
HMDB51 contains 51 action categories with video clips for each action.

Dataset URL: https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
"""

import os
import sys
import urllib.request
import subprocess
from pathlib import Path


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress indication."""
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False


def download_progress(block_num, block_size, total_size):
    """Report download progress."""
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100)
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = '=' * filled + '-' * (bar_length - filled)
    print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)


def extract_rar(rar_path: str, dest_dir: str) -> bool:
    """Extract RAR archive using unrar or alternative tools."""
    print(f"\nExtracting {rar_path} to {dest_dir}...")
    
    # Try unrar first
    if subprocess.run(['which', 'unrar'], capture_output=True).returncode == 0:
        cmd = ['unrar', 'x', rar_path, dest_dir]
        try:
            subprocess.run(cmd, check=True)
            print("Extraction completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting with unrar: {e}")
    
    # Try unar (macOS)
    if subprocess.run(['which', 'unar'], capture_output=True).returncode == 0:
        cmd = ['unar', '-o', dest_dir, rar_path]
        try:
            subprocess.run(cmd, check=True)
            print("Extraction completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting with unar: {e}")
    
    # Try 7z
    if subprocess.run(['which', '7z'], capture_output=True).returncode == 0:
        cmd = ['7z', 'x', rar_path, f'-o{dest_dir}']
        try:
            subprocess.run(cmd, check=True)
            print("Extraction completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting with 7z: {e}")
    
    print("ERROR: No extraction tool found. Please install one of:")
    print("  - unrar: sudo apt-get install unrar (Linux) or brew install unrar (macOS)")
    print("  - unar: brew install unar (macOS)")
    print("  - p7zip: sudo apt-get install p7zip-full (Linux) or brew install p7zip (macOS)")
    return False


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
    
    print(f"\nGenerated captions for {len(captions)} videos in {captions_file}")


def main():
    # Set working directory to script location
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print("=" * 60)
    print("HMDB51 Dataset Download Script")
    print("=" * 60)
    
    url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    rar_file = script_dir / "hmdb51_org.rar"
    
    # Check if already downloaded
    if rar_file.exists():
        print(f"\nFound existing file: {rar_file}")
        response = input("Re-download? (y/n): ").strip().lower()
        if response == 'y':
            rar_file.unlink()
        else:
            print("Using existing file.")
    
    # Download if needed
    if not rar_file.exists():
        if not download_file(url, str(rar_file)):
            print("Download failed!")
            sys.exit(1)
    
    # Extract
    if not extract_rar(str(rar_file), str(script_dir)):
        print("Extraction failed!")
        sys.exit(1)
    
    # Generate captions
    generate_captions(str(script_dir))
    
    print("\n" + "=" * 60)
    print("HMDB51 dataset preparation complete!")
    print("=" * 60)
    print("\nDataset structure:")
    print("  hmdb51/")
    print("    hmdb51_org/           # Original HMDB51 videos by action class")
    print("    captions.txt          # Video-caption pairs")
    print("    hmdb51_org.rar        # Original archive (can be deleted)")
    print("\nYou can delete hmdb51_org.rar to save space.")


if __name__ == "__main__":
    main()

