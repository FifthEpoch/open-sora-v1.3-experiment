#!/usr/bin/env python3
"""
Download and prepare UCF-101 dataset with stratified sampling.

This script:
1. Downloads UCF-101 dataset from official CRCV source
2. Extracts videos to ucf101_org/ directory
3. Performs stratified sampling to select 2000 videos (~20 per class)
4. Deletes non-sampled videos to save disk space
5. Generates captions.txt for training
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import json

# Dataset configuration
UCF101_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
DATASET_NAME = "UCF101.rar"
TARGET_VIDEOS = 2000
NUM_CLASSES = 101
VIDEOS_PER_CLASS = TARGET_VIDEOS // NUM_CLASSES  # ~19 videos per class


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    file_size = os.path.getsize(output_path)
    print(f"✓ Download complete! File size: {file_size / (1024**3):.2f} GB")
    
    # Check if file is too small (likely corrupted or HTML error page)
    if file_size < 1_000_000:  # Less than 1MB is suspicious for UCF-101
        print(f"\n⚠️  WARNING: Downloaded file is only {file_size / 1024:.1f} KB")
        print("This is likely an error page or corrupted download.")
        print("\nPlease download UCF-101 manually from:")
        print("  https://www.crcv.ucf.edu/research/data-sets/ucf101/")
        print("\nLook for 'UCF101.rar' (~6.5GB) and save it to:")
        print(f"  {output_path}")
        os.remove(output_path)
        sys.exit(1)


def extract_rar(rar_path, extract_to):
    """Extract RAR archive."""
    print(f"\nExtracting {rar_path}...")
    
    # Try different extraction tools
    tools = []
    
    # Check for rarfile Python library
    try:
        import rarfile
        print("Using Python rarfile library...")
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(extract_to)
        print("✓ Extraction complete using rarfile!")
        return True
    except ImportError:
        tools.append("rarfile (pip install rarfile)")
    except Exception as e:
        print(f"rarfile failed: {e}")
    
    # Check for unrar command
    if shutil.which('unrar'):
        print("Using unrar command...")
        try:
            import subprocess
            result = subprocess.run(['unrar', 'x', '-y', str(rar_path), str(extract_to)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Extraction complete using unrar!")
                return True
            else:
                print(f"unrar failed: {result.stderr}")
        except Exception as e:
            print(f"unrar failed: {e}")
    else:
        tools.append("unrar")
    
    # Check for unar command
    if shutil.which('unar'):
        print("Using unar command...")
        try:
            import subprocess
            result = subprocess.run(['unar', '-o', str(extract_to), str(rar_path)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Extraction complete using unar!")
                return True
            else:
                print(f"unar failed: {result.stderr}")
        except Exception as e:
            print(f"unar failed: {e}")
    else:
        tools.append("unar")
    
    # If we get here, all methods failed
    print("\n❌ ERROR: Could not extract RAR file. None of the following tools are available:")
    for tool in tools:
        print(f"  - {tool}")
    print("\nPlease install one of:")
    print("  pip install rarfile  # Python library (recommended)")
    print("  brew install unrar   # macOS")
    print("  sudo apt-get install unrar  # Linux")
    print("\nOr manually extract UCF101.rar to: ucf101_org/")
    return False


def parse_ucf101_filename(filename):
    """
    Parse UCF-101 filename to extract class name.
    Format: v_ActionName_g##_c##.avi
    
    Example: v_ApplyEyeMakeup_g01_c01.avi -> ApplyEyeMakeup
    """
    if not filename.startswith('v_'):
        return None
    
    # Remove 'v_' prefix and '.avi' extension
    name_without_prefix = filename[2:]  # Remove 'v_'
    name_without_ext = name_without_prefix.rsplit('.', 1)[0]  # Remove extension
    
    # Split by '_' and take all parts except last two (g##_c##)
    parts = name_without_ext.split('_')
    if len(parts) < 3:
        return None
    
    # Class name is everything except last 2 parts (g## and c##)
    class_name = '_'.join(parts[:-2])
    return class_name


def scan_videos(base_dir):
    """
    Scan UCF-101 directory and organize videos by class.
    Returns: dict mapping class_name -> list of video paths
    """
    base_path = Path(base_dir)
    videos_by_class = {}
    
    print("\nScanning videos...")
    
    # UCF-101 structure can be:
    # Option 1: ucf101_org/v_ActionName_g##_c##.avi (flat structure after extraction)
    # Option 2: ucf101_org/UCF-101/ActionName/*.avi (folder structure)
    
    all_videos = list(base_path.rglob('*.avi'))
    print(f"Found {len(all_videos)} total videos")
    
    for video_path in tqdm(all_videos, desc="Parsing filenames"):
        video_name = video_path.name
        
        # Try to get class from filename
        class_name = parse_ucf101_filename(video_name)
        
        # If filename parsing fails, try parent directory name
        if class_name is None:
            class_name = video_path.parent.name
            if class_name == 'ucf101_org' or class_name == 'UCF-101':
                continue
        
        if class_name not in videos_by_class:
            videos_by_class[class_name] = []
        
        videos_by_class[class_name].append(video_path)
    
    return videos_by_class


def stratified_sampling(videos_by_class, target_total=2000):
    """
    Perform stratified sampling to select ~target_total videos.
    Ensures approximately equal representation from each class.
    """
    num_classes = len(videos_by_class)
    base_samples_per_class = target_total // num_classes
    
    print(f"\nPerforming stratified sampling:")
    print(f"  Target total: {target_total} videos")
    print(f"  Number of classes: {num_classes}")
    print(f"  Base samples per class: {base_samples_per_class}")
    
    sampled_videos = []
    class_stats = []
    
    # First pass: sample base amount from each class
    for class_name, videos in videos_by_class.items():
        if len(videos) <= base_samples_per_class:
            # Take all videos if class has fewer than target
            selected = videos
        else:
            # Random sample
            selected = random.sample(videos, base_samples_per_class)
        
        sampled_videos.extend(selected)
        class_stats.append({
            'class': class_name,
            'total': len(videos),
            'sampled': len(selected)
        })
    
    # Second pass: if we haven't reached target, sample more from larger classes
    remaining = target_total - len(sampled_videos)
    if remaining > 0:
        print(f"  Need {remaining} more videos to reach target...")
        
        # Get classes that have more videos available
        available_classes = [
            (class_name, videos) 
            for class_name, videos in videos_by_class.items() 
            if len(videos) > base_samples_per_class
        ]
        
        if available_classes:
            for _ in range(remaining):
                if not available_classes:
                    break
                
                # Pick a random class and sample one more video
                class_name, videos = random.choice(available_classes)
                
                # Find videos not yet sampled
                already_sampled = [v for v in sampled_videos if v.parent.name == class_name or parse_ucf101_filename(v.name) == class_name]
                available = [v for v in videos if v not in already_sampled]
                
                if available:
                    sampled_videos.append(random.choice(available))
                    
                    # Update stats
                    for stat in class_stats:
                        if stat['class'] == class_name:
                            stat['sampled'] += 1
                            break
    
    # Print sampling summary
    print(f"\n✓ Sampled {len(sampled_videos)} videos total")
    print(f"\nSampling distribution (first 10 classes):")
    for stat in sorted(class_stats, key=lambda x: x['class'])[:10]:
        print(f"  {stat['class']:<30} {stat['sampled']:>3} / {stat['total']:>3}")
    print(f"  ... ({len(class_stats) - 10} more classes)")
    
    return sampled_videos, class_stats


def delete_unsampled_videos(base_dir, sampled_videos):
    """Delete videos that were not sampled to save disk space."""
    base_path = Path(base_dir)
    all_videos = list(base_path.rglob('*.avi'))
    sampled_set = set(sampled_videos)
    
    to_delete = [v for v in all_videos if v not in sampled_set]
    
    print(f"\nDeleting {len(to_delete)} unsampled videos to save space...")
    
    for video_path in tqdm(to_delete, desc="Deleting"):
        try:
            video_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete {video_path}: {e}")
    
    print("✓ Cleanup complete!")


def generate_captions(sampled_videos, output_file):
    """Generate captions.txt file with video-caption pairs."""
    print(f"\nGenerating {output_file}...")
    
    with open(output_file, 'w') as f:
        for video_path in sorted(sampled_videos):
            # Get class name from filename or directory
            class_name = parse_ucf101_filename(video_path.name)
            if class_name is None:
                class_name = video_path.parent.name
            
            # Convert CamelCase to space-separated (e.g., ApplyEyeMakeup -> apply eye makeup)
            caption = ''.join([' ' + c.lower() if c.isupper() else c for c in class_name]).strip()
            caption = caption.replace('_', ' ')
            
            # Write as tab-separated
            f.write(f"{video_path}\t{caption}\n")
    
    print(f"✓ Generated {output_file} with {len(sampled_videos)} entries")


def save_sampling_metadata(class_stats, sampled_videos, output_file):
    """Save sampling metadata for reproducibility."""
    metadata = {
        'total_sampled': len(sampled_videos),
        'num_classes': len(class_stats),
        'class_distribution': class_stats,
        'sampled_videos': [str(v) for v in sampled_videos]
    }
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved sampling metadata to {output_file}")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get script directory
    script_dir = Path(__file__).parent
    base_dir = script_dir / "ucf101_org"
    rar_path = script_dir / DATASET_NAME
    
    print("=" * 70)
    print("UCF-101 Dataset Download and Sampling")
    print("=" * 70)
    
    # Step 1: Download if not exists
    if not rar_path.exists():
        download_file(UCF101_URL, rar_path)
    else:
        print(f"✓ Found existing download: {rar_path}")
    
    # Step 2: Extract if not already extracted
    if not base_dir.exists() or not list(base_dir.rglob('*.avi')):
        base_dir.mkdir(parents=True, exist_ok=True)
        success = extract_rar(rar_path, base_dir)
        if not success:
            print("\n⚠️  Extraction failed. Please extract manually and re-run this script.")
            sys.exit(1)
    else:
        print(f"✓ Found existing videos in: {base_dir}")
    
    # Step 3: Scan and organize videos
    videos_by_class = scan_videos(base_dir)
    
    if not videos_by_class:
        print("❌ ERROR: No videos found! Check extraction.")
        sys.exit(1)
    
    print(f"\n✓ Found {len(videos_by_class)} classes")
    print(f"✓ Total videos: {sum(len(v) for v in videos_by_class.values())}")
    
    # Step 4: Stratified sampling
    sampled_videos, class_stats = stratified_sampling(videos_by_class, TARGET_VIDEOS)
    
    # Step 5: Delete unsampled videos
    print("\n" + "=" * 70)
    response = input("Delete unsampled videos to save disk space? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        delete_unsampled_videos(base_dir, sampled_videos)
    else:
        print("Keeping all videos.")
    
    # Step 6: Generate captions
    captions_file = script_dir / "captions.txt"
    generate_captions(sampled_videos, captions_file)
    
    # Step 7: Save metadata
    metadata_file = script_dir / "sampling_metadata.json"
    save_sampling_metadata(class_stats, sampled_videos, metadata_file)
    
    print("\n" + "=" * 70)
    print("✓ UCF-101 download and sampling complete!")
    print("=" * 70)
    print(f"Videos: {base_dir}")
    print(f"Captions: {captions_file}")
    print(f"Metadata: {metadata_file}")
    print("\nNext step: Run preprocessing with:")
    print("  python preprocess_ucf101.py")


if __name__ == "__main__":
    main()

