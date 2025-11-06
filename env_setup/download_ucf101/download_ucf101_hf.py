#!/usr/bin/env python3
"""
Download UCF-101 dataset from Hugging Face and perform stratified sampling.

This script:
1. Downloads UCF-101 from Hugging Face (no RAR extraction needed!)
2. Extracts videos to ucf101_org/ directory
3. Performs stratified sampling to select 2000 videos (~20 per class)
4. Deletes non-sampled videos to save disk space
5. Generates captions.txt for training

Advantages over RAR download:
- No unrar/unar dependency
- Automatic download management via Hugging Face Hub
- Cached downloads (resume if interrupted)
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import random
import json
import shutil

# Dataset configuration
HF_DATASET = "quchenyuan/UCF101-ZIP"
TARGET_VIDEOS = 2000
NUM_CLASSES = 101
VIDEOS_PER_CLASS = TARGET_VIDEOS // NUM_CLASSES  # ~19 videos per class


def parse_ucf101_filename(filename):
    """
    Parse UCF-101 filename to extract class name.
    Format: v_ActionName_g##_c##.avi
    
    Example: v_ApplyEyeMakeup_g01_c01.avi -> ApplyEyeMakeup
    """
    if not filename.startswith('v_'):
        return None
    
    # Remove 'v_' prefix and extension
    name_without_prefix = filename[2:]
    name_without_ext = name_without_prefix.rsplit('.', 1)[0]
    
    # Split by '_' and take all parts except last two (g##_c##)
    parts = name_without_ext.split('_')
    if len(parts) < 3:
        return None
    
    # Class name is everything except last 2 parts (g## and c##)
    class_name = '_'.join(parts[:-2])
    return class_name


def download_and_extract_ucf101(output_dir):
    """
    Download UCF-101 from Hugging Face and extract to output_dir.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ ERROR: datasets library not found!")
        print("Please install it with: pip install datasets")
        sys.exit(1)
    
    # Check for HuggingFace token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("\n⚠️  WARNING: HF_TOKEN environment variable not set.")
        print("You may encounter rate limiting without authentication.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        
        response = input("\nContinue without token? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nTo set your token:")
            print("  export HF_TOKEN='your_token_here'")
            print("Then run this script again.")
            sys.exit(0)
    
    print("\n" + "=" * 70)
    print("Downloading UCF-101 from Hugging Face")
    print("=" * 70)
    print(f"Dataset: {HF_DATASET}")
    print(f"Output directory: {output_dir}")
    print("\nThis may take a while (~7GB download)...")
    print("Downloads are cached, so you can safely interrupt and resume.\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset (downloads if not cached)
    print("Loading dataset from Hugging Face Hub...")
    try:
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        
        dataset = load_dataset(HF_DATASET, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load dataset from Hugging Face: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify HF_TOKEN is valid (if set)")
        print("3. Try: pip install --upgrade datasets huggingface-hub")
        sys.exit(1)
    
    print(f"✓ Dataset loaded! Found {len(dataset)} videos")
    
    # Extract videos to output directory
    print("\nAccessing dataset cache...")
    
    # HuggingFace datasets stores downloaded videos in a cache directory
    # Instead of iterating through the dataset (which tries to decode videos),
    # we'll access the cache directly
    
    # Get the cache directory for this dataset
    from datasets.config import HF_DATASETS_CACHE
    cache_dir = Path(HF_DATASETS_CACHE) if HF_DATASETS_CACHE else Path.home() / ".cache" / "huggingface" / "datasets"
    
    print(f"Looking for cached videos in: {cache_dir}")
    
    # Find all .avi files in the cache
    cached_videos = []
    for cache_path in cache_dir.rglob("*.avi"):
        cached_videos.append(cache_path)
    
    if not cached_videos:
        print("\n⚠️  No .avi files found in cache. Trying alternative approach...")
        print("The dataset may store videos differently. Let me try accessing the dataset structure...")
        
        # Alternative: Try to access the underlying Arrow table directly
        try:
            # Access the dataset without decoding
            dataset_path = dataset.cache_files[0]['filename'] if dataset.cache_files else None
            if dataset_path:
                print(f"Dataset cache file: {dataset_path}")
                # Look for videos near the cache file
                dataset_cache_dir = Path(dataset_path).parent
                cached_videos = list(dataset_cache_dir.rglob("*.avi"))
                print(f"Found {len(cached_videos)} videos near cache file")
        except Exception as e:
            print(f"Alternative approach failed: {e}")
    
    if not cached_videos:
        print("\n❌ ERROR: Could not find cached video files.")
        print("The UCF-101 dataset from Hugging Face may require 'torchcodec' for video decoding.")
        print("\nAlternative solution:")
        print("1. Download UCF-101 directly from official source")
        print("2. Use Option 3 in README with manual download")
        print("\nOr install torchcodec (experimental):")
        print("  pip install torchcodec")
        return False
    
    print(f"\nFound {len(cached_videos)} videos in cache")
    print("Organizing videos by class...")
    
    # Process cached videos
    for video_path in tqdm(cached_videos, desc="Organizing videos"):
        try:
            video_filename = video_path.name
            
            # Parse class name from filename
            class_name = parse_ucf101_filename(video_filename)
            if class_name is None:
                # Try to use parent directory name
                class_name = video_path.parent.name
                if class_name in ['train', 'test', 'data', 'videos']:
                    # Skip generic directory names
                    continue
            
            # Create class directory
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Copy video to organized structure
            output_path = class_dir / video_filename
            if not output_path.exists():
                shutil.copy2(video_path, output_path)
                
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to process {video_path.name}: {e}")
            continue
    
    print(f"\n✓ Videos extracted to {output_dir}")
    return True


def scan_videos(base_dir):
    """
    Scan UCF-101 directory and organize videos by class.
    Returns: dict mapping class_name -> list of video paths
    """
    base_path = Path(base_dir)
    videos_by_class = {}
    
    print("\nScanning videos...")
    
    # UCF-101 structure: ucf101_org/ClassName/*.avi or ucf101_org/*.avi
    all_videos = list(base_path.rglob('*.avi'))
    print(f"Found {len(all_videos)} total videos")
    
    for video_path in tqdm(all_videos, desc="Organizing by class"):
        # Try to get class from parent directory
        if video_path.parent != base_path:
            class_name = video_path.parent.name
        else:
            # Try to parse from filename
            class_name = parse_ucf101_filename(video_path.name)
            if class_name is None:
                print(f"\n⚠️  Warning: Could not determine class for {video_path.name}")
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
    
    # Remove empty directories
    for class_dir in base_path.iterdir():
        if class_dir.is_dir() and not list(class_dir.iterdir()):
            try:
                class_dir.rmdir()
            except:
                pass
    
    print("✓ Cleanup complete!")


def generate_captions(sampled_videos, output_file):
    """Generate captions.txt file with video-caption pairs."""
    print(f"\nGenerating {output_file}...")
    
    with open(output_file, 'w') as f:
        for video_path in sorted(sampled_videos):
            # Get class name from directory or filename
            if video_path.parent.name != 'ucf101_org':
                class_name = video_path.parent.name
            else:
                class_name = parse_ucf101_filename(video_path.name)
                if class_name is None:
                    continue
            
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
    
    print("=" * 70)
    print("UCF-101 Dataset Download from Hugging Face")
    print("=" * 70)
    print("This script downloads UCF-101 without needing unrar!")
    print("=" * 70)
    
    # Step 1: Download and extract if not already done
    if not base_dir.exists() or not list(base_dir.rglob('*.avi')):
        success = download_and_extract_ucf101(base_dir)
        if not success:
            sys.exit(1)
    else:
        print(f"✓ Found existing videos in: {base_dir}")
    
    # Step 2: Scan and organize videos
    videos_by_class = scan_videos(base_dir)
    
    if not videos_by_class:
        print("❌ ERROR: No videos found! Check extraction.")
        sys.exit(1)
    
    print(f"\n✓ Found {len(videos_by_class)} classes")
    print(f"✓ Total videos: {sum(len(v) for v in videos_by_class.values())}")
    
    # Step 3: Stratified sampling
    sampled_videos, class_stats = stratified_sampling(videos_by_class, TARGET_VIDEOS)
    
    # Step 4: Delete unsampled videos
    print("\n" + "=" * 70)
    response = input("Delete unsampled videos to save disk space? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        delete_unsampled_videos(base_dir, sampled_videos)
    else:
        print("Keeping all videos.")
    
    # Step 5: Generate captions
    captions_file = script_dir / "captions.txt"
    generate_captions(sampled_videos, captions_file)
    
    # Step 6: Save metadata
    metadata_file = script_dir / "sampling_metadata.json"
    save_sampling_metadata(class_stats, sampled_videos, metadata_file)
    
    print("\n" + "=" * 70)
    print("✓ UCF-101 download and sampling complete!")
    print("=" * 70)
    print(f"Videos: {base_dir}")
    print(f"Captions: {captions_file}")
    print(f"Metadata: {metadata_file}")
    print("\nNext step: Run preprocessing with:")
    print("  sbatch preprocess_ucf101.sbatch")


if __name__ == "__main__":
    main()

