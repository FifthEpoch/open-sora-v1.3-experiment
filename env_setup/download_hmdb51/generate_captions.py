#!/usr/bin/env python3
"""
Generate captions for HMDB51 videos based on folder structure.

This script scans the hmdb51_org directory and generates a captions.txt file
with video-caption pairs based on the action category folder names.

Usage:
    python generate_captions.py
    python generate_captions.py --input /path/to/hmdb51_org
"""

import argparse
import sys
from pathlib import Path


def generate_captions(hmdb51_org_dir: str, output_dir: str = None):
    """
    Generate simple captions for HMDB51 videos based on their folder names.
    Each video gets a caption based on the action category it belongs to.
    
    Args:
        hmdb51_org_dir: Path to the hmdb51_org directory containing videos
        output_dir: Directory to write captions.txt (defaults to parent of hmdb51_org)
    """
    hmdb51_org = Path(hmdb51_org_dir)
    
    if not hmdb51_org.exists():
        print(f"ERROR: Directory {hmdb51_org} does not exist!")
        sys.exit(1)
    
    if not hmdb51_org.is_dir():
        print(f"ERROR: {hmdb51_org} is not a directory!")
        sys.exit(1)
    
    # Determine output directory
    if output_dir is None:
        # Default: write to parent of hmdb51_org directory
        output_dir = hmdb51_org.parent
    else:
        output_dir = Path(output_dir)
    
    captions = []
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']  # Support common video formats
    
    # Iterate through all class folders
    print(f"Scanning directory: {hmdb51_org}")
    class_folders = sorted([f for f in hmdb51_org.iterdir() if f.is_dir()])
    print(f"Found {len(class_folders)} action categories")
    
    for class_folder in class_folders:
        action_name = class_folder.name
        
        # Find all video files in this class (try multiple extensions)
        videos_found = False
        for ext in video_extensions:
            for video_file in sorted(class_folder.glob(f'*{ext}')):
                videos_found = True
                captions.append((video_file, action_name))
        
        if videos_found:
            print(f"  {action_name}: {len([v for v, _ in captions if v.parent == class_folder])} videos")
    
    # Write captions to a file
    captions_file = output_dir / 'captions.txt'
    with open(captions_file, 'w') as f:
        for video_file, caption in captions:
            # Use relative path from output directory
            rel_path = video_file.relative_to(output_dir)
            f.write(f"{rel_path}\t{caption}\n")
    
    print(f"\nGenerated captions for {len(captions)} videos in {captions_file}")
    
    # Print summary statistics
    action_counts = {}
    for _, action_name in captions:
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
    
    print("\nSummary by action category:")
    for action_name, count in sorted(action_counts.items()):
        print(f"  {action_name}: {count} videos")
    
    return captions_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for HMDB51 videos based on folder structure"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_hmdb51/hmdb51_org',
        help='Path to hmdb51_org directory (default: /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_hmdb51/hmdb51_org)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Directory to write captions.txt (default: parent of input directory)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HMDB51 Caption Generation")
    print("=" * 60)
    print()
    
    captions_file = generate_captions(args.input, args.output)
    
    print("\n" + "=" * 60)
    print("Caption generation complete!")
    print("=" * 60)
    print(f"\nOutput: {captions_file}")
    print("\nNext step: Run preprocess_hmdb51.py to prepare for training")


if __name__ == "__main__":
    main()

