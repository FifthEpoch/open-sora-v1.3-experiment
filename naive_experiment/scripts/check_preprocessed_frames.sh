#!/bin/bash
# Check the frame count of preprocessed UCF-101 videos using Python/PyAV

PROCESSED_DIR="/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_processed"

# Use Python with PyAV (already installed in opensora13 environment)
python3 << 'PYTHON_SCRIPT'
import av
from pathlib import Path

PROCESSED_DIR = Path("/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_processed")

print("=== Checking preprocessed UCF-101 videos ===\n")
print(f"Directory: {PROCESSED_DIR}\n")

if not PROCESSED_DIR.exists():
    print(f"ERROR: Directory not found: {PROCESSED_DIR}")
    exit(1)

print("=== Sample of 10 preprocessed videos ===")
# Get sample videos from different categories (exclude truncated_for_training subdirs)
sample_videos = []
for category_dir in sorted(PROCESSED_DIR.iterdir()):
    if category_dir.is_dir():
        videos = [v for v in category_dir.glob("*.mp4")]
        if videos:
            sample_videos.extend(videos[:2])  # 2 videos per category
            if len(sample_videos) >= 10:
                break

for video_path in sample_videos[:10]:
    try:
        container = av.open(str(video_path))
        frame_count = 0
        for frame in container.decode(video=0):
            frame_count += 1
        
        video_stream = container.streams.video[0]
        width = video_stream.width
        height = video_stream.height
        container.close()
        
        print(f"{video_path.name}: {frame_count} frames, {width}x{height}")
    except Exception as e:
        print(f"{video_path.name}: ERROR - {e}")

print("\n=== Checking 5 sample conditioning videos (should be 22 frames) ===")
# Check truncated_for_training subdirectories
sample_cond = []
for category_dir in PROCESSED_DIR.iterdir():
    if category_dir.is_dir():
        trunc_dir = category_dir / "truncated_for_training"
        if trunc_dir.exists():
            sample_cond.extend(list(trunc_dir.glob("*.mp4"))[:2])
            if len(sample_cond) >= 5:
                break

for video_path in sample_cond[:5]:
    try:
        container = av.open(str(video_path))
        frame_count = 0
        for frame in container.decode(video=0):
            frame_count += 1
        
        video_stream = container.streams.video[0]
        width = video_stream.width
        height = video_stream.height
        container.close()
        
        print(f"{video_path.name}: {frame_count} frames, {width}x{height}")
    except Exception as e:
        print(f"{video_path.name}: ERROR - {e}")

print("\n" + "="*60)
print("RESULT:")
print("Check the frame counts above:")
print("  - If sample shows 45 frames: ❌ NEED TO RE-PREPROCESS")
print("    Run: cd env_setup/download_ucf101 && sbatch preprocess_ucf101.sbatch")
print("  - If sample shows 49+ frames: ✅ READY TO RUN EXPERIMENT!")
print("    Run: cd naive_experiment/scripts && sbatch run_experiment.sbatch")
print("="*60)

PYTHON_SCRIPT
