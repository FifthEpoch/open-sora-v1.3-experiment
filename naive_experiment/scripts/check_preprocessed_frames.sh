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

print("\n=== Frame count distribution ===")
# Count all preprocessed videos
frame_counts = {}
total_videos = 0

for category_dir in PROCESSED_DIR.iterdir():
    if category_dir.is_dir():
        for video_path in category_dir.glob("*.mp4"):
            try:
                container = av.open(str(video_path))
                frame_count = 0
                for frame in container.decode(video=0):
                    frame_count += 1
                container.close()
                
                frame_counts[frame_count] = frame_counts.get(frame_count, 0) + 1
                total_videos += 1
            except:
                pass

# Print distribution sorted by count (descending)
for frame_count in sorted(frame_counts.keys(), key=lambda x: frame_counts[x], reverse=True):
    print(f"{frame_counts[frame_count]:4d} videos with {frame_count} frames")

print(f"\n=== Total preprocessed videos ===")
print(f"{total_videos}")

print("\n=== Checking conditioning videos (should be 22 frames) ===")
# Check truncated_for_training subdirectories
cond_count = 0
sample_cond_checked = 0
for category_dir in PROCESSED_DIR.iterdir():
    if category_dir.is_dir():
        trunc_dir = category_dir / "truncated_for_training"
        if trunc_dir.exists():
            for video_path in trunc_dir.glob("*.mp4"):
                cond_count += 1
                if sample_cond_checked < 5:
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
                        sample_cond_checked += 1
                    except Exception as e:
                        print(f"{video_path.name}: ERROR - {e}")

print(f"\nTotal conditioning videos: {cond_count}")

print("\n" + "="*60)
print("RESULT:")
if 45 in frame_counts and frame_counts[45] > 0:
    print("❌ Videos have 45 frames - NEED TO RE-PREPROCESS")
    print("   Run: cd env_setup/download_ucf101 && sbatch preprocess_ucf101.sbatch")
elif any(fc >= 49 for fc in frame_counts.keys()):
    print("✅ Videos have 49+ frames - READY TO RUN EXPERIMENT!")
    print("   Run: cd naive_experiment/scripts && sbatch run_experiment.sbatch")
else:
    print("⚠️  Unexpected frame counts detected - please review output above")
print("="*60)

PYTHON_SCRIPT
