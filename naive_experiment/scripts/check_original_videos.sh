#!/bin/bash
# Check original UCF-101 videos in ucf101_org to verify they have enough frames

ORIGINAL_DIR="/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org"

python3 << 'PYTHON_SCRIPT'
import av
from pathlib import Path

ORIGINAL_DIR = Path("/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org")

print("=== Checking ORIGINAL UCF-101 videos (.avi files) ===\n")
print(f"Directory: {ORIGINAL_DIR}\n")

if not ORIGINAL_DIR.exists():
    print(f"ERROR: Directory not found: {ORIGINAL_DIR}")
    exit(1)

print("=== Sample of 10 original .avi videos ===")
# Get sample videos from different categories
sample_videos = []
for category_dir in sorted(ORIGINAL_DIR.iterdir()):
    if category_dir.is_dir():
        videos = list(category_dir.glob("*.avi"))
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
        fps = float(video_stream.average_rate)
        container.close()
        
        print(f"{video_path.name}: {frame_count} frames, {width}x{height}, {fps:.2f} fps")
    except Exception as e:
        print(f"{video_path.name}: ERROR - {e}")

print("\n=== Frame count statistics for ALL original videos ===")
frame_counts = []
total_videos = 0
short_videos = 0

for category_dir in ORIGINAL_DIR.iterdir():
    if category_dir.is_dir():
        for video_path in category_dir.glob("*.avi"):
            try:
                container = av.open(str(video_path))
                frame_count = 0
                for frame in container.decode(video=0):
                    frame_count += 1
                container.close()
                
                frame_counts.append(frame_count)
                total_videos += 1
                
                if frame_count < 49:
                    short_videos += 1
            except:
                pass

if frame_counts:
    import statistics
    print(f"Total videos checked: {total_videos}")
    print(f"Min frames: {min(frame_counts)}")
    print(f"Max frames: {max(frame_counts)}")
    print(f"Average frames: {statistics.mean(frame_counts):.1f}")
    print(f"Median frames: {statistics.median(frame_counts):.1f}")
    print(f"\nVideos with < 49 frames: {short_videos} ({100*short_videos/total_videos:.1f}%)")
    print(f"Videos with >= 49 frames: {total_videos - short_videos} ({100*(total_videos-short_videos)/total_videos:.1f}%)")
else:
    print("No videos found or all failed to read")

print("\n" + "="*60)
print("ANALYSIS:")
if short_videos > 0:
    print(f"⚠️  {short_videos} videos are TOO SHORT (< 49 frames)")
    print("   These will be SKIPPED during preprocessing")
    print(f"   {total_videos - short_videos} videos can be processed to 49 frames")
else:
    print("✅ ALL videos have >= 49 frames - can be processed!")
print("="*60)

PYTHON_SCRIPT

