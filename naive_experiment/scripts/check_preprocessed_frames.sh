#!/bin/bash
# Check the frame count of preprocessed UCF-101 videos

echo "=== Checking preprocessed UCF-101 videos ==="
echo ""

PROCESSED_DIR="/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_processed"

if [ ! -d "$PROCESSED_DIR" ]; then
    echo "ERROR: Directory not found: $PROCESSED_DIR"
    exit 1
fi

echo "Directory: $PROCESSED_DIR"
echo ""

# Check first 10 videos (excluding conditioning videos with "first22frames" in name)
echo "=== Sample of first 10 videos ==="
find "$PROCESSED_DIR" -name "*.mp4" ! -name "*first22frames*" | sort | head -10 | while read video; do
    filename=$(basename "$video")
    frames=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$video" 2>/dev/null)
    dims=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$video" 2>/dev/null)
    echo "$filename: $frames frames, ${dims}"
done

echo ""
echo "=== Frame count distribution ==="
# Count how many videos have each frame count (excluding conditioning videos)
find "$PROCESSED_DIR" -name "*.mp4" ! -name "*first22frames*" | while read video; do
    ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$video" 2>/dev/null
done | sort | uniq -c | sort -rn

echo ""
echo "=== Total preprocessed videos (excluding conditioning) ==="
find "$PROCESSED_DIR" -name "*.mp4" ! -name "*first22frames*" | wc -l

echo ""
echo "=== Total conditioning videos ==="
find "$PROCESSED_DIR" -name "*first22frames*.mp4" | wc -l

