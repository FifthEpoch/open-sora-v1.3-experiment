#!/bin/bash
# Script to verify all video dimensions in the experiment

echo "========================================"
echo "Dimension Verification Script"
echo "========================================"
echo ""

# Check a sample preprocessed video
echo "1. Checking preprocessed videos (main dataset):"
SAMPLE_VIDEO="env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4"
if [ -f "$SAMPLE_VIDEO" ]; then
    ffprobe "$SAMPLE_VIDEO" 2>&1 | grep -E "Stream.*Video" | grep -o "[0-9]*x[0-9]*" | head -1
    echo "  Expected: 544x416"
else
    echo "  ❌ Sample video not found"
fi
echo ""

# Check truncated training videos
echo "2. Checking truncated training videos (first 22 frames):"
TRUNCATED_VIDEO=$(find env_setup/download_ucf101 -name "*_first22frames.mp4" | head -1)
if [ -n "$TRUNCATED_VIDEO" ]; then
    echo "  Found: $TRUNCATED_VIDEO"
    ffprobe "$TRUNCATED_VIDEO" 2>&1 | grep -E "Stream.*Video" | grep -o "[0-9]*x[0-9]*" | head -1
    echo "  Expected: 544x416"
else
    echo "  ℹ️  No truncated videos found yet"
fi
echo ""

# Check conditioning videos (if any exist in results)
echo "3. Checking conditioning videos (if created):"
COND_VIDEO=$(find naive_experiment/scripts/results -name "*_cond_*frames.mp4" 2>/dev/null | head -1)
if [ -n "$COND_VIDEO" ]; then
    echo "  Found: $COND_VIDEO"
    ffprobe "$COND_VIDEO" 2>&1 | grep -E "Stream.*Video" | grep -o "[0-9]*x[0-9]*" | head -1
    echo "  Expected: 544x416"
else
    echo "  ℹ️  No conditioning videos in results yet (will be created during inference)"
fi
echo ""

# Check generated baseline videos
echo "4. Checking generated baseline videos (if any):"
BASELINE_VIDEO=$(find naive_experiment/scripts/results/baselines -name "baseline_*.mp4" 2>/dev/null | head -1)
if [ -n "$BASELINE_VIDEO" ]; then
    echo "  Found: $BASELINE_VIDEO"
    ffprobe "$BASELINE_VIDEO" 2>&1 | grep -E "Stream.*Video" | grep -o "[0-9]*x[0-9]*" | head -1
    echo "  Expected: 544x416"
else
    echo "  ℹ️  No baseline videos generated yet"
fi
echo ""

# Check generated finetuned videos
echo "5. Checking generated finetuned videos (if any):"
FINETUNED_VIDEO=$(find naive_experiment/scripts/results/finetuned -name "finetuned_*.mp4" 2>/dev/null | head -1)
if [ -n "$FINETUNED_VIDEO" ]; then
    echo "  Found: $FINETUNED_VIDEO"
    ffprobe "$FINETUNED_VIDEO" 2>&1 | grep -E "Stream.*Video" | grep -o "[0-9]*x[0-9]*" | head -1
    echo "  Expected: 544x416"
else
    echo "  ℹ️  No finetuned videos generated yet"
fi
echo ""

echo "========================================"
echo "Summary"
echo "========================================"
echo "All dimensions should be 544x416 (W×H)"
echo "This ensures VAE alignment (544 % 8 = 0 ✓)"
echo ""

