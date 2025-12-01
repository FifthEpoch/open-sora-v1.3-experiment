# Clear Cached Conditioning Videos

## Problem

The conditioning videos in `conditioning/` folders are showing **1280×960** instead of the correct **544×416**.

## Root Cause

**Cached conditioning videos from old runs!**

The inference scripts have a cache check:
```python
if cond_video_path.exists():
    return str(cond_video_path)  # Returns OLD cached video!
```

If you ran the experiment before:
1. Reprocessing videos to 544×416
2. Fixing the hardcoded dimensions bug

Then the old conditioning videos (created at wrong dimensions) are still on disk and being reused!

## Solution

Delete all cached conditioning and truncated videos:

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment

# Delete cached conditioning videos from inference
find env_setup/download_ucf101/ucf101_processed -type d -name "conditioning" -exec rm -rf {} + 2>/dev/null || true
find naive_experiment/scripts/results/baselines -type d -name "conditioning" -exec rm -rf {} + 2>/dev/null || true

# Delete cached truncated videos from fine-tuning  
find env_setup/download_ucf101/ucf101_processed -type d -name "truncated_for_training" -exec rm -rf {} + 2>/dev/null || true

# Verify deletion
echo "Checking for remaining cached videos..."
find env_setup/download_ucf101 -name "*_cond_*frames.mp4" 2>/dev/null | wc -l
find env_setup/download_ucf101 -name "*_first*frames.mp4" 2>/dev/null | wc -l
echo "Both counts should be 0"
```

## Why This Happens

### Timeline of Events

1. **Initial run** (before fixes):
   - Videos: 1280×960 (or other wrong dimensions)
   - Creates conditioning videos: 1280×960
   - Cached on disk ✓

2. **Reprocess videos**:
   - Videos: 544×416 ✓ Correct
   - But conditioning videos still cached!

3. **Run experiment**:
   - Reads 544×416 video
   - Checks: "conditioning video exists?" → YES
   - Returns cached 1280×960 video ❌ WRONG!

### The Cache Logic

```python
def split_video_for_conditioning(video_path, ...):
    cond_video_path = output_dir / f"{video_name}_cond_{condition_frames}frames.mp4"
    
    if cond_video_path.exists():
        return str(cond_video_path)  # ← Returns OLD file!
    
    # ... only creates new file if it doesn't exist
```

## Manual Cleanup Commands

If the find commands don't work, manually delete:

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment

# List all conditioning folders
find . -type d -name "conditioning" -o -name "truncated_for_training"

# Delete them one by one
rm -rf env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/conditioning
rm -rf env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/truncated_for_training
# ... repeat for each action class
```

## After Cleanup

1. **Pull latest code** (has dimension fixes):
   ```bash
   cd /scratch/wc3013/open-sora-v1.3-experiment
   git pull origin main
   ```

2. **Verify main videos are correct**:
   ```bash
   ffprobe env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 2>&1 | grep "544x416"
   ```
   Should show `544x416` ✓

3. **Run experiment** (will create NEW conditioning videos at correct size):
   ```bash
   cd naive_experiment/scripts
   sbatch --account=torch_pr_36_mren run_experiment.sbatch
   ```

4. **Verify NEW conditioning videos**:
   ```bash
   # After experiment starts running
   ffprobe naive_experiment/scripts/results/baselines/conditioning/*_cond_22frames.mp4 2>&1 | grep "Stream"
   ```
   Should show `544x416` ✓

## Prevention

To prevent this in the future, we could:

1. **Add dimension check to cache**:
   ```python
   if cond_video_path.exists():
       # Verify dimensions match expected
       container = av.open(str(cond_video_path))
       if container.streams.video[0].width != expected_width:
           cond_video_path.unlink()  # Delete mismatched cache
   ```

2. **Add version to filename**:
   ```python
   cond_video_path = f"{video_name}_cond_{condition_frames}frames_360p16d.mp4"
   ```

3. **Clear cache on config change**:
   ```python
   if config_changed:
       clear_conditioning_cache()
   ```

For now, manual deletion is sufficient since we're fixing a one-time migration issue.

## Summary

**Problem:** Cached conditioning videos from old runs have wrong dimensions  
**Solution:** Delete all cached conditioning/truncated videos  
**Result:** New videos will be created at correct 544×416 dimensions  

The main preprocessed videos are correct - it's just the cached derived videos that need to be regenerated!

