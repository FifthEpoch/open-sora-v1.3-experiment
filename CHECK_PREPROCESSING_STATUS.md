# Check Preprocessing Status

## Quick Check

Run this on the cluster to check if your preprocessed videos have the correct number of frames:

```bash
# On the cluster
cd /scratch/wc3013/open-sora-v1.3-experiment
bash naive_experiment/scripts/check_preprocessed_frames.sh
```

## What to Look For

### ✅ **GOOD** - Videos already have 49+ frames:
```
ApplyEyeMakeup_g01_c01.mp4: 49 frames, 640x480
ApplyEyeMakeup_g01_c02.mp4: 51 frames, 640x480
```
**Action**: No need to re-preprocess! Just pull the latest code and run the experiment.

### ❌ **BAD** - Videos have 45 frames or less:
```
ApplyEyeMakeup_g01_c01.mp4: 45 frames, 640x480
ApplyEyeMakeup_g01_c02.mp4: 44 frames, 640x480
```
**Action**: Need to re-preprocess with the updated script that crops to 49 frames.

## If You Need to Re-preprocess

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment
git pull origin main  # Get the updated preprocessing script

cd env_setup/download_ucf101

# Backup old processed videos (optional)
mv ucf101_processed ucf101_processed_45frames_backup

# Re-run preprocessing (will download if needed, then process to 49 frames)
sbatch preprocess_ucf101.sbatch
```

## Context

The previous preprocessing script used `target_frames=45` by default. The updated script uses `target_frames=49` because:

1. **Open-Sora only supports specific frame counts**: 1, 49, 65, 81, 97, 113
2. **45 frames caused the model to fall back to 33 frames** with wrong dimensions
3. **49 frames is the closest supported bucket** to our original 45

## What Changed in Preprocessing

```python
# OLD (preprocess_ucf101.py)
def crop_to_n_frames(frames, n=45):  # Default was 45
def process_video(..., target_frames=45):  # Default was 45

# NEW (preprocess_ucf101.py) 
def crop_to_n_frames(frames, n=49):  # Now 49
def process_video(..., target_frames=49):  # Now 49
```

## Impact on Experiment

- **Training**: Still uses first 22 frames (unchanged)
- **Generation**: Now generates 27 frames (23-49) instead of 23 frames (23-45)
- **Evaluation**: Compares 27 generated frames vs 27 GT frames
- **Output**: Full 49-frame video (22 conditioning + 27 generated)

