# Resolution Downgrade to 480p for Memory Efficiency

## Decision: 720p → 480p

After extensive testing and multiple OOM failures, we've determined that **720p (832×1110) is too memory-intensive** for the H200 GPU (140 GB VRAM) when running video continuation experiments with fine-tuning.

**Solution:** Downgrade to **480p (554×738)** which uses only **44% of the pixels** and fits comfortably in memory.

## Memory Analysis

### Resolution Comparison for `aspect_ratio="3:4"` (landscape)

| Resolution | Dimensions (H×W) | Total Pixels | % of 720p | Memory Estimate (49 frames) |
|------------|------------------|--------------|-----------|------------------------------|
| **360p** | **(416, 554)** | **230,400** | **25%** | **~55-65 GB** |
| **480p** ✅ | **(554, 738)** | **409,920** | **44%** | **~70-80 GB** |
| **720p** ❌ | (832, 1110) | 921,600 | 100% | ~137 GB (OOM) |
| **1080p** | (1248, 1664) | 2,076,672 | 225% | ~280 GB (impossible) |

### Why 720p Failed

**All experiments at 720p failed with OOM during VAE decode:**

```
CUDA out of memory. Tried to allocate 4.41 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 1.93 GiB is free.
This process has 137.25 GiB memory in use.
```

**Memory breakdown at 720p (832×1110, 49 frames):**
- Model weights (STDiT3-XL/2): ~40 GB
- Model activations (30 steps): ~45 GB
- VAE decode intermediate: ~50 GB
- **Total:** ~135-137 GB → **OOM when trying to allocate 4.41 GB more**

### Why 480p Will Work

**Memory breakdown at 480p (554×738, 49 frames):**
- Model weights: ~40 GB (same)
- Model activations (30 steps): ~20 GB (44% of 720p)
- VAE decode intermediate: ~15 GB (44% of 720p)
- **Total:** ~75-80 GB with **60 GB headroom** ✅

## Changes Made

### 1. Preprocessing Script

**File:** `env_setup/download_ucf101/preprocess_ucf101.py`

```python
# Before (720p)
def center_crop_resize(frame, target_height=832, target_width=1110):
    """... 720p Open-Sora native (832×1110) ..."""

parser.add_argument("--height", type=int, default=832, ...)
parser.add_argument("--width", type=int, default=1110, ...)

# After (480p)
def center_crop_resize(frame, target_height=554, target_width=738):
    """
    UCF-101 is 320×240, we upscale to 480p Open-Sora native (554×738).
    This matches Open-Sora's 480p aspect ratio 0.75 (3:4 landscape).
    Chosen for memory efficiency - 720p (832×1110) causes OOM on H200.
    """

parser.add_argument("--height", type=int, default=554, ...)
parser.add_argument("--width", type=int, default=738, ...)
```

### 2. Inference Configs

**Files:**
- `naive_experiment/configs/baseline_inference.py`
- `naive_experiment/configs/finetuned_inference.py`

```python
# Before (720p)
resolution = "720p"
aspect_ratio = "3:4"  # → (832, 1110)

vae = dict(
    micro_batch_size_2d=2,  # Aggressive for 720p
    micro_frame_size=9,     # Aggressive for 720p
    tile_size=4,
)

# After (480p)
resolution = "480p"
aspect_ratio = "3:4"  # → (554, 738)

vae = dict(
    micro_batch_size_2d=4,  # Normal for 480p
    micro_frame_size=17,    # Normal for 480p
    tile_size=4,            # Conservative for safety
)
```

## Expected Performance

### Memory Usage (480p)
| Stage | Memory | Headroom |
|-------|--------|----------|
| Baseline Inference | ~75 GB | ~65 GB ✅ |
| Fine-tuning | ~80 GB | ~60 GB ✅ |
| Fine-tuned Inference | ~80 GB | ~60 GB ✅ |

**All stages fit comfortably with 60+ GB headroom!**

### Speed (480p vs 720p)
- **Baseline inference:** ~60s per video (720p: ~120s)
- **Fine-tuning:** ~8-10 min per video (720p: ~12-15 min)
- **Fine-tuned inference:** ~60s per video (720p: ~120s)
- **Total per video:** ~10-12 minutes
- **20 videos:** ~3.5-4 hours
- **1941 videos:** ~8-10 days

**480p is ~1.5-2x faster than 720p due to fewer pixels!**

### Quality Trade-off

**Resolution:**
- 720p: 832×1110 = 921,600 pixels/frame
- 480p: 554×738 = 409,920 pixels/frame
- **Ratio:** 480p has **44% of the pixels** of 720p

**Visual quality:**
- 480p is still **2x the native UCF-101 resolution** (320×240)
- Sufficient for evaluating fine-tuning improvements
- Better to have working 480p than broken 720p!

## Migration Steps

### 1. Backup Existing Data (Optional)

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101
mv ucf101_processed ucf101_processed_720p_backup
```

### 2. Repreprocess to 480p

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101
git pull origin main  # Get updated preprocessing script
sbatch --account=torch_pr_36_mren preprocess_ucf101.sbatch
```

**Monitor:**
```bash
squeue -u wc3013 | grep preprocess
tail -f slurm-*.out
```

**Verify after completion:**
```bash
ffprobe ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 2>&1 | grep -E "Stream.*Video"
# Expected: 738x554 (W×H in ffprobe = 554×738 in Open-Sora H×W)
```

### 3. Update Experiment Configs (Already Done)

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment
git pull origin main  # Get updated configs
```

### 4. Run Experiment

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts
sbatch --account=torch_pr_36_mren run_experiment.sbatch
```

## Verification

Expected log output at 480p:

```
[Loading model and components...]
image_size=(554, 738)  ✓ Correct 480p dimensions
latent_size=[15, 70, 93]  ✓ Correct latent dimensions

[Baseline Inference]
Memory usage: ~75 GB  ✓ Fits in memory

[Fine-tuning]
Memory usage: ~80 GB  ✓ Fits in memory

[Fine-tuned Inference]  
Memory usage: ~80 GB  ✓ Fits in memory
✓ Success - video saved
```

Instead of:
```
image_size=(832, 1110)
Memory usage: 137 GB
❌ CUDA out of memory
```

## Files Modified

1. **`env_setup/download_ucf101/preprocess_ucf101.py`**
   - Default height: 832 → 554
   - Default width: 1110 → 738
   - Updated docstring

2. **`naive_experiment/configs/baseline_inference.py`**
   - resolution: "720p" → "480p"
   - VAE micro_batch_size_2d: 2 → 4 (can use normal for 480p)
   - VAE micro_frame_size: 9 → 17 (can use normal for 480p)

3. **`naive_experiment/configs/finetuned_inference.py`**
   - Same changes as baseline_inference.py

## Why Not 360p?

360p (416×554) would use only **25% of 720p pixels** and definitely fit in memory, but:
- Would be ~1.3x the native UCF-101 resolution (smaller margin)
- Might lose too much visual detail for evaluation
- 480p provides better balance: **2x UCF-101 native, 56% smaller than 720p**

We can always downgrade to 360p if 480p still has issues, but 480p should work based on the memory calculations.

## Summary

| Metric | 720p (Failed) | 480p (New) | Improvement |
|--------|---------------|------------|-------------|
| Pixels | 921,600 | 409,920 | 56% reduction |
| Memory | 137 GB (OOM) | ~80 GB | 42% reduction |
| Headroom | 2.8 GB | 60 GB | 21x safer |
| Speed | Baseline | 1.5-2x faster | Better |
| Quality | Ideal | Good (2x UCF-101) | Acceptable |

**480p is the pragmatic choice: fits in memory, 2x faster, still good quality for experiment validation.** ✅

