# Open-Sora v1.3 Experiment Troubleshooting Guide (360p_16d)

## Current Configuration

The experiment is now configured to use **360p_16d (416×544)** resolution:

- **Resolution:** 360p_16d (officially supported by v1.3)
- **Dimensions:** 416×544 (H×W in Open-Sora format)
- **FFprobe shows:** 544×416 (W×H format)
- **Aspect ratio:** 3:4 landscape (0.75)
- **VAE alignment:** Both dimensions divisible by 8 ✓

## Why 360p_16d?

### Open-Sora v1.3 Official Support

Per `docs/report_04.md`:
> "supporting 0s~113 frames, **360p & 720p**, various aspect ratios"

**Only 360p and 720p are officially trained!**

### Why Not Other Resolutions?

| Resolution | Status | Reason |
|------------|--------|--------|
| 480p | ❌ NOT supported | Not in v1.3 training distribution → poor quality |
| 720p | ❌ OOM | Uses 137 GB, H200 has 140 GB → OOM during inference |
| 360p | ⚠️ Misaligned | (416, 554) where 554 % 8 = 2 → RGB flashing blocks |
| **360p_16d** | ✅ **CORRECT** | **(416, 544) both divisible by 8 → proper VAE alignment** |

### VAE Dimension Requirements

Open-Sora VAE v1.3 requires:
- Spatial compression: 8×8
- **Dimensions must be divisible by 8**
- Recommended: divisible by 16 (for tiling)

The `_16d` suffix means "16-divisible" - all dimensions rounded to multiples of 16.

## Common Issues & Fixes

### Issue 1: RGB Flashing Color Blocks ❌

**Symptom:** Video output shows random RGB colored blocks flashing

**Cause:** VAE dimension misalignment (dimensions not divisible by 8)

**Fix:**
- Use `360p_16d` resolution (NOT `360p`)
- Dimensions: 544 % 8 = 0 ✓, 416 % 8 = 0 ✓
- Reprocess videos with correct width (544, NOT 554)

### Issue 2: CUDA Out of Memory ❌

**Symptom:** `torch.cuda.OutOfMemoryError` during VAE decode

**Cause:** Resolution too high (720p uses ~137 GB)

**Fix:**
- Use 360p_16d (uses ~60 GB vs 137 GB for 720p)
- Reduce sampling steps (30 instead of 60)
- Use tile_size=4 for conservative memory usage

### Issue 3: Cached Videos with Wrong Dimensions ❌

**Symptom:** Conditioning videos show old dimensions (e.g., 1280×960)

**Cause:** Cached from previous runs before dimension fixes

**Fix:**
```bash
# Delete all cached conditioning videos
find env_setup/download_ucf101 -type d -name "conditioning" -exec rm -rf {} + 2>/dev/null || true

# Delete cached truncated training videos
find env_setup/download_ucf101 -type d -name "truncated_for_training" -exec rm -rf {} + 2>/dev/null || true

# Verify cleanup
find env_setup/download_ucf101 -name "*_cond_*frames.mp4" | wc -l  # Should be 0
```

### Issue 4: Hardcoded Dimensions in Code ❌

**Symptom:** Code uses wrong dimensions despite correct config

**Cause:** Hardcoded `get_image_size("480p", "4:3")` in experiment scripts

**Fix:** (Already fixed in latest code)
- `run_experiment.py`: Now reads from config
- `single_video_finetune.py`: Now takes resolution/aspect_ratio parameters

## Verification Checklist

Before running experiment, verify:

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment

# 1. Pull latest code
git pull origin main

# 2. Run dimension verification
bash VERIFY_DIMENSIONS.sh

# Expected: All videos show 544x416
```

## Complete Setup Commands

```bash
# 1. Navigate to project
cd /scratch/wc3013/open-sora-v1.3-experiment

# 2. Pull latest fixes
git pull origin main

# 3. Clear any cached videos
find env_setup/download_ucf101 -type d -name "conditioning" -exec rm -rf {} + 2>/dev/null || true
find env_setup/download_ucf101 -type d -name "truncated_for_training" -exec rm -rf {} + 2>/dev/null || true
rm -rf naive_experiment/scripts/results/baselines/conditioning 2>/dev/null || true

# 4. Verify preprocessed videos are correct
ffprobe env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 2>&1 | grep "544x416"

# 5. If videos are NOT 544x416, reprocess:
cd env_setup/download_ucf101
mv ucf101_processed ucf101_processed_backup
sbatch --account=torch_pr_36_mren preprocess_ucf101.sbatch
# Wait ~1-2 hours, then verify again

# 6. Run experiment
cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts
sbatch --account=torch_pr_36_mren run_experiment.sbatch
```

## Expected Performance (360p_16d)

| Metric | Value |
|--------|-------|
| Memory (baseline) | ~60 GB |
| Memory (fine-tuned) | ~65 GB |
| Headroom | ~75 GB ✓ |
| Per video time | ~8-10 minutes |
| 20 videos | ~2.5-3.5 hours |
| 1941 videos | ~6-8 days |

## Files Modified (Latest)

1. **Preprocessing:**
   - `env_setup/download_ucf101/preprocess_ucf101.py`
   - Defaults: height=416, width=544

2. **Inference configs:**
   - `naive_experiment/configs/baseline_inference.py`
   - `naive_experiment/configs/finetuned_inference.py`
   - Both use: resolution="360p_16d", aspect_ratio="3:4"

3. **Experiment scripts:**
   - `naive_experiment/scripts/run_experiment.py`
   - `naive_experiment/scripts/single_video_finetune.py`
   - Now read dimensions from config (no hardcoding)

## Quick Reference: Dimension Conversions

| Format | Notation | Value |
|--------|----------|-------|
| Open-Sora | (H, W) | (416, 544) |
| FFprobe | W×H | 544×416 |
| NumPy/PyTorch | (H, W, C) | (416, 544, 3) |

**Always verify with ffprobe:** Look for `544x416` in output

## Troubleshooting Decision Tree

```
Are generated videos showing RGB flashing blocks?
├─ YES → Check dimension alignment
│   ├─ ffprobe shows 544x416? 
│   │   ├─ YES → Check for cached conditioning videos (delete them)
│   │   └─ NO → Reprocess videos to 544x416
│   └─ After fix: RGB blocks gone ✓
│
└─ NO → Other issues
    ├─ CUDA OOM? → Using 720p? Switch to 360p_16d
    ├─ Poor quality? → Check if using 480p (unsupported)
    └─ Slow inference? → Reduce sampling steps
```

## Emergency Reset

If everything is broken, reset to known good state:

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment
git pull origin main
git reset --hard origin/main

# Clear all generated/cached data
rm -rf env_setup/download_ucf101/ucf101_processed
rm -rf naive_experiment/scripts/results

# Reprocess from scratch
cd env_setup/download_ucf101
sbatch --account=torch_pr_36_mren preprocess_ucf101.sbatch

# After preprocessing, run experiment
cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts
sbatch --account=torch_pr_36_mren run_experiment.sbatch
```

---

**All fixes are in place. After clearing cached videos and pulling latest code, the experiment should work correctly with 360p_16d!** 🎯

