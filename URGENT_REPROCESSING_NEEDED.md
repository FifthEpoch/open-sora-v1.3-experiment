# 🚨 URGENT: Reprocessing Required Before Running Experiment

## Current Status: BROKEN

**Problem:** You're still getting RGB flashing blocks because the videos haven't been reprocessed with the correct 544 width!

**Current videos:** 554 width (misaligned) ❌  
**Required videos:** 544 width (aligned) ✓

## Why This is Happening

The experiment is using **old preprocessed videos** that were created with the wrong dimensions:
- Old preprocessing: 554×416 (554 % 8 = 2 ❌ misaligned)
- Code is now correct: 544×416 (544 % 8 = 0 ✓ aligned)

**But the videos on disk are still 554 width!**

## Immediate Actions Required

### Step 1: Stop Current Experiment

```bash
# On the cluster
scancel <job_id>  # Cancel any running experiment jobs
```

### Step 2: Check Current Video Dimensions

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101

# Check if preprocessed videos exist and their dimensions
if [ -d "ucf101_processed" ]; then
    echo "Found preprocessed videos. Checking dimensions..."
    ffprobe ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 2>&1 | grep -E "Stream.*Video"
fi
```

**Expected output if wrong:**
```
Stream #0:0: Video: ..., 554x416, ...  ← 554 is WRONG!
```

### Step 3: Backup Old Videos (Optional)

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101

# Backup the misaligned videos
if [ -d "ucf101_processed" ]; then
    mv ucf101_processed ucf101_processed_554width_BROKEN
    echo "✓ Backed up old videos"
fi
```

### Step 4: Pull Latest Code

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment
git pull origin main
```

### Step 5: Reprocess UCF-101 with Correct Dimensions

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101

# Submit preprocessing job
sbatch --account=torch_pr_36_mren preprocess_ucf101.sbatch

# Monitor progress
squeue -u wc3013 | grep preprocess
```

**This will take ~1-2 hours to process all 1941 videos.**

### Step 6: Verify Correct Dimensions

After preprocessing completes:

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101

# Check dimensions of a sample video
ffprobe ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 2>&1 | grep -E "Stream.*Video"
```

**Expected output (CORRECT):**
```
Stream #0:0: Video: ..., 544x416, ...  ← 544 is CORRECT!
                          ^^^
```

**Critical check:**
```bash
# Verify divisibility by 8
python3 -c "print('Width 544 % 8 =', 544 % 8, '✓ ALIGNED' if 544 % 8 == 0 else '❌ BROKEN')"
python3 -c "print('Height 416 % 8 =', 416 % 8, '✓ ALIGNED' if 416 % 8 == 0 else '❌ BROKEN')"
```

Should output:
```
Width 544 % 8 = 0 ✓ ALIGNED
Height 416 % 8 = 0 ✓ ALIGNED
```

### Step 7: Run Experiment with Correct Videos

Only AFTER verifying Step 6:

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts
sbatch --account=torch_pr_36_mren run_experiment.sbatch
```

## What Will Happen

### Before Reprocessing (Current - BROKEN)
```
Videos on disk: 554×416 (554 % 8 = 2)
↓
VAE tries to process misaligned dimensions
↓
RGB flashing color blocks ❌
```

### After Reprocessing (FIXED)
```
Videos on disk: 544×416 (544 % 8 = 0)
↓
VAE processes properly aligned dimensions
↓
Clean video output ✓
```

## Why You Can't Skip This

**The preprocessing is NOT just a config change** - it's a physical transformation of the video files:
1. Original UCF-101: 320×240 pixels
2. Preprocessing: Crop and resize to 544×416
3. Saved to disk as MP4 files at 544×416

**If you don't reprocess:**
- The files on disk are still 554×416 (wrong)
- The model will get misaligned dimensions
- RGB flashing blocks will persist no matter what configs you use

## Timeline

| Step | Time | Status |
|------|------|--------|
| Pull latest code | 1 min | ✓ Can do now |
| Backup old videos | 1 min | ✓ Can do now |
| Submit preprocessing | 1 min | ✓ Can do now |
| **Wait for preprocessing** | **1-2 hours** | ⏳ **REQUIRED** |
| Verify dimensions | 1 min | After preprocessing |
| Run experiment | 3-4 hours | After verification |

**Total time to working experiment: ~4-6 hours** (mostly waiting for preprocessing)

## Troubleshooting

### Q: Can I just run the experiment now?
**A: NO!** The videos are still misaligned. You'll get the same RGB flashing blocks.

### Q: The code is correct, why reprocess?
**A: The code generates NEW videos.** Old videos on disk are still wrong dimensions.

### Q: Can I process just a few videos to test?
**A: Yes!** For quick testing:
```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101
python preprocess_ucf101.py --skip-cleanup
# Or process specific videos manually
```

### Q: How do I know if preprocessing is using correct dimensions?
**A: Check the output logs:**
```bash
tail -f slurm_download_prep_ucf101.out
# Should show: "Target resolution: 544×416"
```

## Summary

**DO NOT run the experiment until:**
1. ✓ Code pulled (latest preprocessing script)
2. ✓ Old videos backed up (optional)
3. ⏳ **Preprocessing completed with 544 width**
4. ✓ Dimensions verified with ffprobe (must show 544×416)

**The 10-pixel difference (554 → 544) requires physically reprocessing the video files!**

