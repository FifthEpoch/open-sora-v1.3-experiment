# Resolution Upgrade: 480p → 720p

## Motivation

Testing revealed that **720p resolution produces significantly better video quality** compared to 480p, with:
- Better structural coherence
- Reduced artifacts
- Improved prompt adherence
- More realistic textures

The quality improvement from 720p vastly outweighed any gains from fine-tuning VAE tiling parameters (tile_size 4/8/16 showed minimal differences).

## Changes Made

### 1. Preprocessing (`env_setup/download_ucf101/preprocess_ucf101.py`)
**Before:**
- Target resolution: 480p (640×480)
- Upscale UCF-101 videos from native 320×240 to 640×480

**After:**
- Target resolution: 720p (960×1280 landscape)
- Upscale UCF-101 videos from native 320×240 to 960×1280
- Default arguments updated to `--height 960 --width 1280`

**Note:** Users will need to re-preprocess UCF-101 dataset with new resolution.

### 2. Baseline Inference Config (`naive_experiment/configs/baseline_inference.py`)
- `resolution`: 480p → 720p
- `aspect_ratio`: "3:4" (unchanged, but now yields 960×1280)
- `tile_size`: 4 → 16 (larger tile for 720p, minimal artifacts)
- Updated comments to reflect 960×1280 resolution

### 3. Finetuned Inference Config (`naive_experiment/configs/finetuned_inference.py`)
- Same changes as baseline_inference.py
- Ensures fine-tuned models use same resolution as baseline

### 4. Training Config (`naive_experiment/configs/single_video_finetune.py`)
- `bucket_config`: "480p" → "720p"
- `tile_size`: 4 → 16
- Updated comments to reflect 49 frames and 960×1280 resolution

### 5. Documentation (`naive_experiment/README.md`)
- Updated dataset description: 640×480 → 960×1280
- Updated preprocessing description to mention 720p

### 6. Cleanup: Removed Old 480p Configs
Deleted superseded test configurations:
- `baseline_inference_clean.py` (480p)
- `baseline_inference_largetile.py` (480p)
- `baseline_inference_strong.py` (480p)
- `baseline_inference_official.py` (480p)
- `t2v_generation_test.py` (480p)
- `t2v_generation_official.py` (480p)
- `t2v_generation_noopt.py` (480p)
- `t2v_generation_720p.py` (tile_size=4)
- `t2v_generation_720p_largetile.py` (tile_size=8)
- `t2v_generation_720p_notiling.py` (causes OOM)
- `t2v_generation_720p_tile16.py` (standalone)
- `t2v_generation_720p_quality.py` (superseded)

### 7. Cleanup: Removed Old Test Scripts
Deleted superseded test scripts:
- `test_t2v_720p_tile16.sbatch`
- `test_t2v_quality_sweep.sbatch`

## Remaining Configs (All 720p)

### Core Experiment Configs
- `baseline_inference.py` - ✓ Updated to 720p
- `finetuned_inference.py` - ✓ Updated to 720p
- `single_video_finetune.py` - ✓ Updated to 720p

### T2V Quality-Focused Configs (All 720p, tile_size=16)
- `t2v_720p_balanced.py` - Fast & good (steps=40, cfg=8.5)
- `t2v_720p_enhanced.py` - Better quality (steps=60, cfg=10.0)
- `t2v_720p_ultra.py` - Maximum quality (steps=100, cfg=12.0)

## Migration Steps

### For Users With Existing 480p Data

1. **Re-preprocess UCF-101 dataset:**
   ```bash
   cd env_setup/download_ucf101
   sbatch --account=YOUR_ACCOUNT preprocess_ucf101.sbatch
   ```

2. **Verify preprocessing:**
   ```bash
   # Check a sample video
   ffprobe ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4
   # Should show: 960x1280, 49 frames, 24 fps
   ```

3. **Update paths in configs if needed:**
   - Configs use relative paths by default
   - Should work automatically after re-preprocessing

4. **Re-run experiments:**
   - Baseline inference will use 720p
   - Fine-tuning will train on 720p videos
   - All outputs will be 960×1280 resolution

### Disk Space Considerations

720p videos are **~4x larger** than 480p:
- 480p: ~640×480 = 307,200 pixels/frame
- 720p: ~960×1280 = 1,228,800 pixels/frame

**Estimated storage:**
- Single video (49 frames, 720p, H.264): ~5-10 MB
- Full UCF-101 (~13,320 videos): ~65-130 GB
- Keep original 320×240 videos (~13 GB) until verification complete

## Performance Impact

### Memory Usage
- 720p requires more GPU memory for VAE encoding/decoding
- `tile_size=16` provides good balance (minimal artifacts, manageable memory)
- H200 GPUs (141GB) can handle 720p comfortably

### Generation Time
- 720p generation is slower than 480p (~1.5-2x)
- Acceptable trade-off for significant quality improvement
- Quality-focused configs (balanced/enhanced/ultra) provide speed vs. quality options

## Quality Findings Summary

From T2V generation tests:
1. **Resolution matters more than tiling:** 720p >> 480p, regardless of tile_size
2. **Tiling has minimal impact:** tile_size 4/8/16 showed similar quality
3. **Sampling steps matter:** Higher num_sampling_steps improves quality
4. **CFG scale matters:** Higher cfg_scale improves prompt adherence
5. **Aesthetic conditioning helps:** aes=7.0+ produces better results

**Conclusion:** Focus on resolution and scheduler parameters, not VAE tiling.

## Next Steps

After re-preprocessing:
1. Run T2V quality levels test to confirm optimal parameters
2. Run baseline inference on full dataset (720p)
3. Run fine-tuning experiment on 720p videos
4. Compare results with 480p baseline (if available)

Expected outcome: **Significant quality improvement** with 720p videos.

