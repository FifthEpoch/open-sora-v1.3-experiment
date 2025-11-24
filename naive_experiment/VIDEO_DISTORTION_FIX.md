# Video Distortion Issue - Root Cause and Fix

## Problem Discovered

When comparing the three test videos, we found:

1. **test_conditioning_output.mp4** (4:3 config):
   - Conditioning frames letterboxed (black bars top/bottom)
   - Model generates 740x555, resized to 640x480

2. **test_official_settings_output.mp4** (9:16 portrait config):
   - Conditioning frames **stretched** from landscape to portrait!
   - Model generates 480x854, resized to 640x480

3. **test_strong_conditioning_output.mp4** (4:3 config, no SDEdit):
   - Conditioning frames look distorted/generated
   - Same resize issue + aggressive guidance artifacts

## Root Cause

All three inference scripts had **hardcoded resize to 640x480**:

```python
# Bad code (removed):
target_h, target_w = 480, 640
if full_video.shape[2:] != (target_h, target_w):
    full_video = F.interpolate(
        full_video,
        size=(target_h, target_w),  # Forces everything to 640x480!
        mode='bilinear',
        align_corners=False
    )
```

But Open-Sora generates **different native resolutions** based on config:

| Config | Aspect Ratio | Native Resolution | Resize To | Problem |
|--------|--------------|-------------------|-----------|---------|
| 480p 4:3 | 1.33:1 | 740x555 | 640x480 | Letterboxing |
| 480p 9:16 | 0.56:1 | 480x854 | 640x480 | Stretching! |
| 720p 9:16 | 0.56:1 | 720x1280 | 640x480 | Severe distortion |

## Why This Broke Video Continuation

1. **Conditioning frames mismatch**: Input video (640x480) → conditioning extracted → but then stretched/squashed to match model's native resolution
2. **Latent space confusion**: VAE encodes at native resolution, but final output forced to different aspect ratio
3. **Visual artifacts**: Stretching creates obvious distortion, making it look "generated from scratch"

## Fix Applied

**Removed all hardcoded resizing**. Videos now output at their **native resolution** from the model:

```python
# New code:
# Decode ALL frames (model generates all 49 frames, with first 22 conditioned)
with torch.no_grad():
    full_video = vae.decode(samples.to(dtype)).squeeze(0)

# Keep native resolution - no resize!
print(f"Decoded video shape: {full_video.shape}")
```

### Expected Resolutions After Fix

| Config | Native Output | No Distortion |
|--------|---------------|---------------|
| 480p 4:3 | 740x555 | ✅ Matches UCF-101 aspect |
| 480p 9:16 | 480x854 | ✅ Clean portrait |
| 720p 9:16 | 720x1280 | ✅ High-res portrait |

## Impact on Evaluation

The evaluation script will now receive videos with different resolutions. Need to handle this:

### Option 1: Resize GT to match generated
```python
# In evaluate_continuations.py, resize GT frames to match generated:
if gt_tensor.shape != gen_tensor.shape:
    gt_tensor = F.interpolate(gt_tensor, size=gen_tensor.shape[2:])
```

### Option 2: Use UCF-101's native 4:3 for experiments
Keep using `aspect_ratio = "4:3"` since UCF-101 videos are landscape 4:3.

### Option 3: Crop/pad UCF-101 to portrait
Pre-process UCF-101 to 9:16 portrait before experiments (not recommended - loses data).

## Recommendations

1. **For UCF-101 experiments**: Use `aspect_ratio = "4:3"` (740x555 output)
   - Matches source video aspect ratio
   - No stretching/distortion
   - Clean comparison

2. **For portrait video datasets**: Use `aspect_ratio = "9:16"`
   - Matches training distribution
   - Better model performance

3. **Update evaluation script**: Handle flexible resolutions
   - Resize GT to match generated (preserves generated quality)
   - Or resize both to common resolution

## Files Modified

- `naive_experiment/scripts/test_conditioning_debug.py`
- `naive_experiment/scripts/baseline_inference.py`
- `naive_experiment/scripts/finetuned_inference.py`

## Next Steps

1. **Re-run tests** with fixed code:
   ```bash
   cd naive_experiment/scripts
   sbatch test_conditioning_debug.sbatch
   sbatch test_official_settings.sbatch
   ```

2. **Compare videos** - should now have correct aspect ratios

3. **Update evaluation script** to handle different resolutions

4. **Choose best config** for experiments:
   - If 4:3 works: Use for UCF-101 (native aspect)
   - If 9:16 works better: Model prefers portrait (re-preprocess UCF-101)

