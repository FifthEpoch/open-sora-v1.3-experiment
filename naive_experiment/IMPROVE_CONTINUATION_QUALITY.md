# Improving Video Continuation Quality

## Issues Identified

1. **Rectangular flashes/artifacts** in generated frames (worst in strong conditioning, better but still bad in official settings)
2. **Weak temporal coherence** - generated frames don't strongly continue from conditioning frames
3. **Face angle improved** but overall motion/scene coherence still poor

## Root Causes

### Issue 1: Tiling Artifacts (Rectangular Flashes)

**Cause**: VAE's tiled convolution with small `tile_size=4` creates visible seams between tiles.

From configs:
```python
vae = dict(
    use_tiled_conv3d=True,
    tile_size=4,  # TOO SMALL - causes visible tile boundaries
    ...
)
```

**Why it's worse in strong conditioning**: Higher `image_cfg_scale=10.0` amplifies VAE artifacts.

**Solutions**:

#### Option A: Increase Tile Size (Recommended)
```python
tile_size=16,  # Larger tiles = fewer seams (default is 16)
```

#### Option B: Disable Tiled Convolution (if memory allows)
```python
use_tiled_conv3d=False,  # No tiling = no artifacts, but needs more VRAM
```

#### Option C: Increase Tile Overlap
Currently `temporal_overlap=True` for temporal dimension only. The spatial tiling has no overlap, causing hard boundaries.

### Issue 2: Weak Temporal Coherence

**Possible causes**:

1. **Not enough conditioning frames in latent space**
   - Current: `condition_frame_length=5` (latent) ≈ 16 pixel frames
   - But we extract 22 pixel frames, which encode to ~5 latent frames
   - The model might not have enough context

2. **Conditioning strength too weak**
   - `image_cfg_scale=5.0` (baseline) or `2.0` (official) might be too low
   - Strong test used `10.0` but has bad artifacts

3. **SDEdit noise schedule**
   - SDEdit adds noise to conditioning frames at each step
   - This helps preserve them, but might blur temporal boundaries

4. **Model limitation**
   - Open-Sora v1.3 was primarily trained on generation, not continuation
   - Video continuation might not be its strong suit

## Recommended Improvements

### Strategy 1: Fix Tiling Artifacts First

Create new config: `baseline_inference_notiling.py`

```python
num_frames = 49
condition_frame_length = 5
resolution = "480p"
aspect_ratio = "3:4"  # LANDSCAPE
fps = 24
frame_interval = 1

save_dir = "naive_experiment/results/baselines_notiling"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

cond_type = "v2v_head"
use_sdedit = True
use_oscillation_guidance_for_text = True
use_oscillation_guidance_for_image = True

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    kernel_size=(8, 8, -1),
    use_spatial_rope=True,
    class_dropout_prob=0.0,
    force_huggingface=True,
)
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=False,  # DISABLED - no tiling artifacts
    tile_size=16,  # Not used when tiling disabled
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="google/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=50,
    cfg_scale=8.5,
    scale_image_weight=True,
    initial_image_scale=1.0,
)

image_cfg_scale = 5.0
aes = 7.0
flow = None
```

### Strategy 2: Increase Conditioning Context

Create new config: `baseline_inference_longcond.py`

```python
# Same as above, but:
condition_frame_length = 10  # DOUBLED - more latent frames for conditioning
# This means ~40 pixel frames of conditioning

# Extract more conditioning frames in inference script
PIXEL_CONDITION_FRAMES = 40  # Instead of 22
```

### Strategy 3: Balance Conditioning Strength

The sweet spot might be between official (2.0) and strong (10.0):

Create new config: `baseline_inference_balanced.py`

```python
# Same as notiling, but:
use_tiled_conv3d=False,  # No artifacts
condition_frame_length = 7,  # More context (~ 27 pixel frames)
image_cfg_scale = 5.0,  # Moderate guidance
cfg_scale = 8.5,  # Moderate text guidance
```

### Strategy 4: Try Lower Resolution (Fewer Artifacts)

Create new config: `baseline_inference_360p.py`

```python
resolution = "360p"  # Lower resolution = less memory = less tiling
aspect_ratio = "3:4"  # Still landscape
# Everything else same as baseline
```

## Testing Plan

### Phase 1: Fix Artifacts (Priority 1)

Run 3 tests to isolate tiling issue:

1. **No tiling**: `baseline_inference_notiling.py`
   - Expected: **No rectangular flashes** ✓
   - May have: Same weak coherence

2. **Larger tiles**: Modify baseline to use `tile_size=16`
   - Expected: **Reduced artifacts**
   - Faster than no tiling

3. **Lower resolution**: `baseline_inference_360p.py`
   - Expected: **Cleaner output** (less to tile)
   - Might help coherence too (simpler task)

### Phase 2: Improve Coherence (Priority 2)

After finding artifact-free config:

1. **More conditioning**: `baseline_inference_longcond.py`
   - Use 40 pixel frames (10 latent)
   - Expected: **Stronger context** for continuation

2. **Balanced guidance**: `baseline_inference_balanced.py`
   - `condition_frame_length=7`, `image_cfg_scale=5.0`
   - Expected: **Better guidance without artifacts**

3. **Different scheduler settings**:
   - Try `num_sampling_steps=100` (more denoising steps)
   - Try `initial_image_scale=0.5` (reduce conditioning noise)

### Phase 3: Alternative Approaches (If Still Poor)

If video continuation remains weak:

1. **Fine-tune the model** on UCF-101 continuation task
   - Train model specifically for video continuation
   - This is your original experiment plan!

2. **Try different conditioning strategy**:
   - Instead of `v2v_head`, use `v2v` with masking
   - Condition on **every Nth frame** instead of just the head

3. **Ensemble/multi-generation**:
   - Generate multiple continuations with different seeds
   - Pick the best one based on similarity metrics

## Quick Win: Test Configuration

For your next test, I recommend:

**`baseline_inference_clean.py`** - Best balance for UCF-101:

```python
num_frames = 49
condition_frame_length = 7  # ~27 pixel frames
resolution = "480p"
aspect_ratio = "3:4"  # Landscape

vae = dict(
    use_tiled_conv3d=False,  # ← KEY FIX: No tiling artifacts
    # ... rest same
)

image_cfg_scale = 5.0  # Moderate guidance
cfg_scale = 8.5
use_sdedit = True
```

This should give you:
- ✅ **No rectangular artifacts** (no tiling)
- ✅ **Moderate conditioning strength** (image_cfg_scale=5.0)
- ✅ **More context** (7 latent frames vs 5)
- ✅ **Landscape orientation** (3:4)

## Expected Memory Impact

Disabling tiled convolution requires more VRAM:

| Config | VRAM Usage | Artifacts | Speed |
|--------|-----------|-----------|-------|
| `tile_size=4` | ~42 GB | ❌ Severe | Fast |
| `tile_size=16` | ~60 GB | ⚠️ Mild | Medium |
| No tiling | ~100 GB | ✅ None | Slow |

H200 has 141 GB, so no tiling should work fine.

## Reality Check

**Important**: Open-Sora v1.3 might have fundamental limitations for video continuation:

1. **Training focus**: Model was trained primarily for **text-to-video generation**, not continuation
2. **Architecture**: STDiT3 might not have strong temporal coherence across long sequences
3. **Conditioning mechanism**: v2v_head might be too weak compared to other methods

If after trying these improvements the results are still poor, it strongly suggests:
- ✅ The baseline IS performing as well as Open-Sora can
- ✅ Your fine-tuning experiment is justified and necessary
- ✅ The improvement from fine-tuning will be measurable and significant

This would actually be **good news** for your experiment - it means there's clear room for improvement!

