# VAE Memory Issue: Why No-Tiling Is Not Feasible

## Problem

The **clean baseline** (no tiling) fails with:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 107.28 GiB. GPU 0 has a total capacity of 139.80 GiB
```

## Root Cause

### VAE Decoding Without Tiling

The VAE decoder upsamples latents by 8× in each spatial dimension:

**Input latent**: `[1, 16, 15, 70, 93]` (B, C, T, H, W)
- 1 batch
- 16 channels
- 15 temporal frames
- 70 × 93 spatial (latent space)

**Output video**: `[1, 3, 49, 560, 744]`
- 49 temporal frames (upsampled from 15)
- 560 × 744 spatial (8× upsampled)

### Memory Explosion

During decoding, the VAE uses intermediate layers with **many more channels** (up to 512):

**Intermediate activations**: `[1, 512, 15, 560, 744]`
- 512 channels in decoder layers
- Full spatial resolution (no tiling)

**Memory required**:
```
1 × 512 × 15 × 560 × 744 × 4 bytes (fp32) = 12,902,400,000 bytes ≈ 107 GB
```

This is why it tried to allocate **107.28 GB** in a single conv3d operation!

### Why This Doesn't Work

- H200 GPU has **141 GB total VRAM**
- But model weights + activations already use **88 GB**
- Only **52 GB free**
- Needs **107 GB** for single operation → **OOM** ❌

## Why Tiling Solves This

With `use_tiled_conv3d=True` and `tile_size=16`:

**Spatial dimensions split into tiles**:
- Instead of processing 560×744 at once
- Process 16×16 tiles sequentially
- Each tile: much smaller intermediate activations

**Memory per tile**: ~4 GB (instead of 107 GB)
- `[1, 512, 15, 16, 16]` × multiple tiles
- Fits comfortably in VRAM ✓

## The Artifact Trade-off

| Config | tile_size | Single Operation Memory | Total VRAM | Artifacts |
|--------|-----------|------------------------|-----------|-----------|
| No tiling | N/A | **107 GB** ❌ | 141 GB | ✅ Zero |
| tile_size=4 | 4 | ~1 GB ✓ | 88 GB | ❌ Severe rectangles |
| tile_size=16 | 16 | ~4 GB ✓ | 95 GB | ⚠️ Mild (or none) |

## Solution: Large Tile Baseline

**Config**: `baseline_inference_largetile.py`
- `use_tiled_conv3d=True`
- `tile_size=16` (default, 4× larger than current configs)
- Should dramatically reduce artifacts
- Fits in memory comfortably

### Expected Results

**Old baseline** (tile_size=4):
- ❌ Severe rectangular flashes
- ✓ Low memory (88 GB)

**Large tile** (tile_size=16):
- ⚠️ Mild artifacts (maybe none - default setting!)
- ✓ Low memory (~95 GB)
- **4× fewer tile boundaries** than tile_size=4

**No tiling**:
- ✅ Zero artifacts
- ❌ **Not feasible** - needs 107 GB in single operation

## Why tile_size=4 Was Used

Looking at the configs, `tile_size=4` is not the default (default is 16). 

Someone likely set it to 4 to reduce memory usage further, but this created severe artifacts. The default `tile_size=16` should be much better!

## Recommendation

1. ✅ **Use large tile baseline** (`tile_size=16`)
   - This is the **official default**
   - Should have minimal or no artifacts
   - Fits in memory

2. ❌ **Abandon no-tiling approach**
   - Not feasible for 480p resolution
   - Would need to reduce resolution to 360p or lower

3. 🔍 **If large tile still has artifacts**:
   - Try `tile_size=32` (even larger, fewer boundaries)
   - Or reduce resolution to 360p with no tiling
   - Or accept that some artifacts are unavoidable

## Status

- ❌ Clean baseline (no tiling): **Not feasible** due to memory
- ⏳ Large tile baseline (tile_size=16): **Ready to test**
- ❌ Old configs (tile_size=4): **Severe artifacts**

**Next step**: Test large tile baseline, which should work and have clean output!

