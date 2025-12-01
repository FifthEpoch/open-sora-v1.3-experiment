# Text-to-Video Generation Quality Investigation

## Issue

The generated T2V videos show **significantly lower quality** than expected for Open-Sora v1.3. Videos exhibit:
- Poor visual coherence
- Blurry/distorted frames
- Weak adherence to text prompts
- Overall "broken" appearance

## Root Cause Analysis

### Critical Differences from Official Configuration

Comparing our initial config (`t2v_generation_test.py`) with the official Open-Sora v1.3 config (`configs/opensora-v1-3/inference/t2v.py`):

| Parameter | Official | Our Initial | Impact |
|-----------|----------|-------------|--------|
| `enable_flash_attn` | `True` | `False` | ⚠️ **CRITICAL** - Attention mechanism quality |
| `enable_layernorm_kernel` | `True` | `False` | ⚠️ **CRITICAL** - Normalization quality |
| `tile_size` | `4` | `16` | 🔴 **MAJOR** - Large tiles cause artifacts |
| `use_flaw_fix` | `True` | Missing | 🔴 **MAJOR** - Quality enhancement feature |
| `num_sampling_steps` | `30` | `50` | Minor (more steps != better with wrong config) |
| `use_oscillation_guidance` | `True` (single flag) | Split into text/image | Minor |

### Why These Matter

1. **`enable_flash_attn=False` and `enable_layernorm_kernel=False`**
   - These were disabled because flash-attn and apex packages are **optional** per the installation guide
   - However, **disabling them significantly degrades model quality**, not just speed
   - The model was **trained with these optimizations enabled**
   - Running without them uses fallback implementations that produce different (worse) results

2. **`tile_size=16`** (vs official `4`)
   - Larger tiles mean the VAE processes bigger chunks at once
   - This creates **visible rectangular artifacts** and reduces spatial coherence
   - Trade-off: smaller tiles = better quality but more VRAM

3. **Missing `use_flaw_fix=True`**
   - This is a quality enhancement feature in the Open-Sora scheduler
   - Without it, the model produces lower-quality outputs

## Solutions

### Option A: Enable Optimizations (Recommended)

Install flash-attn and apex properly, then use the official config.

**Pros:**
- Best quality (matches official demos)
- Faster inference (2-3x speedup)
- Model behavior as intended

**Cons:**
- Requires successful installation of flash-attn and apex
- These packages have complex build requirements (CUDA version matching, etc.)

**Config:** `naive_experiment/configs/t2v_generation_official.py`

### Option B: Fix Config Without Optimizations

Keep optimizations disabled but fix other critical parameters.

**Changes:**
- `tile_size`: `16` → `4` (better quality, more VRAM)
- `use_flaw_fix`: Missing → `True` (enable quality fix)
- `num_sampling_steps`: `50` → `30` (match official)
- `use_oscillation_guidance`: Use single flag as in official

**Pros:**
- Should work without additional installations
- Fixes VAE artifacts and scheduler quality issues

**Cons:**
- Still slower than optimal
- Quality may still be degraded vs. models with optimizations enabled
- Unknown if model behavior is correct without flash-attn/apex

**Config:** `naive_experiment/configs/t2v_generation_noopt.py`

## Testing Plan

1. **Check optimization availability:**
   ```bash
   python naive_experiment/scripts/check_optimizations.py
   ```

2. **Test with corrected config:**
   ```bash
   sbatch --account=torch_pr_36_mren naive_experiment/scripts/test_t2v_official.sbatch
   ```
   
   This script will:
   - Automatically detect which optimizations are available
   - Choose the appropriate config (official or no-opt)
   - Generate test videos
   - Compare quality

3. **If quality is still poor with no-opt config:**
   - We need to install flash-attn and apex properly
   - Or investigate if there are other model loading issues

## Key Insight

The quality issue is likely **not a bug in our code**, but rather:
1. Using sub-optimal configuration parameters (tile_size, use_flaw_fix)
2. Running the model without the optimizations it was trained with (flash-attn, apex)

The model *can* technically run without these optimizations, but the quality degrades significantly because the fallback attention and normalization implementations produce different numerical results.

## Next Steps

1. Run `test_t2v_official.sbatch` to see if the corrected config (without optimizations) improves quality
2. If still poor, we need to properly install flash-attn and apex
3. If quality is good with no-opt config, we can proceed with the experiment
4. If quality is only good with optimizations, we need to decide:
   - Invest time in building flash-attn/apex properly, OR
   - Accept degraded quality for this proof-of-concept experiment

## Files Created

- `naive_experiment/scripts/check_optimizations.py` - Check what's available
- `naive_experiment/configs/t2v_generation_official.py` - With flash-attn/apex
- `naive_experiment/configs/t2v_generation_noopt.py` - Without flash-attn/apex but with other fixes
- `naive_experiment/scripts/test_t2v_official.sbatch` - Test script with auto-detection

