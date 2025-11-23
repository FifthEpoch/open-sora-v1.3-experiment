# Conditioning Debug Investigation Plan

## Problem
Generated videos show "close to zero motion and scene coherence" with conditioning frames, suggesting conditioning may not be working properly.

## What We've Verified So Far

### ✅ Code Implementation
1. **Config has correct settings:**
   - `cond_type = "v2v_head"` ✓
   - `condition_frame_length = 22` ✓
   - `use_sdedit = True` ✓
   - `use_oscillation_guidance_for_image = True` ✓
   - `image_cfg_scale = 5.0` ✓

2. **Prompt format is correct:**
   - Includes both `reference_path` and `mask_strategy` ✓
   - `mask_strategy = "0,0,0,0,22,0.0"` (correct format for v2v_head) ✓

3. **Conditioning pipeline matches official Open-Sora:**
   - We call `collect_references_batch()` to encode conditioning frames ✓
   - We call `prep_ref_and_mask()` with `cond_type="v2v_head"` ✓
   - We create `x_cond_mask` with mask_index ✓
   - We pass `z_cond`, `z_cond_mask`, `mask_index`, `image_cfg_scale` to scheduler ✓
   - We use `mask=None` (correct for v2v_head) ✓

4. **Scheduler has SDEdit mechanism:**
   - At each denoising step, conditioned frames are replaced with noised `z_cond` ✓
   - This is in `opensora/schedulers/rf/__init__.py` lines 202-206 ✓

## What Could Still Be Wrong

### Possibility 1: Runtime Values Don't Match Code
Even though the code looks correct, something might be going wrong at runtime:
- `mask_index` might be empty or wrong length
- `ref` tensor might not contain the conditioning frames
- `x_cond_mask` might not be set correctly
- Config values might not be loaded properly

### Possibility 2: Model Not Trained for V2V Continuation
- The Open-Sora v1.3 checkpoint might not have been trained with v2v_head conditioning
- Even with correct code, model might not "understand" the conditioning signal
- Would explain why videos look generated from scratch

### Possibility 3: Latent Space Issues
- VAE encoding of conditioning frames might have issues
- Latent space misalignment between conditioning and generation
- Resolution/aspect ratio mismatch causing interpolation artifacts

### Possibility 4: Hyperparameter Issues
- `image_cfg_scale=5.0` might be too weak (or too strong)
- `use_sdedit=True` might be interfering instead of helping
- Number of sampling steps (50) might be insufficient

## Debug Test Script

I've created `test_conditioning_debug.py` that will:

1. **Load config and print all relevant settings**
2. **Extract conditioning frames and verify count**
3. **Run generation with extensive logging:**
   - Print shapes at every step
   - Verify `mask_index` is correct
   - Verify `ref` tensor contains conditioning frames
   - Verify `x_cond_mask` is set correctly
4. **Save both output and conditioning videos for visual comparison**

### How to Run on Cluster

```bash
# Pull latest code
cd /scratch/$USER/open-sora-v1.3-experiment
git pull origin main

# Submit debug job
sbatch naive_experiment/scripts/test_conditioning_debug.sbatch

# Wait for completion (~10-15 minutes)
# Check output
cat naive_experiment/results/debug/test_conditioning_*.out

# Download videos for visual inspection
scp cluster:/scratch/$USER/open-sora-v1.3-experiment/naive_experiment/results/debug/test_conditioning_output.mp4 .
scp cluster:/scratch/$USER/open-sora-v1.3-experiment/naive_experiment/results/debug/debug_conditioning/conditioning.mp4 .
```

### What to Look For in Debug Output

**All these should show ✓:**
```
=== VERIFICATION CHECKS ===
✓ Config has cond_type='v2v_head': True
✓ mask_index is correct length: True
✓ ref contains conditioning frames: True
✓ x_cond_mask is set: True
✓ use_sdedit is enabled: True
✓ image_cfg_scale is set: True
```

**Specific values to check:**
```
mask_index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
mask_index length: 22
ref non-zero temporal indices: [0, 1, 2, ..., 21]  (should list 22 indices)
ref non-zero count: 22 / <latent_temporal_size>
x_cond_mask non-zero frames: 22
```

**Visual inspection:**
- First 22 frames of output video should be nearly identical to conditioning video
- If they look completely different → conditioning is NOT being applied
- If they match but continuation is poor → model may not be trained for v2v

## Next Steps Based on Results

### If Debug Checks Fail (✗)
→ We have a bug in the implementation. Fix the specific failing check.

### If Debug Checks Pass (✓) BUT Visual Comparison Shows No Conditioning
→ Scheduler/model is not using the conditioning properly. Need to:
1. Verify scheduler.sample() is actually using z_cond and z_cond_mask
2. Check if model forward pass receives x_cond correctly
3. May need to add prints inside scheduler code

### If Debug Checks Pass (✓) AND First 22 Frames Match
→ Conditioning IS working, but continuation quality is poor. Try:
1. Increase `image_cfg_scale` to 7.5 or 10.0 (stronger conditioning)
2. Increase `num_sampling_steps` to 100 (better quality)
3. Try different `aes` values
4. May need to fine-tune model to improve continuation quality

### If Model Not Trained for V2V
→ Consider:
1. Using a different Open-Sora checkpoint trained for video continuation
2. Training/fine-tuning from scratch with v2v_head mask strategy
3. Using different conditioning approach (e.g., i2v_loop)

## Questions to Answer

1. **Is the code executing the conditioning logic correctly?**
   - Debug script will verify this

2. **Is the model using the conditioning signal?**
   - Visual comparison will show this

3. **If conditioning works, why is continuation poor?**
   - May need hyperparameter tuning
   - May need model fine-tuning
   - May need different model architecture

4. **Is Open-Sora v1.3 even capable of video continuation?**
   - Check official Open-Sora docs/demos
   - Check if they have v2v examples
   - May need to contact developers

## Timeline

1. **Immediate (now):** Run debug test on cluster
2. **After test (~30 min):** Analyze debug output and visual results
3. **Based on results:** Either fix bugs OR tune hyperparameters OR investigate model capabilities
4. **If needed:** Test with different Open-Sora checkpoints or conditioning strategies

