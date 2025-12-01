# Video Continuation Quality Tuning Guide

## Current Status

✅ **Conditioning is working correctly** (verified by debug test):
- 5 latent frames (~16 pixel frames) are being conditioned
- `z_cond`, `z_cond_mask`, `mask_index` all set properly
- SDEdit is enabled
- image_cfg_scale = 5.0

❌ **But output quality is poor**:
- Generated video looks like it's from scratch
- No motion coherence with conditioning frames
- Face position, orientation, scene completely different

## Possible Root Causes

### 1. **Model Not Trained for Video Continuation**
The Open-Sora v1.3 checkpoint (`hpcai-tech/OpenSora-STDiT-v4`) may not have been trained with v2v_head conditioning during training.

**Evidence:**
- Official Open-Sora v1.3 examples focus on text-to-video, not video continuation
- Model generates reasonable videos but ignores conditioning signal

### 2. **Conditioning Strength Too Weak**
Even with proper conditioning, the guidance scales may be too weak for the model to respect the reference frames.

### 3. **Latent Space Issues**
The VAE encoding/decoding may introduce artifacts that break temporal coherence.

### 4. **SDEdit Interference**
SDEdit adds noise to conditioning frames, which might be too aggressive.

## Tuning Parameters to Try

### Option A: Increase Conditioning Strength (RECOMMENDED FIRST)

Create `naive_experiment/configs/baseline_inference_strong.py`:

```python
# Try much stronger conditioning
condition_frame_length = 7  # More latent frames (7 ≈ 22 pixel frames)
image_cfg_scale = 10.0      # Much stronger (was 5.0)
cfg_scale = 10.0            # Stronger text guidance (was 8.5)

# Try WITHOUT SDEdit (might be adding too much noise)
use_sdedit = False

# Keep other quality settings
use_oscillation_guidance_for_text = True
use_oscillation_guidance_for_image = True
num_sampling_steps = 50
aes = 7.0
```

### Option B: More Aggressive Conditioning

```python
# Try conditioning on MORE frames
condition_frame_length = 10  # ~32 pixel frames
image_cfg_scale = 15.0       # Even stronger
cfg_scale = 12.0

# Disable oscillation (might interfere)
use_oscillation_guidance_for_text = False
use_oscillation_guidance_for_image = False

# More sampling steps
num_sampling_steps = 100
```

### Option C: Different Conditioning Type

Try `i2v_loop` instead of `v2v_head`:

```python
cond_type = "i2v_loop"  # Conditions on first AND last frame
condition_frame_length = 5
image_cfg_scale = 7.5
use_sdedit = True
```

This conditions on both the first frame and provides a target for the last frame, which might give better temporal coherence.

### Option D: Minimal Noise

```python
# Keep conditioning but minimize noise
condition_frame_length = 5
image_cfg_scale = 20.0  # Very strong
cfg_scale = 15.0
use_sdedit = False  # No noise on conditioning frames
num_sampling_steps = 100  # More steps for quality
```

## Testing Strategy

### Quick Test (Recommended)

Create test configs and run single-video tests:

```bash
# Create test config
cat > naive_experiment/configs/baseline_inference_test.py << 'EOF'
# Copy from baseline_inference.py and modify:
condition_frame_length = 7  # Try 7 latent frames
image_cfg_scale = 10.0      # Much stronger
use_sdedit = False          # No noise
# ... (rest of config)
EOF

# Run single test
python naive_experiment/scripts/test_conditioning_debug.py \
    --config naive_experiment/configs/baseline_inference_test.py \
    --video-path env_setup/download_ucf101/ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4 \
    --caption "A person applying eye makeup" \
    --output-path naive_experiment/results/debug/test_strong_conditioning.mp4
```

### Systematic Grid Search

If single tests show promise, run grid search:

```python
# Grid search over:
condition_frame_length = [5, 7, 10]
image_cfg_scale = [5.0, 10.0, 15.0, 20.0]
use_sdedit = [True, False]

# Test all 24 combinations on 1-2 videos
# Keep the best performing config
```

## Alternative Approaches

### 1. **Use Different Checkpoint**

Try Open-Sora v1.2 or v1.1 which may have better v2v support:

```python
model = dict(
    type="STDiT2-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v2-stage3",
    # v1.2 checkpoint
)
```

### 2. **Use True Inpainting/Outpainting**

Instead of v2v conditioning, directly copy conditioning frames into the latent tensor:

```python
# In inference script, after sampling:
# Force first N latent frames to be conditioning frames
samples[:, :, :condition_frame_length] = z_cond[:, :, :condition_frame_length]
```

This guarantees perfect conditioning but may create temporal discontinuity.

### 3. **Fine-tune Model for Video Continuation**

The naive fine-tuning experiment might actually help! The model needs to learn:
- Respect conditioning frames
- Generate coherent continuations
- Maintain scene/character consistency

### 4. **Use Different Model Architecture**

Consider:
- **AnimateDiff**: Designed for video animation
- **Text2Video-Zero**: Better temporal coherence
- **ModelScope**: Trained specifically for video continuation
- **CogVideo**: Has explicit frame conditioning support

## Diagnostic Test

Before extensive tuning, verify the model CAN respect conditioning:

### Test 1: Perfect Reconstruction
Generate a video where ALL frames are conditioning frames:

```python
condition_frame_length = 15  # All latent frames
image_cfg_scale = 100.0      # Extremely strong
```

**Expected:** Output should be nearly identical to input
**If fails:** Model fundamentally cannot respect conditioning

### Test 2: Interpolation
Condition on first AND last frame:

```python
cond_type = "i2v_loop"
# Provide first 22 frames AND last frame as conditioning
```

**Expected:** Model interpolates between start and end
**If fails:** Model doesn't understand temporal relationships

## Recommended Action Plan

1. **Immediate (10 min):** Test Option A (stronger conditioning, no SDEdit)
   ```bash
   # Modify baseline_inference.py:
   condition_frame_length = 7
   image_cfg_scale = 10.0
   use_sdedit = False
   
   # Run test
   cd naive_experiment/scripts
   sbatch test_conditioning_debug.sbatch
   ```

2. **If Option A fails (30 min):** Try diagnostic tests to verify model capability

3. **If model can't condition (1-2 hours):** 
   - Test different checkpoint (v1.2)
   - Research alternative models
   - Consider this approach infeasible with current Open-Sora

4. **If model CAN condition (2-4 hours):** Systematic grid search for optimal parameters

5. **Long-term (days):** Fine-tune model on video continuation task (your naive experiment)

## Expected Outcomes

### Best Case
- Tuning parameters improves coherence
- First 16 frames match conditioning well
- Continuation maintains scene/character
- Quality suitable for experiments

### Likely Case
- Some parameter combinations improve slightly
- Still not production-quality
- Good enough for research experiments
- Fine-tuning shows improvement

### Worst Case
- No parameter combination works
- Model fundamentally unsuitable for v2v
- Need different model architecture
- Experiments may not be viable

## Resources

- Open-Sora v2v examples: Check official repo for any video continuation demos
- Open-Sora Discord/GitHub issues: Ask if v2v_head is supported
- Alternative models: Research video continuation literature
- Model fine-tuning: Continue with naive experiment to adapt model

