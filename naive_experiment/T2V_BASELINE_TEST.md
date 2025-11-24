# Text-to-Video Baseline Test

## Motivation

After testing video continuation with various configurations:
- ❌ `tile_size=4`: Severe rectangular artifacts
- ⚠️ `tile_size=16`: Still has rectangular artifacts
- ✅ `test_official_settings` (9:16 aspect): Best quality, fewest artifacts

**Key questions**:
1. Are the artifacts inherent to the VAE/model or specific to video continuation?
2. What is the baseline quality for pure text-to-video generation?
3. Does the model understand UCF-101 action categories well?

## Test Setup

### Configuration: `t2v_generation_test.py`

**Pure text-to-video generation** (no conditioning):
- No video conditioning frames (`cond_type=None`)
- No SDEdit (`use_sdedit=False`)
- No image guidance (`image_cfg_scale=None`)
- Standard text CFG (`cfg_scale=7.5`)
- Same VAE settings as other tests (`tile_size=16`)
- Landscape 3:4 aspect ratio (matches UCF-101)

### Test Prompts (UCF-101 Categories)

Testing 5 diverse action categories:

1. **"A person applying eye makeup"** - Fine motor skills, close-up
2. **"A person playing basketball"** - Sports, fast motion
3. **"A person riding a bicycle"** - Outdoor, object interaction
4. **"A person dancing"** - Complex motion, artistic
5. **"A person swimming"** - Water, unique motion

### Run Command

```bash
cd /Users/macrohard/Desktop/Open-Sora-1.3/naive_experiment/scripts
sbatch --account=torch_pr_36_mren test_t2v_generation.sbatch
```

### Expected Output

5 videos in `naive_experiment/results/debug/t2v_samples/`:
- `apply_eye_makeup.mp4`
- `basketball.mp4`
- `cycling.mp4`
- `dancing.mp4`
- `swimming.mp4`

## What This Will Tell Us

### Scenario 1: T2V Has Same Artifacts
**Observation**: Rectangular flashes appear in pure T2V videos too

**Conclusion**: 
- ✅ Artifacts are **VAE issue**, not continuation-specific
- ✅ `tile_size=16` causes artifacts even in standard generation
- ✅ Official settings (9:16) might have different VAE parameters
- 🔍 **Action**: Check official config's VAE settings more carefully

### Scenario 2: T2V Is Clean, Continuation Has Artifacts
**Observation**: Pure T2V videos are clean, but continuation has artifacts

**Conclusion**:
- ✅ Artifacts are **conditioning-specific**
- ✅ Video-to-video conditioning process introduces VAE issues
- ✅ Might need different VAE settings for continuation
- 🔍 **Action**: Focus on conditioning mechanism optimization

### Scenario 3: T2V Quality Is Poor Overall
**Observation**: Low quality, artifacts, poor motion coherence across all videos

**Conclusion**:
- ✅ Model has **fundamental limitations** at this resolution/setting
- ✅ Fine-tuning is definitely necessary
- ✅ Baseline is performing as well as vanilla Open-Sora v1.3 can
- 🎯 **Your experiment is justified!**

### Scenario 4: T2V Quality Is Good
**Observation**: Clean videos, good motion, understands prompts well

**Conclusion**:
- ✅ Model is capable of high-quality generation
- ✅ Video continuation is the weak point, not the model itself
- ✅ Conditioning mechanism needs improvement
- 🔍 **Action**: Focus on continuation-specific optimizations

## Comparison Matrix

After running this test, we can fill in:

| Task | Quality | Artifacts | Motion Coherence | Conclusion |
|------|---------|-----------|------------------|------------|
| **V2V (tile_size=4)** | Poor | ❌ Severe | Weak | Config issue |
| **V2V (tile_size=16)** | ? | ⚠️ Present | Weak | Testing... |
| **V2V (9:16 official)** | Better | ⚠️ Minimal | Weak | Best so far |
| **T2V (tile_size=16)** | ? | ? | ? | **Run this!** |

## Expected Timeline

- Setup/loading: ~2 minutes
- Per video generation: ~2-3 minutes
- Total: ~15 minutes for 5 videos

## Next Steps Based on Results

### If T2V Shows Model Limitation
→ **Proceed with fine-tuning experiment**
- Baseline established
- Model needs task-specific training
- Your experiment is the right approach

### If T2V Is Good But V2V Is Bad
→ **Focus on continuation mechanism**
- Try different aspect ratios
- Try different tile sizes
- Try official v2v config exactly
- Consider that Open-Sora v1.3 might not be great at continuation

### If T2V Has Same Artifacts
→ **Investigate VAE settings**
- Compare official config VAE params
- Try 9:16 aspect for continuation too
- Accept that some artifacts might be unavoidable

## Most Likely Outcome

Based on what we've seen so far, **Scenario 3** (poor overall quality) is most likely:
- Model trained for generation, not continuation
- VAE artifacts appear across multiple settings
- Weak temporal coherence suggests model limitation

**This would validate your experiment**: Fine-tuning on UCF-101 continuation is necessary and will show measurable improvement!

