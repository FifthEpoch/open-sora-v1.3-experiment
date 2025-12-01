# Hyperparameter Search - Experimental Design

## Research Question

**Which hyperparameters most significantly impact test-time fine-tuning performance for video generation?**

## Motivation

The naive experiment (95 videos, fixed parameters) showed promising results:
- **PSNR**: +25.39% improvement
- **SSIM**: +94.11% improvement
- **LPIPS**: +7.69% improvement

However, these were achieved with arbitrarily chosen hyperparameters:
- Learning rate: `1e-5` (standard for fine-tuning)
- Conditioning frames: `22` (chosen to leave ~half video for evaluation)
- Fine-tuning steps: `20` (reasonable without overfitting)

**This search systematically explores whether different values could improve results or reduce computational cost.**

## Experimental Design

### Principle: Isolated Testing

Each search varies **exactly ONE parameter** while holding all others constant. This allows us to:
1. Attribute performance changes to specific parameters
2. Understand parameter sensitivity
3. Identify optimal values for each parameter independently
4. Avoid confounding effects from simultaneous changes

### Baseline Configuration

From the successful naive experiment:
- Learning Rate: `1e-5`
- Conditioning Frames: `22`
- Fine-tuning Steps: `20`
- Videos per config: `20` (for faster iteration)

### Search Spaces

#### 1. Learning Rate Search
**Hypothesis**: The learning rate controls convergence speed and stability.

**Range**: `[1e-6, 5e-6, 1e-5, 5e-5, 1e-4]` (5 values, logarithmic scale)

**Fixed**: `condition_frames=22`, `finetune_steps=20`

**Expected Behaviors**:
- `1e-6` (very low): Slow convergence, may not adapt in 20 steps
- `5e-6` (low): Conservative updates, stable but slow
- `1e-5` (baseline): Proven to work well
- `5e-5` (high): Faster adaptation, possible instability
- `1e-4` (very high): Risk of divergence or overshooting

**Key Questions**:
- Can we converge faster with higher LR?
- Is our baseline LR already optimal?
- What's the upper bound before instability?

#### 2. Conditioning Frame Count Search
**Hypothesis**: More conditioning frames provide better context but reduce evaluation length.

**Range**: `[5, 11, 17, 22, 28]` (5 values)

**Fixed**: `lr=1e-5`, `finetune_steps=20`

**Trade-off Analysis**:
```
Conditioning | Training    | Evaluation  | Context     | Eval Quality
Frames (C)   | Frames      | Frames (E)  | Quality     | Assessment
-------------|-------------|-------------|-------------|-------------
5            | 6-22 (17)   | 23-45 (23)  | Minimal     | 40 frames
11           | 12-22 (11)  | 23-45 (23)  | Low         | 34 frames
17           | 18-22 (5)   | 23-45 (23)  | Medium      | 28 frames
22           | 23-39 (17)  | 40-45 (6)   | High        | 23 frames
28           | 29-39 (11)  | 40-45 (6)   | Very High   | 17 frames
```

**Important Note**: For fair comparison, evaluation always uses frames **beyond** what the model saw during training.

**Key Questions**:
- Is more context always better?
- Does reduced evaluation length matter?
- What's the optimal context/evaluation trade-off?

#### 3. Fine-tuning Steps Search
**Hypothesis**: More steps improve quality but have diminishing returns and higher cost.

**Range**: `[5, 10, 20, 40, 80]` (5 values, exponential)

**Fixed**: `lr=1e-5`, `condition_frames=22`

**Computational Cost** (per video, 20 videos total):
```
Steps | Time/Video | Total Time | Cost vs Baseline
------|------------|------------|------------------
5     | ~2 min     | ~40 min    | 0.4x
10    | ~2.5 min   | ~50 min    | 0.5x
20    | ~5 min     | ~100 min   | 1.0x (baseline)
40    | ~8 min     | ~160 min   | 1.6x
80    | ~14 min    | ~280 min   | 2.8x
```

**Key Questions**:
- Are 20 steps enough for convergence?
- Is there a "sweet spot" for quality/time trade-off?
- Do we see diminishing returns after a certain point?
- Risk of overfitting with too many steps?

## Metrics

For each configuration, we measure:

### Quality Metrics
1. **PSNR** (Peak Signal-to-Noise Ratio)
   - Range: typically 7-15 dB for this task
   - ↑ Higher is better
   - Measures pixel-level reconstruction accuracy

2. **SSIM** (Structural Similarity Index)
   - Range: 0-1
   - ↑ Higher is better
   - Measures perceptual similarity of structure

3. **LPIPS** (Learned Perceptual Image Patch Similarity)
   - Range: typically 0.7-0.9 for this task
   - ↓ Lower is better
   - Measures perceptual similarity using deep features

### Efficiency Metrics
4. **Training Time** (seconds)
   - Important for determining practical feasibility
   - Affects total experiment cost

5. **Improvement Percentage**
   - `(finetuned - baseline) / baseline × 100%`
   - Normalized comparison across metrics

## Recovery Mechanism

**Problem**: Jobs on shared clusters are frequently killed due to under-utilization, long runtime, or resource contention.

**Solution**: Multi-level checkpoint recovery system

### Level 1: Experiment-level Recovery
```json
{
  "lr_search": {
    "lr_1e-6": {"status": "completed", "summary": {...}},
    "lr_5e-6": {"status": "in_progress", "last_video": 15},
    "lr_1e-5": {"status": "pending"}
  }
}
```

- **Completed**: Skip entirely
- **In Progress**: Resume from `last_video + 1`
- **Pending**: Start from beginning

### Level 2: Video-level Recovery
Inherited from naive experiment's `run_experiment.py`:
- `--start-from-video` flag
- Per-video checkpoints
- Incremental `metrics.json` updates

### Level 3: Baseline Caching
Baseline inference is expensive (~49s per video). We cache it per conditioning frame count:
```json
{
  "baseline_cf22": {
    "status": "completed",
    "manifest_path": "/path/to/baseline_manifest.csv",
    "condition_frames": 22
  }
}
```

This allows all configurations with the same conditioning frame count to share baseline results.

## Expected Insights

### Practical Insights
1. **Optimal Parameters**: Best combination for quality vs. cost
2. **Sensitivity Analysis**: Which parameters matter most?
3. **Robustness**: How stable is performance across parameter ranges?

### Theoretical Insights
1. **Convergence Behavior**: How quickly does fine-tuning converge?
2. **Context Requirements**: How much video context is needed?
3. **Overfitting Risk**: Can we overfit to single samples?

### Design Insights for Lightweight TTA
1. **Parameter-Efficiency**: Can we reduce trainable parameters?
2. **Step-Efficiency**: Can we converge in fewer steps?
3. **Context-Efficiency**: Can we use less conditioning?

## Limitations & Future Work

### Current Limitations
1. **Sample Size**: 20 videos per config (vs. 95 in naive experiment)
   - Trade-off: Speed vs. statistical power
   - Mitigation: Use stratified sampling

2. **Independence Assumption**: Parameters tested separately
   - May miss interaction effects
   - Future: Grid search or Bayesian optimization

3. **Computational Budget**: ~25 GPU-hours total
   - Can't test all combinations
   - Focused on most impactful parameters

### Future Extensions
1. **Joint Optimization**: Test parameter combinations
2. **Model Architecture**: Test different fine-tuning strategies (LoRA, adapters)
3. **Longer Videos**: Test on videos > 45 frames
4. **Different Datasets**: Generalization beyond UCF-101

## Statistical Considerations

### Power Analysis
- **Effect Size**: Based on naive experiment, we expect 10-100% improvements
- **Sample Size**: 20 videos provides adequate power for large effects
- **Significance**: Focus on practical significance over statistical significance

### Multiple Comparisons
- Testing 15 configurations total (5 per search)
- Report all results transparently
- Best configuration identified post-hoc

### Variance Sources
1. **Inter-video variance**: Different video content
2. **Training stochasticity**: Random initialization, sampling
3. **Measurement noise**: Metric computation

Mitigation: Report mean ± std dev, use consistent random seeds

## Timeline

### Sequential Execution (1 GPU)
1. LR Search: ~8.5 hours
2. CF Search: ~8.5 hours
3. Steps Search: ~10 hours
**Total: ~27 hours**

### Parallel Execution (3 GPUs)
All searches simultaneously
**Total: ~10 hours**

### Analysis
- Results visualization: ~30 minutes
- Report writing: ~2 hours

## Success Criteria

### Minimum Success
- All 15 configurations complete successfully
- Clear ranking of configurations by performance
- Identification of best parameter values

### Full Success
- Statistical significance in parameter effects
- Clear trends and interpretable results
- Actionable insights for TTA method design
- Reproducible results with documentation

## References

1. Naive Experiment: `../scripts/run_experiment.py`
2. SLOT Paper: https://arxiv.org/pdf/2505.12392
3. UCF-101 Dataset: https://www.crcv.ucf.edu/data/UCF101.php

