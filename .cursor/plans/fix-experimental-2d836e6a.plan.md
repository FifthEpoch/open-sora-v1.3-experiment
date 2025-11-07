<!-- 2d836e6a-d522-4864-a6a0-ef4f3cee9fa6 58564ec1-bc96-4d11-a439-91a9876e7151 -->
# Test-Time Adaptation for Video Generation - Implementation Plan

## Overview

Create a comprehensive experimental framework to develop lightweight Test-Time Adaptation (TTA) methods for video generation models, inspired by SLOT. The plan addresses all possible outcomes from the initial naive fine-tuning experiment and provides clear paths to parameter-efficient TTA methods.

## Phase 0: Reorganize Existing Work

### Restructure directory layout

Move `naive_experiment/` → `experiments/01_naive_full_finetune/`

- Preserves all existing scripts, configs, and sbatch files
- Updates all internal path references
- Updates git to track new structure

**Files to update:**

- All `.sbatch` files: Update paths to configs and scripts
- `run_experiment.py`: Update relative imports and config paths
- `README.md`: Update documentation with new paths

## Phase 1: Enhanced Analysis Tools (Works for ANY Outcome)

### Create analysis module: `experiments/analysis/`

**New files:**

1. `experiments/analysis/metrics.py`

   - Existing: PSNR, SSIM, LPIPS calculation
   - **Add temporal consistency**: Frame-to-frame PSNR/SSIM over time
   - **Add optical flow analysis**: Using RAFT or Farneback, measure motion prediction accuracy
   - Function: `compute_temporal_consistency(video1, video2) -> dict`
   - Function: `compute_optical_flow_error(pred_video, gt_video) -> float`

2. `experiments/analysis/visualize_results.py`

   - Generate comparison videos (baseline | fine-tuned | ground truth side-by-side)
   - Plot metric curves over video frames
   - Per-category box plots for metric distributions
   - Learning curve visualization (training loss vs. steps)

3. `experiments/analysis/categorize_videos.py`

   - Compute video complexity metrics (scene complexity, motion intensity)
   - Cluster videos by improvement score
   - Identify which UCF-101 action categories benefit most/least
   - Output: CSV with video metadata + improvement scores

4. `experiments/analysis/analyze_experiment.py`

   - Main analysis script that runs all above analyses
   - Takes experiment results directory as input
   - Generates comprehensive report: `analysis_report.html`
   - **Usage:** `python analyze_experiment.py --results-dir experiments/01_naive_full_finetune/results/`

## Phase 2: Decision Tree Branch - Outcome-Specific Experiments

### For Outcome 1: Clear Improvement

**Directory:** `experiments/02_hyperparameter_search/`

**Goal:** Find optimal fine-tuning configuration before moving to parameter-efficient methods

**Files:**

1. `configs/finetune_sweep.py` - Config for steps=[5,10,20,50,100], lr=[1e-6,5e-6,1e-5,5e-5,1e-4]
2. `scripts/run_hp_search.py` - Orchestrates grid search
3. `scripts/run_hp_search.sbatch` - SLURM job for hyperparameter sweep
4. `scripts/analyze_hp_results.py` - Find best config, visualize heatmap

**Key features:**

- Checkpointing: Save state every 10 videos, resume if interrupted
- Batching: Process videos in batches of 50
- Result caching: Store all intermediate outputs for later analysis

### For Outcome 2: No/Minimal Improvement

**Directory:** `experiments/03_diagnostic_finetune/`

**Goal:** Determine if fine-tuning setup is fundamentally broken or model is already optimal

**Files:**

1. `configs/aggressive_finetune.py` - High LR (1e-4), more steps (500), more frames (1-32)
2. `configs/temporal_loss_finetune.py` - Add temporal consistency loss to training objective
3. `scripts/run_diagnostic.py` - Test extreme configurations
4. `scripts/verify_weight_changes.py` - Explicitly check if model weights change during fine-tuning
5. `scripts/run_diagnostic.sbatch` - SLURM job

**Key diagnostics:**

- Log training loss every step
- Compute weight L2 distance before/after fine-tuning
- Measure KL divergence of output distributions
- Qualitative: Generate 10 videos with different random seeds, check variance

### For Outcome 3: Performance Degradation

**Directory:** `experiments/04_regularized_finetune/`

**Goal:** Prevent catastrophic forgetting through regularization

**Files:**

1. `configs/l2_regularized_finetune.py` - Add weight decay, L2 penalty on weight changes
2. `configs/few_step_finetune.py` - Only 1, 3, 5 steps with low LR
3. `configs/ewc_finetune.py` - Elastic Weight Consolidation to prevent forgetting
4. `scripts/run_regularized.py` - Test regularized variants
5. `scripts/run_regularized.sbatch` - SLURM job

**Key features:**

- Implement EWC: Compute Fisher information matrix on validation set
- Track forgetting metric: Performance on held-out validation videos
- Measure distance from base model at each step

### For Outcome 4: Mixed Results

**Directory:** `experiments/05_adaptive_tta/`

**Goal:** Build predictor for which videos benefit from TTA

**Files:**

1. `scripts/compute_video_features.py` - Extract motion, complexity, scene features
2. `scripts/train_benefit_predictor.py` - Train classifier: "will video benefit from TTA?"
3. `configs/selective_finetune.py` - Only fine-tune predicted-to-benefit videos
4. `scripts/run_adaptive.py` - Full adaptive TTA pipeline
5. `scripts/run_adaptive.sbatch` - SLURM job

**Video features to compute:**

- Optical flow magnitude (motion intensity)
- Scene complexity (edge density, color variance)
- Temporal smoothness (frame-to-frame similarity)
- Action category (from UCF-101 labels)

## Phase 3: Parameter-Efficient TTA Methods (Final Goal)

### Method A: LoRA-based TTA

**Directory:** `experiments/06_lora_tta/`

**Goal:** Replace full fine-tuning with LoRA (Low-Rank Adaptation)

**Files:**

1. `configs/lora_tta.py` - LoRA config: rank=[4,8,16,32], alpha=[8,16,32]
2. `scripts/lora_finetune.py` - Modified fine-tuning with LoRA layers
3. `scripts/lora_inference.py` - Inference with LoRA weights
4. `scripts/run_lora_experiment.py` - Full LoRA TTA pipeline
5. `scripts/run_lora_experiment.sbatch` - SLURM job
6. `utils/lora_layers.py` - LoRA implementation for Open-Sora STDiT blocks

**Key implementation:**

```python
# Apply LoRA to attention layers in STDiT blocks
# For each video:
#   1. Initialize LoRA weights (random or from base)
#   2. Freeze base model, train only LoRA weights (20 steps)
#   3. Generate continuation with LoRA-adapted model
#   4. Discard LoRA weights, move to next video
```

**Efficiency metrics to track:**

- Training time per video vs. full fine-tuning
- Memory usage (MB)
- Number of trainable parameters
- Inference speedup from cached activations

### Method B: Final-Layer-Only TTA (SLOT-inspired)

**Directory:** `experiments/07_final_layer_tta/`

**Goal:** Most lightweight TTA - only adapt final hidden layer before output

**Files:**

1. `configs/final_layer_tta.py` - Config for delta vector size, learning rate
2. `scripts/final_layer_finetune.py` - Train additive delta on final layer features
3. `scripts/final_layer_inference.py` - Inference with delta vector
4. `scripts/run_final_layer_experiment.py` - Full final-layer TTA pipeline
5. `scripts/run_final_layer_experiment.sbatch` - SLURM job
6. `utils/delta_adapter.py` - Delta vector implementation

**Key implementation (SLOT-style):**

```python
# For each video:
#   1. Forward pass through entire model, cache final layer features
#   2. Initialize delta vector (size = hidden_dim)
#   3. Optimize delta to minimize CE loss on conditioning frames (5 steps)
#   4. At inference: final_features + delta -> output head
#   5. Discard delta, move to next video
```

**Expected speedup:**

- No backward pass through full model (only final layer)
- Cached features reused across optimization steps
- Minimal memory overhead (single vector per video)

### Comparison Experiment

**Directory:** `experiments/08_method_comparison/`

**Goal:** Head-to-head comparison of all TTA methods

**Files:**

1. `scripts/run_all_methods.py` - Run full fine-tuning, LoRA, final-layer on same videos
2. `scripts/compare_results.py` - Generate comparison table and plots
3. `scripts/efficiency_benchmark.py` - Measure time/memory for each method

**Comparison metrics:**

- **Quality:** PSNR, SSIM, LPIPS, temporal consistency
- **Efficiency:** Time per video, memory usage, parameter count
- **Trade-off:** Quality vs. efficiency Pareto frontier plot

## Phase 4: Checkpointing & Resumability

### Add to all experiment scripts

**Key features:**

1. **Video-level checkpointing:**
   ```python
   # Save after each video
   checkpoint = {
       'completed_videos': [...],
       'results': [...],
       'current_idx': idx
   }
   torch.save(checkpoint, f'{output_dir}/checkpoint.pth')
   ```

2. **Resume logic:**
   ```python
   if os.path.exists(f'{output_dir}/checkpoint.pth'):
       checkpoint = torch.load(...)
       start_idx = checkpoint['current_idx'] + 1
   ```

3. **Batch splitting:**

   - Add `--batch-start` and `--batch-end` arguments to all scripts
   - SLURM script: Submit multiple jobs for non-overlapping video ranges
   - Example: Job 1 processes videos 0-49, Job 2 processes videos 50-99

4. **Job chaining:**

   - For experiments >48 hours, split into dependent job chains
   - Use SLURM `--dependency=afterok:$JOB_ID` to auto-submit next batch

## Implementation Order

### Week 1: Setup & Initial Analysis

1. Reorganize directories (Phase 0)
2. Implement enhanced analysis tools (Phase 1)
3. Wait for initial naive experiment results
4. Run comprehensive analysis on results

### Week 2-3: Branch Based on Outcome

- **If Outcome 1:** Run hyperparameter search (experiments/02)
- **If Outcome 2:** Run diagnostic experiments (experiments/03)
- **If Outcome 3:** Run regularized experiments (experiments/04)
- **If Outcome 4:** Build adaptive TTA (experiments/05)

### Week 3-4: Parameter-Efficient Methods

1. Implement LoRA TTA (experiments/06) in parallel with final-layer TTA (experiments/07)
2. Run both methods on 100 videos
3. Compare results

### Week 4-5: Final Comparison & Analysis

1. Run all methods on same video set (experiments/08)
2. Generate comprehensive comparison
3. Write paper draft with results

## Deliverables

### Code Deliverables

- Reorganized experiment structure under `experiments/`
- Modular analysis tools usable across all experiments
- 4 outcome-specific experiment directories (02-05)
- 2 parameter-efficient TTA implementations (06-07)
- Comparison framework (08)
- All scripts with checkpointing and resumability

### Analysis Deliverables

- Comprehensive analysis report for naive experiment
- Hyperparameter search results (if applicable)
- Method comparison table (quality vs. efficiency)
- Visualization suite for paper figures

### Documentation

- Updated README for each experiment directory
- Usage examples for all scripts
- SLURM job templates for 48-hour limit
- Configuration file documentation

## Notes

- All experiments use UCF-101 preprocessed dataset (1941 videos, 640x480, 24fps, 45 frames)
- Conditioning frames: 1-22, Generation target: 23-45
- Base model: Open-Sora v1.3 (STDiT-XL/2)
- All jobs designed for H200 GPU with <48 hour runtime
- Checkpointing ensures no work lost if job times out

### To-dos

- [ ] Create download_ucf101.py with stratified sampling (2000 videos, ~20 per class) and captions.txt generation
- [ ] Create preprocess_ucf101.py to resize to 640×480, resample to 24fps, crop to 45 frames, generate CSV with conditioning_frames=22
- [ ] Create preprocess_ucf101.sbatch for cluster job submission (16 CPUs, 64GB RAM, 8 hours)
- [ ] Create env_setup/download_ucf101/README.md documenting dataset, download, sampling, preprocessing, and usage
- [ ] Delete entire env_setup/download_hmdb51/ directory
- [ ] Update env_setup/README.md to replace HMDB51 references with UCF-101
- [ ] Update naive_experiment/README.md to replace HMDB51 with UCF-101, cite recent papers, update statistics
- [ ] Update naive_experiment/scripts (run_experiment.py, run_experiment.sbatch, baseline_inference.py) to replace HMDB51 paths/references