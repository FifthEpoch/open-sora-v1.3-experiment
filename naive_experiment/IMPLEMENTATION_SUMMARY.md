# Implementation Summary

This document summarizes the three tasks that were completed:

## Task 1: Timing Instrumentation

### What was implemented:
- Added timing measurement to `baseline_inference.py` and `finetuned_inference.py`
- Timing starts after function entry and ends after video save (excluding model loading/unloading)
- Baseline inference time is saved in `baseline_manifest.csv` as `baseline_inference_time_sec`
- Finetuned inference time is printed to stdout as `INFERENCE_TIME:<seconds>`

### Status: **Partially Complete**
- ✅ Timing added to inference scripts
- ⚠️ Fine-tuning timing not yet captured (would require modifications to `train.py`)
- ⚠️ Timing data not yet integrated into `metrics.json` (would require updates to `run_experiment.py` and `evaluate_continuations.py`)

### To complete:
1. Capture fine-tuning time in `run_experiment.py` (measure time between `torchrun` start and end)
2. Parse timing from `finetuned_inference.py` stdout in `run_experiment.py`
3. Pass timing data through manifests to `evaluate_continuations.py`
4. Include timing in final `metrics.json` output

## Task 2: Visualization Script

### What was implemented:
A complete visualization tool in `naive_experiment/scripts/visualization/`

**Files created:**
- `plot_metrics.py`: Python script to generate comparison plots
- `README.md`: Documentation and usage instructions

**Features:**
- Loads `metrics.json` and extracts baseline vs fine-tuned metrics
- Calculates averages and standard errors for PSNR, SSIM, and LPIPS
- Generates two plots:
  1. **metrics_comparison.png**: Bar charts with error bars comparing averages
  2. **metrics_distributions.png**: Histograms showing metric distributions
- Prints summary statistics and percentage improvements to stdout

**Usage:**
```bash
cd naive_experiment/scripts/visualization
python plot_metrics.py \
    --metrics-json ../results/metrics.json \
    --output-dir ./plots
```

### Status: **Complete** ✅

## Task 3: Scale to 100 Videos with Checkpoint Cleanup

### What was implemented:

**1. Checkpoint Cleanup (`run_experiment.py`)**
- Added automatic cleanup of checkpoint directories after each video's evaluation
- Uses `shutil.rmtree()` to delete `video_ckpt_dir` after metrics are computed
- Includes error handling to prevent cleanup failures from breaking the experiment
- Logs cleanup actions for debugging

**2. Updated Experiment Scale (`run_experiment.sbatch`)**
- Changed `--num-videos` from 10 to 100
- Experiment will now process 100 videos with stratified sampling across UCF-101 classes

**3. Storage Strategy:**
- Baseline outputs: Kept until all 100 are generated (needed for batch evaluation)
- Fine-tuned outputs: Kept (relatively small, ~1-2GB for 100 videos)
- Checkpoints: **Deleted immediately** after each video's evaluation (~50GB per video × 100 = 5TB saved!)

### Status: **Complete** ✅

## Summary

### Completed:
1. ✅ Timing instrumentation added to inference scripts
2. ✅ Full visualization pipeline with plots and statistics
3. ✅ Checkpoint cleanup after evaluation
4. ✅ Scaled experiment to 100 videos

### Remaining (for full timing integration):
- Capture fine-tuning timing in `run_experiment.py`
- Parse and store timing in manifests
- Include timing in `metrics.json` via `evaluate_continuations.py`

### Next Steps:
1. **Test the visualization script** on current `metrics.json`:
   ```bash
   cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/visualization
   python plot_metrics.py --metrics-json ../results/metrics.json --output-dir ./plots
   ```

2. **Run the 100-video experiment**:
   ```bash
   cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts
   git pull origin main
   sbatch run_experiment.sbatch
   ```

3. **Monitor storage usage** during the run to ensure cleanup is working:
   ```bash
   watch -n 60 du -sh /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/finetuned_checkpoints/
   ```

## Storage Estimates (100 videos)

| Component | Size per Video | Total for 100 | Kept/Deleted |
|-----------|---------------|---------------|--------------|
| Baseline outputs | ~15-20MB | ~1.5-2GB | ✅ Kept |
| Finetuned outputs | ~15-20MB | ~1.5-2GB | ✅ Kept |
| Checkpoints | ~50GB | ~5TB | ❌ Deleted |
| Conditioning clips | ~5MB | ~500MB | ✅ Kept |
| **Total Storage Used** | | **~4-5GB** | After cleanup |

The checkpoint cleanup saves approximately **5TB** of storage!

