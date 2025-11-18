# Hyperparameter Search - Quick Start Guide

## Overview

This experiment systematically tests **3 key hyperparameters**:
1. **Learning Rate** (5 values: 1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
2. **Conditioning Frames** (5 values: 5, 11, 17, 22, 28)
3. **Fine-tuning Steps** (5 values: 5, 10, 20, 40, 80)

Each search isolates ONE parameter while keeping others at baseline values.

## Prerequisites

1. ✅ Environment setup complete (`opensora13` conda environment)
2. ✅ UCF-101 dataset downloaded and preprocessed
3. ✅ Access to H200 GPU on cluster

## Quick Commands

### 1. Learning Rate Search

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/hyperparameter_search/scripts
sbatch run_hp_search.sbatch lr
```

**What it does**: Tests 5 learning rates (1e-6 to 1e-4) on 20 videos each  
**Time**: ~8.5 hours total (5 configs × 1.7 hours)  
**Output**: `results/lr_search/`

### 2. Conditioning Frames Search

```bash
sbatch run_hp_search.sbatch condition_frames
```

**What it does**: Tests 5 conditioning frame counts (5 to 28 frames) on 20 videos each  
**Time**: ~8.5 hours total  
**Output**: `results/condition_frames_search/`

### 3. Fine-tuning Steps Search

```bash
sbatch run_hp_search.sbatch steps
```

**What it does**: Tests 5 fine-tuning step counts (5 to 80 steps) on 20 videos each  
**Time**: ~10 hours total (varies by step count)  
**Output**: `results/steps_search/`

## Resume After Interruption

If your job is killed, **just resubmit the same command**:

```bash
# The script automatically resumes from progress.json
sbatch run_hp_search.sbatch lr  # Continues where it left off
```

The script will:
- ✅ Skip completed configurations
- ✅ Resume incomplete configurations from last video
- ✅ Start pending configurations from scratch

## Check Progress

```bash
# Check SLURM output
tail -f slurm_hp_search_<job_id>.out

# Check progress JSON
cat results/progress.json | python -m json.tool

# Check specific search results
cat results/lr_search/summary.json | python -m json.tool
```

## Analyze Results

After a search completes, generate plots and statistics:

```bash
# Learning rate analysis
python analyze_hp_results.py \
    --search-type lr \
    --results-dir /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/hyperparameter_search/results

# Conditioning frames analysis
python analyze_hp_results.py \
    --search-type condition_frames \
    --results-dir /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/hyperparameter_search/results

# Steps analysis
python analyze_hp_results.py \
    --search-type steps \
    --results-dir /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/hyperparameter_search/results
```

This generates:
- 📊 High-resolution plots comparing all configurations
- 📈 Summary statistics for each metric
- 🏆 Best configuration identification

## Understanding Results

Each experiment produces:

```
results/
├── progress.json                     # Overall progress tracker
├── lr_search/
│   ├── lr_1e-6/                      # Individual config results
│   │   ├── metrics.json              # Per-video metrics
│   │   ├── baselines/                # Baseline outputs (shared)
│   │   ├── finetuned/                # Fine-tuned outputs
│   │   └── ...
│   ├── lr_5e-6/
│   ├── ...
│   └── summary.json                  # Aggregated results
└── ...
```

### Key Metrics in summary.json

```json
{
  "config_name": "lr_1e-5",
  "lr": 1e-5,
  "psnr_improvement_pct": 25.39,      // Higher is better
  "ssim_improvement_pct": 94.11,      // Higher is better
  "lpips_improvement_pct": 7.69,      // Lower LPIPS is better (this % is positive = good)
  "avg_finetune_time_sec": 206.32     // Computational cost
}
```

## Expected Outcomes

### Learning Rate
- **Too low (1e-6)**: Slow convergence, minimal improvement
- **Optimal**: Best quality/stability trade-off
- **Too high (1e-4)**: Unstable, potential degradation

### Conditioning Frames
- **Too few (5)**: Poor context, worse predictions
- **Optimal**: Balance of context and evaluation length
- **Too many (28)**: Less evaluation data, but richer context

### Fine-tuning Steps
- **Too few (5)**: Under-fitting
- **Optimal**: Best quality/time trade-off
- **Too many (80)**: Diminishing returns, overfitting risk

## Troubleshooting

### Job killed during run
✅ **Solution**: Just resubmit. Recovery is automatic.

### Out of memory
- Reduce `--num-videos` in config (default: 20)
- Use fewer fine-tuning steps temporarily

### Wrong Python environment
```bash
# Check environment
conda env list

# Verify packages
conda activate opensora13
python -c "import torch; print(torch.__version__)"
```

### Progress.json corruption
```bash
# Backup and reset
cp results/progress.json results/progress_backup.json
echo "{}" > results/progress.json
```

## Parallel Execution

Run all three searches simultaneously (requires 3 GPUs):

```bash
sbatch run_hp_search.sbatch lr
sbatch run_hp_search.sbatch condition_frames
sbatch run_hp_search.sbatch steps
```

Total time with 3 GPUs: **~10 hours** (vs ~27 hours sequential)

## Next Steps

After hyperparameter search:
1. Identify optimal configuration from each search
2. Run combined optimal config on full dataset (95 videos)
3. Compare against baseline naive experiment
4. Use insights for lightweight TTA method design

## Support

For issues or questions:
1. Check SLURM logs: `slurm_hp_search_*.err`
2. Check progress.json for stuck configurations
3. Review README.md for detailed documentation

