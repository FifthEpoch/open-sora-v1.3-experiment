# Enhanced Quality Configuration for Naive Experiment

## Summary

The naive fine-tuning experiment has been configured to use **Enhanced Quality** parameters based on T2V quality testing results.

## Configuration Changes

### Enhanced Quality Parameters Applied

Both `baseline_inference.py` and `finetuned_inference.py` now use:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `num_sampling_steps` | 60 | More denoising iterations for better detail |
| `cfg_scale` | 10.0 | Stronger prompt adherence |
| `use_oscillation_guidance` | True | Dynamic guidance to avoid local minima |
| `use_flaw_fix` | True | Post-processing artifact removal |
| `aes` | 7.0 | "Excellent" aesthetic quality target |
| `flow` | 6.0 | Higher motion strength for dynamic videos |

### Resolution

- **720p (960×1280)** - matching Open-Sora v1.3 training distribution
- **49 frames** total (22 conditioning + 27 continuation)
- **tile_size=16** for VAE (optimal for 720p)

## Performance Estimates

Based on quality levels testing:

### Per Video Time
- **Computational time:** ~128 seconds (~2.1 minutes)
  - Sampling: 118s (60 iterations @ 1.98s/step)
  - VAE Decode: ~10s

### Full Experiment (100 videos)
- **Total compute time:** ~213 minutes (**3.5 hours**)
- **With overhead:** ~4-5 hours (including model loading, I/O, checkpointing)

### Full Dataset (1941 videos)
- **Total compute time:** ~69 hours (**2.9 days**)
- **With overhead:** ~72-75 hours (**3 days**)

## Prerequisites

### ⚠️ CRITICAL: Re-preprocess UCF-101 to 720p

The videos **MUST** be re-preprocessed to 720p before running the experiment:

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment
cd env_setup/download_ucf101
sbatch --account=torch_pr_36_mren preprocess_ucf101.sbatch
```

**Verification after preprocessing:**
```bash
ffprobe ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4
# Should show: 960x1280, 49 frames, 24 fps
```

## Running the Experiment

### Step 1: Ensure 720p Preprocessing is Complete

Check preprocessing status:
```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101
squeue -u wc3013 | grep preprocess  # Check if still running
```

Verify a sample video:
```bash
ffprobe ucf101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.mp4
```

### Step 2: Submit Experiment Job

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment
git pull origin main
cd naive_experiment/scripts
sbatch --account=torch_pr_36_mren run_experiment.sbatch
```

### Step 3: Monitor Progress

```bash
# Check job status
squeue -u wc3013

# View live output
tail -f /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/slurm_run_experiment.out

# Check progress JSON
cat /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/scripts/results/progress.json | grep video_idx | tail -5
```

## Expected Outputs

### Directory Structure
```
naive_experiment/scripts/results/
├── baselines/                    # All O_b outputs (720p, 49 frames)
│   ├── baseline_0000_*.mp4
│   ├── baseline_0001_*.mp4
│   └── ...
├── finetuned/                    # All O_f outputs (720p, 49 frames)
│   ├── finetuned_0000_*.mp4
│   ├── finetuned_0001_*.mp4
│   └── ...
├── finetuned_checkpoints/        # Temporary checkpoints (cleaned up)
├── experiment_manifest.csv       # Mapping of videos to outputs
└── progress.json                 # Checkpointing for resumption
```

### Video Specifications
- **Resolution:** 960×1280 (720p landscape)
- **Frames:** 49 total
- **FPS:** 24
- **Format:** H.264 MP4

## Quality vs. Speed Trade-offs

Comparison with other quality levels:

| Config | Steps | Compute Time/Video | 100 Videos | Quality |
|--------|-------|-------------------|------------|---------|
| Balanced | 40 | 89s (~1.5 min) | ~2.5 hrs | Good |
| **Enhanced** | **60** | **128s (~2.1 min)** | **~3.5 hrs** | **Better** ✅ |
| Ultra | 100 | 207s (~3.4 min) | ~5.7 hrs | Best |

**Enhanced** provides the best balance of quality and speed for the experiment.

## Troubleshooting

### Job Killed Due to Low GPU Utilization

The script includes GPU keepalive to maintain ~50% utilization during I/O operations.

If still killed, resume from checkpoint:
```bash
# Check last completed video
cat results/progress.json | grep video_idx | tail -1

# Edit run_experiment.sbatch to add resume flags:
python "${SCRIPT_DIR}/run_experiment.py" \
    --data-csv "${PROJECT_ROOT}/env_setup/download_ucf101/ucf101_metadata.csv" \
    --output-dir results \
    --num-videos 100 \
    --skip-baseline \
    --start-from-video <LAST_COMPLETED + 1>
```

### Videos Still 480p

Re-run preprocessing:
```bash
cd /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101
# Backup old 480p data if needed
mv ucf101_processed ucf101_processed_480p_backup
# Run new 720p preprocessing
sbatch --account=torch_pr_36_mren preprocess_ucf101.sbatch
```

### Out of Memory Errors

Enhanced config with 720p should fit on H200 (141GB VRAM). If OOM occurs:
- Verify you're on H200: `nvidia-smi` should show ~141GB total memory
- Check if other processes are using GPU: `nvidia-smi`
- Reduce `micro_batch_size` in VAE config (currently 1, already minimal)

## Next Steps After Experiment

1. **Evaluate Results:**
   ```bash
   cd naive_experiment/scripts
   sbatch --account=torch_pr_36_mren run_evaluation.sbatch
   ```

2. **Analyze Metrics:**
   - Check `results/metrics.json` for PSNR, SSIM, LPIPS scores
   - Compare baseline vs. finetuned performance

3. **Visualize Results:**
   - Use scripts in `visualization/` to generate plots
   - Create side-by-side comparison videos

## Configuration Files Updated

- ✅ `naive_experiment/configs/baseline_inference.py` - Enhanced quality params
- ✅ `naive_experiment/configs/finetuned_inference.py` - Enhanced quality params
- ✅ `naive_experiment/configs/single_video_finetune.py` - 720p bucket config
- ✅ `env_setup/download_ucf101/preprocess_ucf101.py` - 720p output

All configurations are now aligned for 720p Enhanced quality experiments! 🎯

