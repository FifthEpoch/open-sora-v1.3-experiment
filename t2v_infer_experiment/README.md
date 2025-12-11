# Text-to-Video Inference Experiment

This experiment generates videos from **text only** (no conditioning frames) using the same captions from the UCF101 videos used in the LoRA experiments.

## Purpose

Compare T2V generation quality against:
1. **Baseline V2V**: Video continuation with conditioning frames
2. **LoRA V2V**: Video continuation with LoRA fine-tuning
3. **Full FT V2V**: Video continuation with full fine-tuning

This helps answer: **How much do conditioning frames help compared to pure text-to-video?**

## How It Works

1. Uses the **exact same `sampled_videos.csv`** from the LoRA experiments
2. Takes only the text caption (e.g., "A person applying eye makeup")
3. Generates 49 frames from scratch using just the caption
4. Compares generated frames 22-49 against ground truth frames 22-49
   - This matches the comparison used in V2V experiments

## Running the Experiment

```bash
cd /scratch/wc3013/open-sora-v1.3-experiment
sbatch t2v_infer_experiment/scripts/run_t2v_experiment.sbatch
```

## Output

Results saved to: `t2v_infer_experiment/results/t2v_baseline/`

- `generated/` - Generated T2V videos
- `video_*/metrics.json` - Per-video metrics
- `metrics_summary.json` - Aggregated results

## Expected Results

T2V should perform **worse** than V2V methods because:
- No visual context from conditioning frames
- Must generate scene/appearance from scratch
- Higher chance of semantic drift from the original video

This provides a lower bound for comparison.

