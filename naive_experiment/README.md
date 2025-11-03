# Naive Fine-tuning Experiment for Video Continuation

This experiment tests whether fine-tuning Open-Sora v1.3 with a single input video improves video continuation performance on that same video.

## Experiment Design

### Overview

For each HMDB51 video V of length 45 frames:
1. **Split**: V = [V' | GT]
   - V' = first 8 frames (conditioning input, ~0.33s)
   - GT = remaining 37 frames (ground truth for evaluation, ~1.54s)

2. **Baseline (O_b)**: 
   - Generate 45 frames with first 8 conditioned on V' (using masks)
   - Extract frames 9-45 as continuation
   - Compare continuation to GT

3. **Fine-tuned (O_f)**: 
   - Fine-tune Open-Sora v1.3 on full V (all 45 frames) with v2v_head masking
   - Generate 45 frames with first 8 conditioned on V' (using masks)
   - Extract frames 9-45 as continuation
   - Compare continuation to GT

4. **Evaluate**: Compare O_b and O_f continuations against GT using PSNR, SSIM, LPIPS

### Key Considerations

**Video Splitting**: 
- We feed the **full 45-frame video** to the model
- The model uses **masking** to condition on first 8 frames, generate remaining 37 frames (~1.5s)
- During inference: `num_frames=45`, `condition_frame_length=8`, `cond_type="v2v_head"`
- The mask ensures first 8 frames match the conditioning input, last 37 are generated
- **Rationale**: ~18% conditioning / 82% generation matches Open-Sora v1.3 examples, giving enough continuation for visual evaluation

**Dataset Size**: Our fine-tuning dataset contains exactly **one video sample** (45 frames total).

**Batch Size Strategy**: 
- Batch size must be ≤ 1 since we only have one training sample
- Use gradient accumulation if needed for training stability
- Modified training config sets `batch_size=1, accumulation_steps=1`

**Efficient Checkpoint Management**:
1. Load vanilla checkpoint once
2. Generate all baseline outputs (O_b) for all videos
3. For each video:
   - Load vanilla checkpoint
   - Fine-tune on that video
   - Generate O_f
   - Save O_f, discard fine-tuned weights
   - Move to next video

## File Structure

```
naive_experiment/
├── README.md                          # This file
├── configs/
│   ├── baseline_inference.py         # Config for baseline inference
│   ├── single_video_finetune.py      # Config for per-video fine-tuning
│   └── finetuned_inference.py        # Config for fine-tuned inference
├── scripts/
│   ├── run_experiment.py              # Main orchestrator
│   ├── run_experiment.sbatch          # SLURM batch submission script
│   ├── baseline_inference.py          # Batch generate baselines
│   ├── single_video_finetune.py      # Fine-tune on single video
│   ├── finetuned_inference.py        # Generate with fine-tuned model
│   └── evaluate_continuations.py     # Compute PSNR/SSIM/LPIPS
└── results/
    ├── baselines/                     # All O_b outputs
    ├── finetuned/                     # All O_f outputs
    └── metrics.json                   # Evaluation results
```

## Methodology

### Video Splitting

Based on HMDB51 preprocessing (from `env_setup/download_hmdb51/README.md`):
- Total frames: 45
- Conditioning frames: 8 (~0.33s at 24 fps)
- Continuation target: 37 frames (~1.54s)

**Note**: The `conditioning_frames: 32` field in HMDB51 metadata is only metadata. For our experiment, we use 8 conditioning frames to generate longer continuations that are easier to evaluate visually.

### Training Configuration

**Fine-tuning parameters** (to be tuned):
- Learning rate: `1e-5` (lower than base `1e-4`)
- Epochs/steps: Small number (e.g., 10-50 steps)
- Batch size: `1` (single video)
- Accumulation steps: `1-4` (depending on memory)

**Masking Strategy**:
- **During fine-tuning**: Use `v2v_head` masking on the full 45-frame video
  - First 22 frames are conditioned (masked to stay as input, training defaults to `latent_t // 2`)
  - Model learns to generate remaining 23 frames that match GT
- **During inference**: Use different masking
  - First 8 frames are conditioned (via `condition_frame_length=8`)
  - Generate remaining 37 frames for longer, more evaluable continuations

### Evaluation Metrics

Following Open-Sora v1.3 VAE evaluation practices:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FloLPIPS**: Flow-based LPIPS (if applicable)

## Prerequisites

1. **Downloaded and preprocessed HMDB51 dataset**:
   ```bash
   cd env_setup/download_hmdb51
   python download_hmdb51.py
   python preprocess_hmdb51.py
   ```

2. **Installed Open-Sora dependencies**:
   - Flash-attention and apex are **optional** for this experiment
   - Configs set `enable_flash_attn=False` and `enable_layernorm_kernel=False`
   - See `env_setup/README.md` for full environment setup

## Usage

### Option 1: Submit as SLURM batch job (Recommended for cluster)

```bash
# Submit to H100 GPU partition
cd naive_experiment/scripts
sbatch run_experiment.sbatch
```

The script will:
- Request 1 H100 GPU, 8 CPUs, 64GB RAM for 48 hours
- Automatically configure scratch environment to avoid /home quotas
- Load opensora13 conda environment
- Download checkpoints from HuggingFace automatically
- Process specified number of videos

**Customize the experiment** by editing `run_experiment.sbatch`:
- Modify `--num-videos` argument
- Change output directory path
- Adjust fine-tuning parameters

### Option 2: Run interactively (Not recommended on login node)

```bash
# Run locally or on interactive GPU node
cd naive_experiment
python scripts/run_experiment.py \
    --data-csv /path/to/hmdb51/hmdb51_metadata.csv \
    --output-dir results \
    --num-videos 10  # Start with a subset
```

Note: Checkpoints will auto-download from HuggingFace if not specified.

## Expected Outcomes

**Hypothesis**: Fine-tuning on the input video should improve continuation quality by:
- Learning video-specific temporal patterns
- Better understanding of local motion dynamics
- Reduced artifacts on the specific video style

**Potential Issues**:
- Overfitting with only 1 sample
- Mode collapse due to extreme data scarcity
- Unstable training dynamics

## References

- Open-Sora v1.3: https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.3
- Commands Documentation: https://github.com/hpcaitech/Open-Sora/blob/opensora/v1.3/docs/commands.md
- HMDB51 Dataset: https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

