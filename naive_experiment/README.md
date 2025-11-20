# Naive Fine-tuning Experiment for Video Continuation

This experiment tests whether fine-tuning Open-Sora v1.3 with a single input video improves video continuation performance on that same video.

## Experiment Design

### Overview

For each UCF-101 video V of length 49 frames:
1. **Split**: V = [V_train | V_test]
   - V_train = first 22 frames (frames 1-22, ~0.92s)
   - V_test = remaining 27 frames (frames 23-49, ~1.13s for evaluation)

2. **Baseline (O_b)**: 
   - Generate 49 frames with first 22 conditioned (using masks)
   - Extract frames 23-49 as continuation (27 frames)
   - Compare continuation to V_test

3. **Fine-tuned (O_f)**: 
   - Fine-tune Open-Sora v1.3 on **only first 22 frames** (frames 1-22)
     - Training uses frames 1-8 as conditioning, frames 9-22 as ground truth (14 frames)
     - Model **never sees frames 23-49** during training
   - Generate 49 frames with first 22 conditioned (using masks)
   - Extract frames 23-49 as continuation (27 frames)
   - Compare continuation to V_test

4. **Evaluate**: Compare O_b and O_f continuations (frames 23-49) against V_test using PSNR, SSIM, LPIPS

**Why This Design Ensures Fair Comparison**:
- Fine-tuning only uses frames 1-22, never seeing the evaluation target (frames 23-49)
- Both baseline and fine-tuned models generate the same frames (23-49) at inference time
- Eliminates the unfair advantage of fine-tuned model having seen partial ground truth

**Note on Frame Count**: 49 frames is used because Open-Sora was trained on specific bucket sizes (1, 49, 65, 81, 97, 113 frames). Using 49 ensures stable generation matching the model's training distribution.

### Key Considerations

**Video Splitting**: 
- We feed the **full 49-frame video** to the model during inference
- The model uses **masking** to condition on first 22 frames, generate remaining 27 frames (~1.13s)
- During inference: `num_frames=49`, `condition_frame_length=22`, `cond_type="v2v_head"`
- The mask ensures first 22 frames match the conditioning input, last 27 are generated
- **Rationale**: ~45% conditioning / 55% generation provides substantial context while leaving meaningful continuation for evaluation

**Dataset Size**: Our fine-tuning dataset contains exactly **one video sample** (22 frames only - frames 23-49 withheld).

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

## Dataset: UCF-101

We use **UCF-101** (University of Central Florida - 101 action classes) as our evaluation dataset:

**Why UCF-101?**
- **Community Standard**: Used in recent video generation papers (W.A.L.T ECCV 2024, TrajVLM-Gen 2025, DIGAN ICLR 2022)
- **Benchmark Comparability**: Established FVD baselines enable direct comparison with published work
- **Dynamic Motion**: 101 action classes include sports, gymnastics, martial arts (large bodily movements)
- **Visually Engaging**: Action-rich videos make compelling demo material for presentations
- **Manageable Scale**: 2,000 videos (stratified sampling) balances diversity with computational feasibility

**Dataset Statistics**:
- Total videos: 2,000 (sampled from 13,320 via stratified sampling)
- Action classes: 101 (sports, music, daily activities)
- Sampling strategy: ~20 videos per class for balanced representation
- Native resolution: 320×240 (upscaled to 640×480 during preprocessing)
- Preprocessed: 640×480, 24 fps, 45 frames per video

See `env_setup/download_ucf101/README.md` for dataset details and download instructions.

## Methodology

### Video Splitting

Based on UCF-101 preprocessing (from `env_setup/download_ucf101/README.md`):
- Total frames: 45
- Training frames: 22 (frames 1-22, ~0.92s at 24 fps)
  - Within training: 8 conditioning frames + 14 ground truth frames
- Evaluation target: 23 frames (frames 23-45, ~0.96s)
- Inference conditioning: 22 frames (generates frames 23-45)

**Note**: The `conditioning_frames: 22` field in UCF-101 metadata matches our inference design.

### Training Configuration

**Fine-tuning parameters** (to be tuned):
- Learning rate: `1e-5` (lower than base `1e-4`)
- Epochs/steps: Small number (e.g., 10-50 steps)
- Batch size: `1` (single video)
- Accumulation steps: `1-4` (depending on memory)

**Masking Strategy**:
- **During fine-tuning**: Use `v2v_head` masking on truncated 22-frame video
  - First 8 frames are conditioned (masked to stay as input, via `latent_t // 2` approximation)
  - Model learns to generate frames 9-22 that match GT (14 frames for training)
  - Frames 23-45 are **never seen** during fine-tuning
- **During inference**: Use same 22-frame conditioning
  - First 22 frames are conditioned (via `condition_frame_length=22`)
  - Generate frames 23-45 for fair evaluation (23 frames that model never saw during training)

### Evaluation Metrics

Following Open-Sora v1.3 VAE evaluation practices:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FloLPIPS**: Flow-based LPIPS (if applicable)

## Prerequisites

1. **Downloaded and preprocessed UCF-101 dataset**:
   ```bash
   cd env_setup/download_ucf101
   python download_ucf101.py
   sbatch preprocess_ucf101.sbatch
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
    --data-csv /path/to/ucf101/ucf101_metadata.csv \
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
- UCF-101 Dataset: https://www.crcv.ucf.edu/research/data-sets/ucf101/

