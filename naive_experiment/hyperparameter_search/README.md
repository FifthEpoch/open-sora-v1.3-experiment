# Hyperparameter Search Experiment

This experiment systematically explores the impact of three key hyperparameters on fine-tuning performance:

1. **Learning Rate** (`lr`)
2. **Conditioning Frame Count** (`condition_frames`)
3. **Fine-tuning Steps** (`finetune_steps`)

## Design Principles

- **Isolation**: Each experiment varies only ONE parameter while keeping others at their baseline values
- **Recovery**: Full checkpoint recovery support for interrupted jobs (common on shared clusters)
- **Efficiency**: Reuses baseline inference across all experiments
- **Reproducibility**: Fixed random seed and stratified sampling

## Baseline Configuration

Based on the successful naive experiment:
- Learning Rate: `1e-5`
- Conditioning Frames: `22`
- Fine-tuning Steps: `20`
- Evaluation on: `20 videos` (subset for faster iteration)

## Experiment Grid

### 1. Learning Rate Search
**Fixed**: `condition_frames=22`, `finetune_steps=20`  
**Varied**: `lr ∈ {1e-6, 5e-6, 1e-5, 5e-5, 1e-4}`

**Hypothesis**: Higher LR may converge faster but risk instability; lower LR may need more steps.

### 2. Conditioning Frame Count Search
**Fixed**: `lr=1e-5`, `finetune_steps=20`  
**Varied**: `condition_frames ∈ {5, 11, 17, 22, 28}`

**Hypothesis**: More conditioning frames provide better context but leave fewer unseen frames for evaluation.

### 3. Fine-tuning Steps Search
**Fixed**: `lr=1e-5`, `condition_frames=22`  
**Varied**: `finetune_steps ∈ {5, 10, 20, 40, 80}`

**Hypothesis**: More steps improve quality but have diminishing returns and higher computational cost.

## Directory Structure

```
hyperparameter_search/
├── README.md                           # This file
├── configs/
│   ├── lr_search.yaml                  # Learning rate configurations
│   ├── condition_frames_search.yaml    # Conditioning frame configurations
│   └── steps_search.yaml               # Fine-tuning steps configurations
├── scripts/
│   ├── run_hp_search.py                # Main orchestrator with recovery
│   ├── run_hp_search.sbatch            # SLURM submission script
│   └── analyze_hp_results.py           # Analysis and visualization
└── results/
    ├── lr_search/                      # Learning rate results
    │   ├── lr_1e-6/
    │   ├── lr_5e-6/
    │   └── ...
    ├── condition_frames_search/        # Conditioning frame results
    │   ├── cf_5/
    │   ├── cf_11/
    │   └── ...
    ├── steps_search/                   # Fine-tuning steps results
    │   ├── steps_5/
    │   ├── steps_10/
    │   └── ...
    ├── progress.json                   # Overall progress tracking
    └── summary.json                    # Aggregated results
```

## Usage

### 1. Run a Specific Search

```bash
# Learning rate search
cd /scratch/wc3013/open-sora-v1.3-experiment/naive_experiment/hyperparameter_search/scripts
sbatch run_hp_search.sbatch --search-type lr

# Conditioning frames search
sbatch run_hp_search.sbatch --search-type condition_frames

# Steps search
sbatch run_hp_search.sbatch --search-type steps
```

### 2. Resume from Interruption

The script automatically detects incomplete experiments in `progress.json` and resumes:

```bash
# Same command - will skip completed experiments
sbatch run_hp_search.sbatch --search-type lr
```

### 3. Analyze Results

```bash
python scripts/analyze_hp_results.py \
    --search-type lr \
    --results-dir results/lr_search
```

## Recovery Mechanism

Each experiment creates entries in `results/progress.json`:

```json
{
  "lr_search": {
    "lr_1e-6": {"status": "completed", "psnr": 9.5, "ssim": 0.35, ...},
    "lr_5e-6": {"status": "in_progress", "videos_completed": 15},
    "lr_1e-5": {"status": "pending"}
  }
}
```

The orchestrator:
1. Checks `progress.json` for each configuration
2. Skips `"completed"` experiments
3. Resumes `"in_progress"` from last completed video
4. Starts `"pending"` experiments from scratch

## Output Metrics

For each hyperparameter configuration:
- **PSNR** (↑ better)
- **SSIM** (↑ better)
- **LPIPS** (↓ better)
- **Timing**: inference + fine-tuning duration
- **Per-video metrics** for detailed analysis

## Expected Runtime

- **20 videos per configuration**
- **~5 minutes per video** (based on naive experiment)
- **~1.7 hours per configuration**

Total for all 15 configurations: **~25 hours** (can run in parallel if multiple GPUs available)

## References

- Baseline experiment: `../scripts/run_experiment.py`
- Baseline results: `../scripts/results/metrics.json`

