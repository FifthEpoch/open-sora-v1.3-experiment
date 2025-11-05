# UCF-101 Dataset

UCF-101 (University of Central Florida - 101) is a widely-used action recognition dataset containing 101 action categories with 13,320 videos.

## Dataset Information

- **Source**: [CRCV, University of Central Florida](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
- **Release Year**: 2012
- **Actions**: 101 action categories (sports, music, daily activities)
- **Total Videos**: 13,320 (full dataset)
- **Sampled Videos**: 2,000 (stratified sampling, ~20 videos per class)
- **Format**: AVI format
- **Resolution**: 320×240 (native)
- **Total Size**: ~6.5GB (full dataset RAR archive)
- **Still Relevant**: Used in recent video generation papers (W.A.L.T ECCV 2024, TrajVLM-Gen 2025, DIGAN ICLR 2022)

## Why UCF-101?

UCF-101 remains the standard benchmark for video generation evaluation in 2024-2025 despite being released in 2012:

- **Community Standard**: Nearly universal in video generation papers, enables direct FVD comparison with published baselines
- **Diverse Actions**: 101 classes spanning sports (basketball, gymnastics), music (playing instruments), and daily activities
- **Dynamic Motion**: Includes large bodily movements (backflips, cartwheels, martial arts) ideal for testing temporal consistency
- **Visually Engaging**: Action-rich videos make compelling demo material for presentations
- **Benchmark Continuity**: Established baseline scores allow measuring progress over time

## Download and Setup

### Requirements

```bash
# Activate opensora13 environment
conda activate /scratch/wc3013/conda-envs/opensora13

# Required packages (should already be installed)
pip install rarfile tqdm
```

### Option 1: Automatic Download (Recommended)

The download script will:
1. Download UCF-101 dataset (~6.5GB)
2. Extract videos
3. Perform stratified sampling (2,000 videos, ~20 per class)
4. Generate captions.txt

```bash
cd env_setup/download_ucf101
python download_ucf101.py
```

**Note**: The script requires RAR extraction. It will try:
- Python `rarfile` library (install with: `pip install rarfile`)
- `unrar` command (macOS: `brew install unrar`, Linux: `sudo apt-get install unrar`)
- `unar` command (macOS: `brew install unar`)

If extraction fails, you can manually download and extract UCF101.rar from the official site.

### Option 2: Manual Download

If automatic download fails:

1. Download UCF-101 manually:
   - Visit: https://www.crcv.ucf.edu/research/data-sets/ucf101/
   - Download "UCF101.rar" (~6.5GB)
   - Save to: `env_setup/download_ucf101/UCF101.rar`

2. Extract the archive to `ucf101_org/` directory

3. Run the sampling script:
   ```bash
   cd env_setup/download_ucf101
   python download_ucf101.py
   ```
   (The script will detect existing videos and skip download/extraction)

## Directory Structure

After download and sampling:

```
env_setup/download_ucf101/
├── download_ucf101.py          # Download and sampling script
├── preprocess_ucf101.py        # Video preprocessing script
├── preprocess_ucf101.sbatch    # SLURM batch job for preprocessing
├── README.md                    # This file
├── UCF101.rar                   # Downloaded archive (~6.5GB)
├── ucf101_org/                  # Extracted videos (sampled 2,000)
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   ├── v_ApplyEyeMakeup_g01_c02.avi
│   └── ... (~2,000 videos)
├── captions.txt                 # Video-caption pairs
└── sampling_metadata.json       # Sampling statistics
```

## Stratified Sampling Details

To ensure balanced representation across all 101 action classes:

- **Target**: 2,000 videos total
- **Strategy**: ~20 videos per class (2000 ÷ 101 ≈ 19.8)
- **Implementation**:
  - Classes with ≤20 videos: Take all videos
  - Classes with >20 videos: Random sample 20 videos
  - Remaining quota: Distributed across larger classes
- **Random Seed**: 42 (for reproducibility)

This ensures the evaluation dataset covers all motion types while remaining computationally tractable.

## Preprocessing for Open-Sora v1.3

After downloading and sampling, preprocess videos for training.

### Requirements

All dependencies should already be installed in the opensora13 conda environment:

```bash
pip install av opencv-python pillow torch torchvision numpy pandas tqdm
```

### Option 1: Submit SLURM Batch Job (Recommended)

Video preprocessing is CPU-intensive and should be run as a batch job:

```bash
cd env_setup/download_ucf101
sbatch preprocess_ucf101.sbatch
```

The batch job will:
- Request 16 CPU cores, 64GB RAM for 8 hours on CPU partition
- Automatically configure scratch environment (avoids /home quotas)
- Process all 2,000 videos to `ucf101_processed/`
- Generate `ucf101_metadata.csv` with training metadata
- Run non-interactively (uses `--skip-cleanup` flag)

### Option 2: Run Interactively (Not Recommended for Cluster)

```bash
conda activate /scratch/wc3013/conda-envs/opensora13
cd env_setup/download_ucf101
python preprocess_ucf101.py
```

**Note**: May timeout on login nodes. Use sbatch for cluster environments.

### What Preprocessing Does

1. **Center-crop and upscale to 640×480**: UCF-101 is 320×240 native, upscaled to match Open-Sora training resolution
2. **Resample to 24 fps**: Standardizes frame rate using temporal interpolation
3. **Crop to 45 frames**: Uniformly samples 45 frames (~1.875 seconds at 24fps)
4. **Skip short videos**: Videos with <45 frames after resampling are skipped
5. **Generate metadata CSV**: Creates `ucf101_metadata.csv` with required fields
6. **Optional cleanup**: Prompts to delete original videos to save space (interactive mode only)

### Output Structure

After preprocessing:

```
env_setup/download_ucf101/
├── download_ucf101.py
├── preprocess_ucf101.py
├── preprocess_ucf101.sbatch
├── README.md
├── ucf101_org/                  # Original downloaded videos (2,000)
│   └── *.avi
├── ucf101_processed/            # Preprocessed videos
│   ├── v_ApplyEyeMakeup_g01_c01.mp4
│   ├── v_ApplyEyeMakeup_g01_c02.mp4
│   └── ... (~2,000 videos)
├── captions.txt
├── sampling_metadata.json
└── ucf101_metadata.csv          # Training metadata
```

## CSV Metadata Format

The `ucf101_metadata.csv` file contains:

| Column | Description | Value |
|--------|-------------|-------|
| `path` | Relative path to preprocessed video | e.g., `ucf101_processed/v_ApplyEyeMakeup_g01_c01.mp4` |
| `num_frames` | Number of frames | 45 |
| `conditioning_frames` | Conditioning frames for inference | 22 |
| `height` | Video height | 480 |
| `width` | Video width | 640 |
| `fps` | Frame rate | 24 |
| `text` | Action caption | e.g., `apply eye makeup` |

**Note on `conditioning_frames`**: Set to 22 to match the updated experimental design where:
- **During inference**: 22 frames (1-22) are used as conditioning to generate frames 23-45
- **During fine-tuning**: Only first 22 frames of video are used (model never sees frames 23-45)

## Usage in Experiment

Point your experiment scripts to the metadata CSV:

```bash
python naive_experiment/scripts/run_experiment.py \
  --data-csv /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_metadata.csv \
  --output-dir results \
  --num-videos 100
```

Or in training configs:

```python
data = dict(
    type="VideoTextDataset",
    data_path="/path/to/env_setup/download_ucf101/ucf101_metadata.csv",
    num_frames=45,
    frame_interval=1,
    image_size=(480, 640),
)
```

## Disk Space Management

- **Downloaded archive**: ~6.5GB (UCF101.rar)
- **Extracted original videos**: ~3GB (2,000 sampled from 13,320)
- **Preprocessed videos**: ~2GB (640×480, 45 frames, compressed)
- **Total before cleanup**: ~11.5GB
- **Total after cleanup**: ~2GB (keep only preprocessed)

The preprocessing script will prompt to delete original videos after successful preprocessing.

## Action Categories

UCF-101's 101 classes span diverse human activities:

**Sports**: Basketball, Soccer, Volleyball, Tennis, Golf, Baseball, Cricket, etc.

**Music**: Playing Piano, Violin, Guitar, Drums, Flute, etc.

**Daily Activities**: Brushing Teeth, Applying Makeup, Shaving, Typing, etc.

**Body Movements**: Handstand, Push-ups, Pull-ups, Jumping, Lunges, etc.

**Human Interactions**: Hugging, Kissing, Shaking Hands, etc.

See `sampling_metadata.json` for the full distribution in your sampled dataset.

## Notes

- **Filename Format**: `v_ActionName_g##_c##.avi` where `g##` is group number and `c##` is clip number
- **Captions**: Generated from action class names (e.g., "ApplyEyeMakeup" → "apply eye makeup")
- **Stratified Sampling**: Ensures balanced representation across all 101 classes
- **Random Seed**: Fixed at 42 for reproducible sampling
- **Preprocessing**: Maintains temporal consistency via frame interpolation
- **Short Videos**: Automatically skipped during preprocessing (no manual filtering needed)

## Comparison with Other Datasets

| Dataset | Year | Videos | Classes | Our Use | Why UCF-101 Chosen |
|---------|------|--------|---------|---------|-------------------|
| **UCF-101** | 2012 | 13,320 (2,000 sampled) | 101 | ✅ Current | Standard benchmark, dynamic motion, recent papers |
| HMDB51 | 2011 | 6,766 | 51 | ❌ Replaced | Smaller, older, less community adoption |
| Kinetics-400 | 2017 | 300k+ | 400 | ❌ Too large | Requires massive storage/compute |
| SSv2 | 2018 | 220k | 174 | ❌ Less engaging | Hand-object interactions, minimal bodily motion |

UCF-101 provides the best balance of scale, diversity, community adoption, and visual engagement for our video continuation experiment.

