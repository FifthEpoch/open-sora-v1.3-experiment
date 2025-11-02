# HMDB51 Dataset

The HMDB51 (Human Motion Database 51) is an action recognition dataset containing 51 action categories.

## Dataset Information

- **Source**: [Serre Lab, Brown University](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- **Actions**: 51 different human action categories
- **Format**: Video clips in AVI format
- **Total size**: ~2GB compressed
- **Note**: The official download URL may be broken. If automatic download fails, you may need to manually download from the official website or find an alternative source.

## Download Options

**Option 1: Use Hugging Face Datasets (Recommended - No RAR extraction needed!)**

HMDB51 is available pre-extracted through Hugging Face Datasets. This avoids the RAR extraction problem entirely:

```bash
# Activate opensora13 environment
conda activate /scratch/wc3013/conda-envs/opensora13

# Install datasets (should already be in requirements)
pip install datasets

# Download using Hugging Face Datasets
cd env_setup/download_hmdb51
python download_hmdb51_hf.py
```

This script downloads the dataset directly from Hugging Face and avoids any RAR extraction issues!

**Option 2: Manual RAR Download and Extract**

## Manual Download Instructions

1. **Install extraction tool** (if not already installed):
   
   **IMPORTANT**: HMDB51 is a `.rar` file, which requires `unrar` to extract. p7zip does NOT support RAR format.
   
   **On macOS**:
   ```bash
   brew install unrar  # or: brew install unar
   ```
   
   **On Linux (with sudo)**:
   ```bash
   sudo apt-get install unrar  # or: sudo apt-get install unrar-free
   ```
   
   **On cluster (without sudo)** - limited options:
   - Ask your cluster admin to install `unrar`
   - Or manually compile `unrar` from source and add to PATH
   - Or download a pre-extracted version of HMDB51 from another source

2. **Run the download script**:
   ```bash
   # IMPORTANT: Activate the opensora13 conda environment first!
   conda activate /scratch/wc3013/conda-envs/opensora13
   
   cd env_setup/download_hmdb51
   python download_hmdb51.py
   ```

3. **Wait for completion**:
   - The script will download ~2GB archive
   - Extract videos to `hmdb51_org/` directory
   - Generate `captions.txt` file with video-caption pairs
   - Total time: ~10-20 minutes depending on your internet connection

## Directory Structure

After running the script, your directory will look like:

```
env_setup/download_hmdb51/
├── download_hmdb51.py       # Download script
├── README.md                 # This file
├── hmdb51_org.rar            # Original archive (can be deleted after extraction)
├── hmdb51_org/               # Extracted videos organized by action class
│   ├── apply_eye_makeup/
│   ├── apply_lipstick/
│   ├── archery/
│   └── ... (51 action classes total)
└── captions.txt              # Video-caption pairs for training
```

## Captions Format

The `captions.txt` file contains tab-separated video-caption pairs:
```
env_setup/download_hmdb51/hmdb51_org/action_name/video_name.avi    action_name
```

Example:
```
env_setup/download_hmdb51/hmdb51_org/apply_eye_makeup/v_ApplyEyeMakeup_g01_c01.avi    apply_eye_makeup
env_setup/download_hmdb51/hmdb51_org/apply_lipstick/v_ApplyLipstick_g01_c01.avi       apply_lipstick
```

## Preprocessing for Open-Sora v1.3

After downloading the HMDB51 dataset, you need to preprocess the videos for Open-Sora training.

### Requirements

All dependencies should already be installed in the opensora13 conda environment.

Or install manually:
```bash
pip install av opencv-python pillow torch torchvision numpy pandas tqdm
```

### Preprocessing Steps

Run the preprocessing script:

```bash
# IMPORTANT: Activate the opensora13 conda environment first!
conda activate /scratch/wc3013/conda-envs/opensora13

cd env_setup/download_hmdb51
python preprocess_hmdb51.py
```

**What the preprocessing does:**

1. **Center-crop to 480p**: Resizes and center-crops all videos to 640×480 pixels
2. **Resample to 24 fps**: Changes frame rate to 24 fps using temporal interpolation
3. **Crop to 45 frames**: Uniformly crops each video to exactly 45 frames
4. **Generate CSV metadata**: Creates `hmdb51_metadata.csv` with required fields
5. **Skip short videos**: Videos shorter than 45 frames after resampling are skipped
6. **Optional cleanup**: Prompts to delete original dataset to save disk space (~2GB)

### Output Structure

After preprocessing:

```
env_setup/download_hmdb51/
├── download_hmdb51.py              # Download script
├── preprocess_hmdb51.py            # Preprocessing script
├── README.md                        # This file
├── hmdb51_org/                      # Original downloaded videos
│   └── <action_classes>/           # 51 action folders
├── hmdb51_processed/                # Preprocessed videos
│   └── <action_classes>/           # Same structure as original
├── captions.txt                     # Original captions
└── hmdb51_metadata.csv              # Training metadata CSV
```

### CSV Format

The `hmdb51_metadata.csv` file contains the following columns:

| Column | Description | Value |
|--------|-------------|-------|
| `path` | Relative path to preprocessed video | e.g., `hmdb51_processed/apply_eye_makeup/video.mp4` |
| `num_frames` | Number of frames in video | 45 |
| `conditioning_frames` | Number of conditioning frames | 32 |
| `height` | Video height in pixels | 480 |
| `width` | Video width in pixels | 640 |
| `fps` | Frame rate | 24 |
| `text` | Video caption/action label | e.g., `apply_eye_makeup` |

### Usage in Open-Sora

Now you can use the preprocessed dataset for training:

1. **Add to your training config**:
   ```python
   # In your training config file
   data = dict(
       type="VideoTextDataset",
       data_path="/path/to/env_setup/download_hmdb51/hmdb51_metadata.csv",
       num_frames=45,
       frame_interval=1,
       image_size=(480, 640),
       ...
   )
   ```

2. **Or use directly**:
   ```bash
   # Point to the CSV file when training
   python scripts/train.py \
       --config configs/your_config.py \
       --csv_path hmdb51/hmdb51_metadata.csv
   ```

## Disk Space Management

After preprocessing completes successfully, the script will prompt you to delete the original dataset to save disk space:

- **Original dataset size**: ~2GB (compressed) + ~2GB (extracted) = ~4GB total
- **Preprocessed videos**: Significantly smaller due to resizing and frame reduction
- **Space savings**: Approximately ~2-3GB depending on your videos

When prompted, you can:
- Type `yes` or `y` to delete the original `hmdb51_org/` folder and `hmdb51_org.rar` file
- Type `no` or press Enter to keep the original dataset

**Note**: The original dataset is only deleted if preprocessing completes successfully. You can always re-download it later if needed.

## Notes

- The original HMDB51 dataset uses action class labels as ground truth
- Captions are generated based on the folder name (action category)
- Each action category contains multiple video clips showing different variations of that action
- Videos shorter than 45 frames after resampling to 24 fps are automatically skipped
- Preprocessing maintains the original directory structure for easy navigation
- Original dataset can be safely deleted after preprocessing to save disk space

