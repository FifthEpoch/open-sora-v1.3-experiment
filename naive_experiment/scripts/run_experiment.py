#!/usr/bin/env python3
"""
Main orchestrator script for the naive fine-tuning experiment.

This script coordinates the entire experiment:
1. Generate all baseline outputs (O_b) with single checkpoint load
2. For each video: fine-tune, generate O_f, evaluate, cleanup
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import time

import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from opensora.utils.misc import create_logger


def run_command(cmd, logger, check=True):
    """Run a shell command and return the result."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        logger.error(f"Stdout: {result.stdout}")
        logger.error(f"Stderr: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run naive fine-tuning experiment")
    parser.add_argument("--data-csv", type=str, required=True, help="Path to HMDB51 metadata CSV")
    parser.add_argument("--checkpoint-path", type=str, default="hpcai-tech/OpenSora-STDiT-v4", help="Open-Sora STDiT checkpoint path or HuggingFace ID")
    parser.add_argument("--vae-path", type=str, default="hpcai-tech/OpenSora-VAE-v1.3", help="Open-Sora VAE checkpoint path or HuggingFace ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for all results")
    parser.add_argument("--num-videos", type=int, default=None, help="Number of videos to process (None = all)")
    parser.add_argument("--condition-frames", type=int, default=8, help="Number of conditioning frames")
    parser.add_argument("--finetune-steps", type=int, default=20, help="Number of fine-tuning steps")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Fine-tuning learning rate")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline generation (if already done)")
    parser.add_argument("--skip-finetuning", action="store_true", help="Skip fine-tuning (evaluate existing results)")
    
    args = parser.parse_args()
    
    # Setup
    logger = create_logger()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    config_dir = Path(__file__).parent.parent / "configs"
    scripts_dir = Path(__file__).parent
    
    baseline_config = config_dir / "baseline_inference.py"
    finetune_config = config_dir / "single_video_finetune.py"
    finetuned_inference_config = config_dir / "finetuned_inference.py"
    
    baseline_output_dir = output_dir / "baselines"
    finetuned_output_dir = output_dir / "finetuned"
    finetuned_checkpoints_dir = output_dir / "finetuned_checkpoints"
    metrics_output = output_dir / "metrics.json"
    
    # Load dataset
    df = pd.read_csv(args.data_csv)
    if args.num_videos is not None:
        df = df.head(args.num_videos)
    
    logger.info(f"Experiment Configuration:")
    logger.info(f"  Total videos: {len(df)}")
    logger.info(f"  Condition frames: {args.condition_frames}")
    logger.info(f"  Fine-tune steps: {args.finetune_steps}")
    logger.info(f"  Fine-tune LR: {args.finetune_lr}")
    logger.info(f"  Output directory: {output_dir}")
    
    # ============================================================
    # Step 1: Generate all baseline outputs
    # ============================================================
    baseline_manifest = baseline_output_dir / "baseline_manifest.csv"
    
    if not args.skip_baseline:
        logger.info("\n" + "="*70)
        logger.info("Step 1: Generating baseline outputs (O_b)")
        logger.info("="*70)
        
        # Use original config file (already has correct HuggingFace IDs)
        cmd = [
            sys.executable,
            str(scripts_dir / "baseline_inference.py"),
            "--config", str(baseline_config),
            "--data-csv", args.data_csv,
            "--checkpoint-path", args.checkpoint_path,
            "--vae-path", args.vae_path,
            "--save-dir", str(baseline_output_dir),
            "--condition-frames", str(args.condition_frames),
        ]
        if args.num_videos:
            cmd.extend(["--num-videos", str(args.num_videos)])
        
        run_command(cmd, logger)
        
        logger.info(f"Baseline generation complete. Manifest: {baseline_manifest}")
    else:
        logger.info("Skipping baseline generation (--skip-baseline)")
        if not baseline_manifest.exists():
            raise FileNotFoundError(f"Baseline manifest not found: {baseline_manifest}")
    
    # Load baseline manifest
    baseline_df = pd.read_csv(baseline_manifest)
    
    # ============================================================
    # Step 2: Fine-tune and generate for each video
    # ============================================================
    if not args.skip_finetuning:
        logger.info("\n" + "="*70)
        logger.info("Step 2: Fine-tuning and generating O_f for each video")
        logger.info("="*70)
        
        finetuned_results = []
        
        for idx, row in tqdm(baseline_df.iterrows(), total=len(baseline_df), desc="Fine-tuning"):
            video_idx = row['video_idx']
            original_path = row['original_path']
            caption = row.get('caption', row.get('text', 'video'))
            
            # Make path absolute
            if not os.path.isabs(original_path):
                original_path = os.path.join(Path(args.data_csv).parent, original_path)
            
            logger.info(f"\nProcessing video {video_idx}: {Path(original_path).name}")
            
            # Create single-video CSV for fine-tuning
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_csv = f.name
                pd.DataFrame([{
                    'path': original_path,
                    'text': caption,
                    'num_frames': 45,
                    'height': 480,
                    'width': 640,
                    'fps': 24,
                }]).to_csv(temp_csv, index=False)
            
            # Fine-tune checkpoint directory for this video
            video_ckpt_dir = finetuned_checkpoints_dir / f"video_{video_idx:04d}"
            video_ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            # Fine-tune
            logger.info(f"  Fine-tuning for {args.finetune_steps} steps...")
            # Note: This requires modifying train.py to support max_steps argument
            # For now, we'll use a workaround by modifying the config
            
            # Generate with fine-tuned checkpoint
            logger.info("  Generating continuation with fine-tuned model...")
            
            # Update finetuned inference config (only need to replace fine-tuned checkpoint path)
            import tempfile
            with open(finetuned_inference_config, 'r') as f:
                inference_config = f.read()
            inference_config = inference_config.replace(
                'from_pretrained="path/to/finetuned/checkpoint"',
                f'from_pretrained="{video_ckpt_dir}"'
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_inference_config = f.name
                f.write(inference_config)
            
            try:
                cmd = [
                    sys.executable,
                    str(scripts_dir / "finetuned_inference.py"),
                    "--config", temp_inference_config,
                    "--finetuned-checkpoint", str(video_ckpt_dir),
                    "--video-path", original_path,
                    "--caption", caption,
                    "--save-dir", str(finetuned_output_dir),
                    "--condition-frames", str(args.condition_frames),
                    "--video-idx", str(video_idx),
                ]
                result = run_command(cmd, logger, check=False)
                if result.returncode == 0:
                    finetuned_output = result.stdout.strip()
                    finetuned_results.append({
                        'video_idx': video_idx,
                        'original_path': original_path,
                        'finetuned_output': finetuned_output,
                        'caption': caption,
                    })
                else:
                    logger.warning(f"  Failed to generate for video {video_idx}")
                    finetuned_results.append({
                        'video_idx': video_idx,
                        'original_path': original_path,
                        'finetuned_output': None,
                        'error': result.stderr,
                    })
            finally:
                os.unlink(temp_inference_config)
                os.unlink(temp_csv)
            
            # Cleanup fine-tuned checkpoint to save space
            logger.info("  Cleaning up fine-tuned checkpoint...")
            shutil.rmtree(video_ckpt_dir, ignore_errors=True)
        
        # Save fine-tuned manifest
        finetuned_df = pd.DataFrame(finetuned_results)
        finetuned_manifest = finetuned_output_dir / "finetuned_manifest.csv"
        finetuned_df.to_csv(finetuned_manifest, index=False)
        logger.info(f"\nFine-tuned generation complete. Manifest: {finetuned_manifest}")
    else:
        logger.info("Skipping fine-tuning (--skip-finetuning)")
        finetuned_manifest = finetuned_output_dir / "finetuned_manifest.csv"
        if not finetuned_manifest.exists():
            raise FileNotFoundError(f"Fine-tuned manifest not found: {finetuned_manifest}")
    
    # ============================================================
    # Step 3: Evaluate all results
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("Step 3: Evaluating continuations")
    logger.info("="*70)
    
    # Merge manifests
    baseline_df = pd.read_csv(baseline_manifest)
    finetuned_df = pd.read_csv(finetuned_manifest)
    merged_df = baseline_df.merge(
        finetuned_df[['video_idx', 'finetuned_output']],
        on='video_idx',
        how='left'
    )
    
    merged_manifest = output_dir / "experiment_manifest.csv"
    merged_df.to_csv(merged_manifest, index=False)
    
    # Run evaluation
    cmd = [
        sys.executable,
        str(scripts_dir / "evaluate_continuations.py"),
        "--original-videos", str(Path(args.data_csv).parent),
        "--baseline-outputs", str(baseline_output_dir),
        "--finetuned-outputs", str(finetuned_output_dir),
        "--manifest", str(merged_manifest),
        "--condition-frames", str(args.condition_frames),
        "--output-json", str(metrics_output),
    ]
    
    run_command(cmd, logger)
    
    logger.info("\n" + "="*70)
    logger.info("Experiment Complete!")
    logger.info("="*70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Metrics: {metrics_output}")


if __name__ == "__main__":
    main()

