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

from opensora.datasets.aspect import get_image_size
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
    parser.add_argument("--data-csv", type=str, required=True, help="Path to UCF-101 metadata CSV")
    parser.add_argument("--checkpoint-path", type=str, default="hpcai-tech/OpenSora-STDiT-v4", help="Open-Sora STDiT checkpoint path or HuggingFace ID")
    parser.add_argument("--vae-path", type=str, default="hpcai-tech/OpenSora-VAE-v1.3", help="Open-Sora VAE checkpoint path or HuggingFace ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for all results")
    parser.add_argument("--num-videos", type=int, default=None, help="Number of videos to process (None = all)")
    parser.add_argument("--condition-frames", type=int, default=22, help="Number of conditioning frames")
    parser.add_argument("--finetune-steps", type=int, default=20, help="Number of fine-tuning steps")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Fine-tuning learning rate")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline generation (if already done)")
    parser.add_argument("--skip-finetuning", action="store_true", help="Skip fine-tuning (evaluate existing results)")
    
    args = parser.parse_args()
    
    # Setup
    logger = create_logger()
    # Use absolute output directory to avoid relative path issues in manifests/eval
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
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
            "--save-dir", str(baseline_output_dir),
            "--condition-frames", str(args.condition_frames),
        ]
        # Only pass checkpoint paths if explicitly provided (not defaults)
        if args.checkpoint_path and args.checkpoint_path not in ["hpcai-tech/OpenSora-STDiT-v4"]:
            cmd.extend(["--checkpoint-path", args.checkpoint_path])
        if args.vae_path and args.vae_path not in ["hpcai-tech/OpenSora-VAE-v1.3"]:
            cmd.extend(["--vae-path", args.vae_path])
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
        target_height, target_width = get_image_size("480p", "4:3")
        
        for idx, row in tqdm(baseline_df.iterrows(), total=len(baseline_df), desc="Fine-tuning"):
            video_idx = row['video_idx']
            original_path = row['original_path']
            caption = row.get('caption', row.get('text', 'video'))
            
            # Make path absolute
            if not os.path.isabs(original_path):
                original_path = os.path.join(Path(args.data_csv).parent, original_path)
            
            logger.info(f"\nProcessing video {video_idx}: {Path(original_path).name}")
            
            # Create truncated video (first 22 frames) for training
            logger.info("  Creating truncated video (first 22 frames) for training...")
            from naive_experiment.scripts.single_video_finetune import create_truncated_video
            truncated_video_path = create_truncated_video(original_path, num_frames=22)
            logger.info(f"  Truncated video saved to: {truncated_video_path}")
            
            # Create single-video CSV for fine-tuning (using truncated video)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_csv = f.name
                pd.DataFrame([{
                    'path': truncated_video_path,
                    'text': caption,
                    'num_frames': 22,  # Only first 22 frames for training
                    'height': int(target_height),
                    'width': int(target_width),
                    'fps': 24,
                    'aspect_ratio': 1.33,  # 4:3
                }]).to_csv(temp_csv, index=False)
            
            # Fine-tune checkpoint directory for this video
            video_ckpt_dir = finetuned_checkpoints_dir / f"video_{video_idx:04d}"
            video_ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            # Fine-tune
            logger.info(f"  Fine-tuning for {args.finetune_steps} steps...")
            
            # Create temporary fine-tuning config with correct paths
            import tempfile
            import shutil
            with open(finetune_config, 'r') as f:
                finetune_config_content = f.read()
            
            # Update config: outputs, epochs (since 1 epoch = 1 step with batch_size=1 and 1 video), ckpt_every, lr
            finetune_config_content = finetune_config_content.replace(
                'outputs = "naive_experiment/results/finetuned_checkpoints"',
                f'outputs = "{video_ckpt_dir}"'
            )
            finetune_config_content = finetune_config_content.replace(
                'epochs = 1',
                f'epochs = {args.finetune_steps}'  # With batch_size=1 and 1 video, 1 epoch = 1 step
            )
            finetune_config_content = finetune_config_content.replace(
                'ckpt_every = 50',
                f'ckpt_every = 1'  # Save every step (since we want the final checkpoint)
            )
            finetune_config_content = finetune_config_content.replace(
                'lr = 1e-5',
                f'lr = {args.finetune_lr}'
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_finetune_config = f.name
                f.write(finetune_config_content)
            
            # Update the config with data path before running training
            from mmengine.config import Config
            train_cfg = Config.fromfile(temp_finetune_config)
            train_cfg.dataset.data_path = temp_csv
            # Ensure single-sample training actually yields steps
            train_cfg.batch_size = 1
            train_cfg.drop_last = False
            train_cfg.shuffle = False
            train_cfg.model.from_pretrained = args.checkpoint_path
            train_cfg.vae.from_pretrained = args.vae_path
            
            # Save updated config
            train_cfg.dump(temp_finetune_config)
            
            # Run fine-tuning via train.py
            train_script = Path(__file__).parent.parent.parent / "scripts" / "train.py"
            finetune_cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node", "1",
                str(train_script),
                temp_finetune_config,
                "--data-path", temp_csv,
                "--ckpt-path", args.checkpoint_path,
                "--outputs", str(video_ckpt_dir),
            ]
            
            # Capture fine-tune stdout/stderr to files for debugging
            finetune_stdout_path = video_ckpt_dir / "finetune_stdout.log"
            finetune_stderr_path = video_ckpt_dir / "finetune_stderr.log"
            with open(finetune_stdout_path, "w") as f_out, open(finetune_stderr_path, "w") as f_err:
                finetune_result = subprocess.run(finetune_cmd, stdout=f_out, stderr=f_err, text=True)
            # Log directory contents after training attempt
            try:
                logger.info(f"  Contents of checkpoint dir ({video_ckpt_dir}):")
                for p in sorted(video_ckpt_dir.iterdir()):
                    logger.info(f"    - {p.name}{'/' if p.is_dir() else ''}")
            except Exception as e:
                logger.warning(f"  Could not list checkpoint dir: {e}")

            if finetune_result.returncode != 0:
                logger.error(f"  Fine-tuning failed for video {video_idx}")
                try:
                    tail_err = Path(finetune_stderr_path).read_text()[-2000:]
                except Exception:
                    tail_err = "<unable to read stderr log>"
                logger.error(f"  See logs:\n    stdout: {finetune_stdout_path}\n    stderr: {finetune_stderr_path}")
                logger.error(f"  Stderr tail:\n{tail_err}")
                finetuned_results.append({
                    'video_idx': video_idx,
                    'original_path': original_path,
                    'finetuned_output': None,
                    'error': f"Fine-tuning failed; see {finetune_stderr_path}",
                })
                os.unlink(temp_finetune_config)
                continue
            
            # Find the saved checkpoint
            # train.py creates an experiment subdir: outputs/{index}-{model_name}/epoch{epoch}-global_step{step}/model
            exp_subdirs = sorted([d for d in video_ckpt_dir.iterdir() if d.is_dir()])
            if not exp_subdirs:
                logger.error(f"  No experiment directory found under {video_ckpt_dir}")
                finetuned_results.append({
                    'video_idx': video_idx,
                    'original_path': original_path,
                    'finetuned_output': None,
                    'error': f"No experiment directory found after fine-tuning",
                })
                os.unlink(temp_finetune_config)
                continue
            latest_exp_dir = exp_subdirs[-1]
            # Since we save every step (ckpt_every=1), find the latest epoch dir
            epoch_dirs = sorted([d for d in latest_exp_dir.iterdir() if d.is_dir() and d.name.startswith("epoch")])
            if not epoch_dirs:
                logger.error(f"  No checkpoint directories found in {latest_exp_dir}")
                finetuned_results.append({
                    'video_idx': video_idx,
                    'original_path': original_path,
                    'finetuned_output': None,
                    'error': f"No checkpoint found after fine-tuning",
                })
                os.unlink(temp_finetune_config)
                continue
            
            # Use the latest checkpoint (last epoch/step)
            latest_ckpt_dir = epoch_dirs[-1]
            expected_ckpt_subdir = latest_ckpt_dir / "model"
            if not expected_ckpt_subdir.exists():
                logger.error(f"  Checkpoint model directory not found at {expected_ckpt_subdir}")
                finetuned_results.append({
                    'video_idx': video_idx,
                    'original_path': original_path,
                    'finetuned_output': None,
                    'error': f"Checkpoint model directory not found at {expected_ckpt_subdir}",
                })
                os.unlink(temp_finetune_config)
                continue
            
            logger.info(f"  Using checkpoint: {latest_ckpt_dir}")
            
            # The checkpoint path for loading is the parent of 'model' (epoch{epoch}-global_step{global_step})
            actual_ckpt_path = expected_ckpt_subdir.parent
            
            # Generate with fine-tuned checkpoint
            logger.info("  Generating continuation with fine-tuned model...")
            
            # Update finetuned inference config (only need to replace fine-tuned checkpoint path)
            with open(finetuned_inference_config, 'r') as f:
                inference_config = f.read()
            inference_config = inference_config.replace(
                'from_pretrained="path/to/finetuned/checkpoint"',
                f'from_pretrained="{actual_ckpt_path}"'
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_inference_config = f.name
                f.write(inference_config)
            
            try:
                cmd = [
                    sys.executable,
                    str(scripts_dir / "finetuned_inference.py"),
                    "--config", temp_inference_config,
                    "--finetuned-checkpoint", str(actual_ckpt_path),
                    "--video-path", original_path,
                    "--caption", caption,
                    "--save-dir", str(finetuned_output_dir),
                    "--condition-frames", str(args.condition_frames),
                    "--video-idx", str(video_idx),
                ]
                result = run_command(cmd, logger, check=False)
                if result.returncode == 0:
                    # Extract only the last line (the output path) to avoid VAE loading messages
                    stdout_lines = result.stdout.strip().split('\n')
                    finetuned_output = stdout_lines[-1] if stdout_lines else None
                    logger.info(f"  Captured stdout: {len(stdout_lines)} lines, last line: {finetuned_output}")
                    # Validate it's actually a path (contains '/' and ends with '.mp4')
                    if finetuned_output and '/' in finetuned_output and finetuned_output.endswith('.mp4'):
                        logger.info(f"  ✓ Valid output path extracted: {finetuned_output}")
                        finetuned_results.append({
                            'video_idx': video_idx,
                            'original_path': original_path,
                            'finetuned_output': finetuned_output,
                            'caption': caption,
                        })
                    else:
                        logger.warning(f"  ✗ Could not parse output path from stdout. Last line: {finetuned_output}")
                        logger.warning(f"  Full stdout (first 1000 chars): {result.stdout[:1000]}...")
                        logger.warning(f"  Full stdout (last 500 chars): ...{result.stdout[-500:]}")
                        finetuned_results.append({
                            'video_idx': video_idx,
                            'original_path': original_path,
                            'finetuned_output': None,
                            'error': f"Could not parse output path from stdout",
                        })
                else:
                    logger.warning(f"  Failed to generate for video {video_idx}")
                    logger.error(f"  Error output:\n{result.stderr}")
                    finetuned_results.append({
                        'video_idx': video_idx,
                        'original_path': original_path,
                        'finetuned_output': None,
                        'error': result.stderr,
                    })
            finally:
                os.unlink(temp_inference_config)
                os.unlink(temp_finetune_config)
                os.unlink(temp_csv)
            
            # Do NOT clean up checkpoints/logs; keep for debugging and evaluation
            logger.info("  Keeping fine-tuned checkpoint directory for inspection.")
        
        # Save fine-tuned manifest
        finetuned_output_dir.mkdir(parents=True, exist_ok=True)
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
    logger.info(f"Loading baseline manifest: {baseline_manifest}")
    baseline_df = pd.read_csv(baseline_manifest)
    logger.info(f"Baseline manifest has {len(baseline_df)} rows, columns: {list(baseline_df.columns)}")
    logger.info(f"Baseline manifest sample:\n{baseline_df.head(2)}")
    
    logger.info(f"Loading finetuned manifest: {finetuned_manifest}")
    finetuned_df = pd.read_csv(finetuned_manifest)
    logger.info(f"Finetuned manifest has {len(finetuned_df)} rows, columns: {list(finetuned_df.columns)}")
    logger.info(f"Finetuned manifest sample:\n{finetuned_df.head(2)}")
    
    merged_df = baseline_df.merge(
        finetuned_df[['video_idx', 'finetuned_output']],
        on='video_idx',
        how='left'
    )
    
    merged_manifest = output_dir / "experiment_manifest.csv"
    merged_df.to_csv(merged_manifest, index=False)
    logger.info(f"Merged manifest saved to: {merged_manifest}")
    logger.info(f"Merged manifest has {len(merged_df)} rows, columns: {list(merged_df.columns)}")
    logger.info(f"Merged manifest sample:\n{merged_df.head(2)}")
    
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

