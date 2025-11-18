#!/usr/bin/env python3
"""
Hyperparameter Search Orchestrator with Checkpoint Recovery

This script runs systematic hyperparameter searches for the fine-tuning experiment,
testing learning rate, conditioning frame count, and fine-tuning steps.

Features:
- Isolated testing: One parameter varied at a time
- Checkpoint recovery: Resume from interrupted experiments
- Shared baseline: Reuse baseline inference across configurations
- Progress tracking: JSON-based state management
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GPUKeepalive:
    """Maintain GPU utilization during I/O operations to prevent job cancellation."""
    
    def __init__(self, device='cuda', target_utilization=0.35):
        self.device = device
        self.target_utilization = target_utilization
        self.running = False
        self.thread = None
        self.tensors = []
        
    def _keepalive_loop(self):
        """Run dummy matmul operations to maintain GPU utilization (~30-40%)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            
            logger_msg = f"GPU keepalive starting (target: {self.target_utilization*100:.0f}% utilization)"
            print(logger_msg)
            
            self.tensors = [
                torch.randn(4096, 4096, device=self.device) for _ in range(4)
            ]
            
            while self.running:
                for i in range(len(self.tensors)):
                    _ = torch.matmul(self.tensors[i], self.tensors[(i+1) % len(self.tensors)])
                time.sleep(0.01)
                
        except Exception as e:
            print(f"GPU keepalive error: {e}")
    
    def start(self):
        """Start the keepalive thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._keepalive_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the keepalive thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.tensors = []
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


def setup_logging(log_dir: Path):
    """Set up logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"hp_search_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_yaml_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_progress(progress_file: Path) -> Dict:
    """Load progress tracking JSON."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress: Dict, progress_file: Path):
    """Save progress tracking JSON."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def run_baseline_inference(
    data_csv: Path,
    output_dir: Path,
    condition_frames: int,
    logger: logging.Logger
) -> Tuple[bool, Path]:
    """
    Run baseline inference for all videos.
    Returns (success, manifest_path).
    """
    logger.info(f"Running baseline inference with {condition_frames} conditioning frames...")
    
    baseline_script = PROJECT_ROOT / "naive_experiment" / "scripts" / "baseline_inference.py"
    baseline_output = output_dir / "baselines"
    baseline_output.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", str(baseline_script),
        "--data-csv", str(data_csv),
        "--output-dir", str(baseline_output),
        "--condition-frames", str(condition_frames)
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        
        manifest_path = baseline_output / "baseline_manifest.csv"
        if not manifest_path.exists():
            logger.error(f"Baseline manifest not found at {manifest_path}")
            return False, manifest_path
        
        logger.info(f"✓ Baseline inference completed. Manifest: {manifest_path}")
        return True, manifest_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Baseline inference failed: {e}")
        logger.error(f"  stdout: {e.stdout}")
        logger.error(f"  stderr: {e.stderr}")
        return False, None


def run_single_configuration(
    config_name: str,
    param_value: any,
    baseline_manifest: Path,
    output_dir: Path,
    lr: float,
    condition_frames: int,
    finetune_steps: int,
    logger: logging.Logger,
    keepalive: GPUKeepalive,
    start_from_video: int = 0
) -> Tuple[bool, Dict]:
    """
    Run fine-tuning and inference for a single hyperparameter configuration.
    Returns (success, metrics_dict).
    """
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"{'='*60}")
    logger.info(f"  LR: {lr}")
    logger.info(f"  Conditioning frames: {condition_frames}")
    logger.info(f"  Fine-tuning steps: {finetune_steps}")
    if start_from_video > 0:
        logger.info(f"  Resuming from video {start_from_video}")
    
    # Use the naive experiment's run_experiment.py with specific parameters
    naive_script = PROJECT_ROOT / "naive_experiment" / "scripts" / "run_experiment.py"
    
    cmd = [
        "python", str(naive_script),
        "--baseline-manifest", str(baseline_manifest),
        "--output-dir", str(config_dir),
        "--lr", str(lr),
        "--condition-frames", str(condition_frames),
        "--finetune-steps", str(finetune_steps),
        "--start-from-video", str(start_from_video)
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        keepalive.start()
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        
        keepalive.stop()
        
        # Check for metrics.json
        metrics_file = config_dir / "metrics.json"
        if not metrics_file.exists():
            logger.error(f"✗ Metrics file not found at {metrics_file}")
            return False, {}
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        # Calculate aggregate metrics
        from naive_experiment.scripts.visualization.plot_metrics import extract_metrics
        
        metrics = extract_metrics(metrics_data)
        
        import numpy as np
        
        summary = {
            "config_name": config_name,
            "lr": lr,
            "condition_frames": condition_frames,
            "finetune_steps": finetune_steps,
            "num_videos": len(metrics['baseline']['psnr']),
            "baseline_psnr_mean": float(np.mean(metrics['baseline']['psnr'])) if metrics['baseline']['psnr'] else 0,
            "baseline_ssim_mean": float(np.mean(metrics['baseline']['ssim'])) if metrics['baseline']['ssim'] else 0,
            "baseline_lpips_mean": float(np.mean(metrics['baseline']['lpips'])) if metrics['baseline']['lpips'] else 0,
            "finetuned_psnr_mean": float(np.mean(metrics['finetuned']['psnr'])) if metrics['finetuned']['psnr'] else 0,
            "finetuned_ssim_mean": float(np.mean(metrics['finetuned']['ssim'])) if metrics['finetuned']['ssim'] else 0,
            "finetuned_lpips_mean": float(np.mean(metrics['finetuned']['lpips'])) if metrics['finetuned']['lpips'] else 0,
        }
        
        # Calculate improvements
        if summary['baseline_psnr_mean'] > 0:
            summary['psnr_improvement_pct'] = float(
                ((summary['finetuned_psnr_mean'] - summary['baseline_psnr_mean']) /
                 summary['baseline_psnr_mean']) * 100
            )
        if summary['baseline_ssim_mean'] > 0:
            summary['ssim_improvement_pct'] = float(
                ((summary['finetuned_ssim_mean'] - summary['baseline_ssim_mean']) /
                 summary['baseline_ssim_mean']) * 100
            )
        if summary['baseline_lpips_mean'] > 0:
            summary['lpips_improvement_pct'] = float(
                ((summary['baseline_lpips_mean'] - summary['finetuned_lpips_mean']) /
                 summary['baseline_lpips_mean']) * 100
            )
        
        # Add timing if available
        if 'timing' in metrics:
            timing = metrics['timing']
            if timing.get('baseline_inference'):
                summary['avg_baseline_inference_time_sec'] = float(np.mean(timing['baseline_inference']))
            if timing.get('finetune'):
                summary['avg_finetune_time_sec'] = float(np.mean(timing['finetune']))
            if timing.get('finetuned_inference'):
                summary['avg_finetuned_inference_time_sec'] = float(np.mean(timing['finetuned_inference']))
        
        logger.info(f"✓ Configuration {config_name} completed")
        logger.info(f"  PSNR: {summary['baseline_psnr_mean']:.4f} → {summary['finetuned_psnr_mean']:.4f} ({summary.get('psnr_improvement_pct', 0):+.2f}%)")
        logger.info(f"  SSIM: {summary['baseline_ssim_mean']:.4f} → {summary['finetuned_ssim_mean']:.4f} ({summary.get('ssim_improvement_pct', 0):+.2f}%)")
        logger.info(f"  LPIPS: {summary['baseline_lpips_mean']:.4f} → {summary['finetuned_lpips_mean']:.4f} ({summary.get('lpips_improvement_pct', 0):+.2f}%)")
        
        return True, summary
        
    except subprocess.CalledProcessError as e:
        keepalive.stop()
        logger.error(f"✗ Configuration {config_name} failed: {e}")
        logger.error(f"  stdout: {e.stdout[-1000:]}")  # Last 1000 chars
        logger.error(f"  stderr: {e.stderr[-1000:]}")
        return False, {}
    except Exception as e:
        keepalive.stop()
        logger.error(f"✗ Unexpected error in {config_name}: {e}")
        return False, {}


def run_search(
    search_type: str,
    config_path: Path,
    data_csv: Path,
    results_dir: Path,
    logger: logging.Logger
):
    """Run a complete hyperparameter search."""
    # Load configuration
    config = load_yaml_config(config_path)
    search_name = config['search_name']
    search_param = config['search_param']
    fixed_params = config['fixed_params']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {search_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Search parameter: {search_param}")
    logger.info(f"Fixed parameters: {fixed_params}")
    
    # Setup output directories
    search_results_dir = results_dir / search_name
    search_results_dir.mkdir(parents=True, exist_ok=True)
    
    progress_file = results_dir / "progress.json"
    progress = load_progress(progress_file)
    
    if search_name not in progress:
        progress[search_name] = {}
    
    # Get parameter values to test
    param_key = f"{search_param}_values"
    if param_key not in config:
        logger.error(f"Configuration missing {param_key}")
        return
    
    param_configs = config[param_key]
    
    # Initialize GPU keepalive
    keepalive = GPUKeepalive()
    
    # Run baseline inference (if not already done for this conditioning frame count)
    condition_frames = fixed_params['condition_frames']
    baseline_cache_key = f"baseline_cf{condition_frames}"
    
    if baseline_cache_key not in progress:
        success, baseline_manifest = run_baseline_inference(
            data_csv=data_csv,
            output_dir=search_results_dir / "shared_baseline",
            condition_frames=condition_frames,
            logger=logger
        )
        
        if not success:
            logger.error("Failed to run baseline inference. Aborting.")
            return
        
        progress[baseline_cache_key] = {
            "status": "completed",
            "manifest_path": str(baseline_manifest),
            "condition_frames": condition_frames
        }
        save_progress(progress, progress_file)
    else:
        baseline_manifest = Path(progress[baseline_cache_key]['manifest_path'])
        logger.info(f"Using cached baseline from: {baseline_manifest}")
    
    # Run each configuration
    results_summary = []
    
    for param_config in param_configs:
        config_name = param_config['name']
        param_value = param_config['value']
        
        # Check if already completed
        if config_name in progress[search_name]:
            status = progress[search_name][config_name].get('status')
            if status == 'completed':
                logger.info(f"✓ {config_name} already completed. Skipping.")
                results_summary.append(progress[search_name][config_name].get('summary', {}))
                continue
            elif status == 'in_progress':
                # Resume from last video
                start_from = progress[search_name][config_name].get('last_video', 0) + 1
                logger.info(f"Resuming {config_name} from video {start_from}")
            else:
                start_from = 0
        else:
            start_from = 0
            progress[search_name][config_name] = {'status': 'pending'}
            save_progress(progress, progress_file)
        
        # Mark as in progress
        progress[search_name][config_name]['status'] = 'in_progress'
        save_progress(progress, progress_file)
        
        # Determine parameters based on search type
        if search_param == 'lr':
            lr = param_value
            cf = fixed_params['condition_frames']
            steps = fixed_params['finetune_steps']
        elif search_param == 'condition_frames':
            lr = fixed_params['lr']
            cf = param_value
            steps = fixed_params['finetune_steps']
        elif search_param == 'finetune_steps':
            lr = fixed_params['lr']
            cf = fixed_params['condition_frames']
            steps = param_value
        else:
            logger.error(f"Unknown search parameter: {search_param}")
            continue
        
        # Run configuration
        success, summary = run_single_configuration(
            config_name=config_name,
            param_value=param_value,
            baseline_manifest=baseline_manifest,
            output_dir=search_results_dir,
            lr=lr,
            condition_frames=cf,
            finetune_steps=steps,
            logger=logger,
            keepalive=keepalive,
            start_from_video=start_from
        )
        
        if success:
            progress[search_name][config_name]['status'] = 'completed'
            progress[search_name][config_name]['summary'] = summary
            results_summary.append(summary)
        else:
            progress[search_name][config_name]['status'] = 'failed'
        
        save_progress(progress, progress_file)
    
    # Save summary
    summary_file = search_results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Search {search_name} completed!")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter search with checkpoint recovery"
    )
    parser.add_argument(
        "--search-type",
        type=str,
        required=True,
        choices=['lr', 'condition_frames', 'steps'],
        help="Type of hyperparameter search to run"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_metadata.csv",
        help="Path to UCF-101 metadata CSV"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Override default results directory"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    hp_search_dir = PROJECT_ROOT / "naive_experiment" / "hyperparameter_search"
    
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = hp_search_dir / "results"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = results_dir / "logs"
    logger = setup_logging(log_dir)
    
    logger.info(f"Hyperparameter Search - {args.search_type}")
    logger.info(f"Results directory: {results_dir}")
    
    # Map search type to config file
    config_map = {
        'lr': 'lr_search.yaml',
        'condition_frames': 'condition_frames_search.yaml',
        'steps': 'steps_search.yaml'
    }
    
    config_path = hp_search_dir / "configs" / config_map[args.search_type]
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Run search
    run_search(
        search_type=args.search_type,
        config_path=config_path,
        data_csv=Path(args.data_csv),
        results_dir=results_dir,
        logger=logger
    )
    
    logger.info("All done!")


if __name__ == "__main__":
    main()

