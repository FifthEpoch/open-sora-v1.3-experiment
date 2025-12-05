#!/usr/bin/env python3
"""
Hyperparameter sweep for fine-tuning configurations.

Tests different combinations of:
- Fine-tuning steps (20, 50, 100)
- Learning rates (1e-5, 5e-5, 1e-4)

Usage:
    python run_hp_sweep.py --data-csv path/to/metadata.csv --output-dir results/hp_sweep
"""

import argparse
import os
import sys
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for fine-tuning")
    parser.add_argument("--data-csv", type=str, required=True, help="Path to UCF-101 metadata CSV")
    parser.add_argument("--output-dir", type=str, default="naive_experiment/results/hp_sweep", 
                        help="Output directory for sweep results")
    parser.add_argument("--num-videos", type=int, default=3, 
                        help="Number of videos to test per configuration")
    parser.add_argument("--condition-frames", type=int, default=22, 
                        help="Number of conditioning frames")
    parser.add_argument("--checkpoint-path", type=str, default="hpcai-tech/OpenSora-STDiT-v4-360p",
                        help="Base model checkpoint path")
    parser.add_argument("--configs", type=str, default="default",
                        choices=["default", "extended", "minimal", "aggressive", "focused", "ultrafast", "test_20steps_2e4", "test_50steps_7e5"],
                        help="Which configuration set to test")
    return parser.parse_args()


# Define hyperparameter configurations to test
HP_CONFIGS = {
    # Default: Test Option 1 and Option 2
    "default": [
        {"name": "baseline_20steps_1e5", "steps": 20, "lr": 1e-5},    # Current baseline
        {"name": "option1_50steps_1e5", "steps": 50, "lr": 1e-5},     # Option 1: More steps
        {"name": "option2_20steps_5e5", "steps": 20, "lr": 5e-5},     # Option 2: Higher LR
    ],
    # Extended: More configurations
    "extended": [
        {"name": "baseline_20steps_1e5", "steps": 20, "lr": 1e-5},
        {"name": "option1_50steps_1e5", "steps": 50, "lr": 1e-5},
        {"name": "option2_20steps_5e5", "steps": 20, "lr": 5e-5},
        {"name": "combo_50steps_5e5", "steps": 50, "lr": 5e-5},       # Combination
        {"name": "aggressive_20steps_1e4", "steps": 20, "lr": 1e-4},  # Very aggressive
        {"name": "gentle_100steps_5e6", "steps": 100, "lr": 5e-6},    # Very gentle
    ],
    # Minimal: Just two options for quick testing
    "minimal": [
        {"name": "option1_50steps_1e5", "steps": 50, "lr": 1e-5},
        {"name": "option2_20steps_5e5", "steps": 20, "lr": 5e-5},
    ],
    # Aggressive: Build on Option 2's success with higher LR/more steps
    "aggressive": [
        {"name": "combo_50steps_5e5", "steps": 50, "lr": 5e-5},       # More steps + winning LR
        {"name": "combo_100steps_5e5", "steps": 100, "lr": 5e-5},     # Even more steps
        {"name": "high_lr_20steps_1e4", "steps": 20, "lr": 1e-4},     # 2x the winning LR
        {"name": "high_lr_50steps_1e4", "steps": 50, "lr": 1e-4},     # High LR + more steps
        {"name": "very_high_lr_20steps_2e4", "steps": 20, "lr": 2e-4}, # Push LR further
    ],
    # Focused: Test around the winning config (5e-5)
    "focused": [
        {"name": "lr_3e5_20steps", "steps": 20, "lr": 3e-5},          # Between 1e-5 and 5e-5
        {"name": "lr_7e5_20steps", "steps": 20, "lr": 7e-5},          # Between 5e-5 and 1e-4
        {"name": "lr_5e5_30steps", "steps": 30, "lr": 5e-5},          # Winning LR + slight more steps
        {"name": "lr_5e5_40steps", "steps": 40, "lr": 5e-5},          # Winning LR + more steps
    ],
    # Ultra-fast: High LR + very few steps (test speed vs quality tradeoff)
    # Theory: If 1e-4 @ 20 steps ≈ 5e-5 @ 20 steps, then higher LR with fewer steps might work
    "ultrafast": [
        {"name": "10steps_2e4", "steps": 10, "lr": 2e-4},             # 2x aggressive LR, half steps
        {"name": "10steps_5e4", "steps": 10, "lr": 5e-4},             # 5x aggressive LR
        {"name": "5steps_5e4", "steps": 5, "lr": 5e-4},               # Very few steps, high LR
        {"name": "5steps_1e3", "steps": 5, "lr": 1e-3},               # Push the limit
        {"name": "10steps_1e3", "steps": 10, "lr": 1e-3},             # 10x LR
        {"name": "15steps_2e4", "steps": 15, "lr": 2e-4},             # Slightly more steps at 2e-4
    ],
    # Single test: 20 steps @ 2e-4
    "test_20steps_2e4": [
        {"name": "20steps_2e4", "steps": 20, "lr": 2e-4},             # Compare to 15 and 50 steps
    ],
    # Single test: 50 steps @ 7e-5
    "test_50steps_7e5": [
        {"name": "50steps_7e5", "steps": 50, "lr": 7e-5},             # Between 5e-5 (10.55) and 1e-4 (9.25)
    ],
}


def run_single_config(
    config: dict,
    data_csv: str,
    output_dir: str,
    num_videos: int,
    condition_frames: int,
    checkpoint_path: str,
    shared_baseline_dir: str = None,
    is_first_config: bool = False,
) -> dict:
    """Run experiment with a single HP configuration."""
    
    config_name = config["name"]
    steps = config["steps"]
    lr = config["lr"]
    
    # Create output directory for this config
    config_output_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Testing configuration: {config_name}")
    print(f"  Steps: {steps}, LR: {lr}")
    print(f"  Output: {config_output_dir}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "naive_experiment" / "scripts" / "run_experiment.py"),
        "--data-csv", data_csv,
        "--output-dir", config_output_dir,
        "--num-videos", str(num_videos),
        "--condition-frames", str(condition_frames),
        "--finetune-steps", str(steps),
        "--finetune-lr", str(lr),
        "--checkpoint-path", checkpoint_path,
    ]
    
    # For subsequent configs, reuse shared baselines instead of regenerating
    if not is_first_config and shared_baseline_dir:
        cmd.append("--skip-baseline")
        # Copy baselines from first config to this config
        src_baseline = os.path.join(shared_baseline_dir, "baselines")
        dst_baseline = os.path.join(config_output_dir, "baselines")
        if os.path.exists(src_baseline) and not os.path.exists(dst_baseline):
            shutil.copytree(src_baseline, dst_baseline)
            print(f"  Copied baselines from {src_baseline}")
    
    start_time = time.time()
    
    # Run the experiment
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    
    elapsed_time = time.time() - start_time
    
    # Save logs
    with open(os.path.join(config_output_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(config_output_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)
    
    return {
        "config_name": config_name,
        "steps": steps,
        "lr": lr,
        "return_code": result.returncode,
        "elapsed_time": elapsed_time,
        "output_dir": config_output_dir,
        "success": result.returncode == 0,
    }


def load_metrics(config_output_dir: str) -> dict:
    """Load evaluation metrics from a config's output directory and compute averages."""
    # Try both possible filenames
    metrics_file = os.path.join(config_output_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(config_output_dir, "evaluation_metrics.json")
    if not os.path.exists(metrics_file):
        return {}
    
    with open(metrics_file) as f:
        metrics_list = json.load(f)
    
    if not metrics_list:
        return {}
    
    # metrics_list is a list of dicts with structure:
    # [{"video_idx": 0, "baseline": {"psnr": X, "ssim": Y, "lpips": Z}, "finetuned": {...}}, ...]
    
    # Compute averages for baseline and finetuned
    result = {"baseline": {}, "finetuned": {}}
    
    for key in ["baseline", "finetuned"]:
        psnr_vals = [m[key]["psnr"] for m in metrics_list if key in m and "psnr" in m[key]]
        ssim_vals = [m[key]["ssim"] for m in metrics_list if key in m and "ssim" in m[key]]
        lpips_vals = [m[key]["lpips"] for m in metrics_list if key in m and "lpips" in m[key]]
        
        if psnr_vals:
            result[key]["psnr"] = sum(psnr_vals) / len(psnr_vals)
        if ssim_vals:
            result[key]["ssim"] = sum(ssim_vals) / len(ssim_vals)
        if lpips_vals:
            result[key]["lpips"] = sum(lpips_vals) / len(lpips_vals)
    
    return {"average": result}


def summarize_results(results: list, output_dir: str):
    """Create a summary of all configurations tested."""
    
    summary = []
    
    for r in results:
        config_metrics = load_metrics(r["output_dir"])
        
        # Extract average metrics if available
        avg_metrics = config_metrics.get("average", {})
        
        summary.append({
            "config": r["config_name"],
            "steps": r["steps"],
            "lr": r["lr"],
            "success": r["success"],
            "elapsed_time_min": r["elapsed_time"] / 60,
            "avg_psnr_finetuned": avg_metrics.get("finetuned", {}).get("psnr"),
            "avg_ssim_finetuned": avg_metrics.get("finetuned", {}).get("ssim"),
            "avg_lpips_finetuned": avg_metrics.get("finetuned", {}).get("lpips"),
            "avg_psnr_baseline": avg_metrics.get("baseline", {}).get("psnr"),
            "avg_ssim_baseline": avg_metrics.get("baseline", {}).get("ssim"),
            "avg_lpips_baseline": avg_metrics.get("baseline", {}).get("lpips"),
        })
    
    # Save summary
    summary_file = os.path.join(output_dir, "hp_sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("="*80)
    print(f"{'Config':<30} {'Steps':>6} {'LR':>10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-"*80)
    
    for s in summary:
        psnr = f"{s['avg_psnr_finetuned']:.2f}" if s['avg_psnr_finetuned'] else "N/A"
        ssim = f"{s['avg_ssim_finetuned']:.4f}" if s['avg_ssim_finetuned'] else "N/A"
        lpips = f"{s['avg_lpips_finetuned']:.4f}" if s['avg_lpips_finetuned'] else "N/A"
        
        status = "✓" if s["success"] else "✗"
        print(f"{status} {s['config']:<28} {s['steps']:>6} {s['lr']:>10.0e} {psnr:>8} {ssim:>8} {lpips:>8}")
    
    print("="*80)
    print(f"\nFull results saved to: {summary_file}")
    
    # Find best config
    successful = [s for s in summary if s["success"] and s.get("avg_psnr_finetuned")]
    if successful:
        best_psnr = max(successful, key=lambda x: x["avg_psnr_finetuned"])
        best_ssim = max(successful, key=lambda x: x["avg_ssim_finetuned"])
        best_lpips = min(successful, key=lambda x: x["avg_lpips_finetuned"])
        
        print("\nBest configurations:")
        print(f"  Best PSNR:  {best_psnr['config']} (PSNR={best_psnr['avg_psnr_finetuned']:.2f})")
        print(f"  Best SSIM:  {best_ssim['config']} (SSIM={best_ssim['avg_ssim_finetuned']:.4f})")
        print(f"  Best LPIPS: {best_lpips['config']} (LPIPS={best_lpips['avg_lpips_finetuned']:.4f})")


def main():
    args = parse_args()
    
    # Get configurations to test
    configs = HP_CONFIGS[args.configs]
    
    print(f"Starting hyperparameter sweep at {datetime.now()}")
    print(f"Testing {len(configs)} configurations on {args.num_videos} videos each")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save sweep configuration
    sweep_config = {
        "timestamp": datetime.now().isoformat(),
        "data_csv": args.data_csv,
        "num_videos": args.num_videos,
        "condition_frames": args.condition_frames,
        "checkpoint_path": args.checkpoint_path,
        "configs": configs,
    }
    with open(os.path.join(args.output_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_config, f, indent=2)
    
    # Run each configuration
    results = []
    shared_baseline_dir = None  # Will be set to first config's output dir
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running configuration: {config['name']}")
        
        is_first = (i == 0)
        result = run_single_config(
            config=config,
            data_csv=args.data_csv,
            output_dir=args.output_dir,
            num_videos=args.num_videos,
            condition_frames=args.condition_frames,
            checkpoint_path=args.checkpoint_path,
            shared_baseline_dir=shared_baseline_dir,
            is_first_config=is_first,
        )
        results.append(result)
        
        # After first config succeeds, use its baselines for subsequent configs
        if is_first and result["success"]:
            shared_baseline_dir = result["output_dir"]
            print(f"  ✓ Will reuse baselines from {shared_baseline_dir} for subsequent configs")
        
        if not result["success"]:
            print(f"  ⚠ Configuration failed! Check logs at {result['output_dir']}")
    
    # Summarize results
    summarize_results(results, args.output_dir)
    
    print(f"\nHyperparameter sweep completed at {datetime.now()}")


if __name__ == "__main__":
    main()

