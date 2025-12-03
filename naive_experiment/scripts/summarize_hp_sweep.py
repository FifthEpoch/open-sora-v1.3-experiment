#!/usr/bin/env python3
"""
Summarize results from an existing hyperparameter sweep.

Usage:
    python summarize_hp_sweep.py --sweep-dir /path/to/hp_sweep
"""

import argparse
import os
import json


def load_metrics(config_output_dir: str) -> dict:
    """Load evaluation metrics from a config's output directory and compute averages."""
    # Try both possible filenames
    metrics_file = os.path.join(config_output_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(config_output_dir, "evaluation_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"  Warning: No metrics file found in {config_output_dir}")
        return {}
    
    print(f"  Loading metrics from: {metrics_file}")
    with open(metrics_file) as f:
        metrics_list = json.load(f)
    
    if not metrics_list:
        return {}
    
    # metrics_list is a list of dicts with structure:
    # [{"video_idx": 0, "baseline": {"psnr": X, "ssim": Y, "lpips": Z}, "finetuned": {...}}, ...]
    
    # Compute averages for baseline and finetuned
    result = {"baseline": {}, "finetuned": {}}
    
    for key in ["baseline", "finetuned"]:
        psnr_vals = [m[key]["psnr"] for m in metrics_list if key in m and "psnr" in m.get(key, {})]
        ssim_vals = [m[key]["ssim"] for m in metrics_list if key in m and "ssim" in m.get(key, {})]
        lpips_vals = [m[key]["lpips"] for m in metrics_list if key in m and "lpips" in m.get(key, {})]
        
        if psnr_vals:
            result[key]["psnr"] = sum(psnr_vals) / len(psnr_vals)
        if ssim_vals:
            result[key]["ssim"] = sum(ssim_vals) / len(ssim_vals)
        if lpips_vals:
            result[key]["lpips"] = sum(lpips_vals) / len(lpips_vals)
    
    return {"average": result}


def main():
    parser = argparse.ArgumentParser(description="Summarize HP sweep results")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Path to sweep output directory")
    args = parser.parse_args()
    
    sweep_dir = args.sweep_dir
    
    # Load sweep config to get the configurations that were tested
    sweep_config_file = os.path.join(sweep_dir, "sweep_config.json")
    if os.path.exists(sweep_config_file):
        with open(sweep_config_file) as f:
            sweep_config = json.load(f)
        configs = sweep_config.get("configs", [])
    else:
        # Infer configs from subdirectories
        configs = []
        for name in os.listdir(sweep_dir):
            subdir = os.path.join(sweep_dir, name)
            if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "metrics.json")):
                # Try to parse name for steps and lr
                parts = name.split("_")
                try:
                    steps = int([p for p in parts if "steps" in p][0].replace("steps", ""))
                    lr_str = [p for p in parts if "e" in p.lower()][-1]
                    lr = float(lr_str.replace("e", "e-") if "e-" not in lr_str.lower() else lr_str)
                except:
                    steps = 0
                    lr = 0
                configs.append({"name": name, "steps": steps, "lr": lr})
    
    print(f"Found {len(configs)} configurations to summarize")
    
    # Load metrics for each config
    summary = []
    for config in configs:
        config_name = config["name"]
        config_dir = os.path.join(sweep_dir, config_name)
        
        if not os.path.isdir(config_dir):
            print(f"  Skipping {config_name}: directory not found")
            continue
        
        print(f"\nProcessing: {config_name}")
        config_metrics = load_metrics(config_dir)
        avg_metrics = config_metrics.get("average", {})
        
        summary.append({
            "config": config_name,
            "steps": config.get("steps", 0),
            "lr": config.get("lr", 0),
            "success": bool(avg_metrics),
            "avg_psnr_finetuned": avg_metrics.get("finetuned", {}).get("psnr"),
            "avg_ssim_finetuned": avg_metrics.get("finetuned", {}).get("ssim"),
            "avg_lpips_finetuned": avg_metrics.get("finetuned", {}).get("lpips"),
            "avg_psnr_baseline": avg_metrics.get("baseline", {}).get("psnr"),
            "avg_ssim_baseline": avg_metrics.get("baseline", {}).get("ssim"),
            "avg_lpips_baseline": avg_metrics.get("baseline", {}).get("lpips"),
        })
    
    # Save summary
    summary_file = os.path.join(sweep_dir, "hp_sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "="*100)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("="*100)
    print(f"{'Config':<30} {'Steps':>6} {'LR':>10} {'PSNR(ft)':>10} {'SSIM(ft)':>10} {'LPIPS(ft)':>10} {'PSNR(bl)':>10}")
    print("-"*100)
    
    for s in summary:
        psnr_ft = f"{s['avg_psnr_finetuned']:.2f}" if s['avg_psnr_finetuned'] else "N/A"
        ssim_ft = f"{s['avg_ssim_finetuned']:.4f}" if s['avg_ssim_finetuned'] else "N/A"
        lpips_ft = f"{s['avg_lpips_finetuned']:.4f}" if s['avg_lpips_finetuned'] else "N/A"
        psnr_bl = f"{s['avg_psnr_baseline']:.2f}" if s['avg_psnr_baseline'] else "N/A"
        
        lr_str = f"{s['lr']:.0e}" if s['lr'] else "N/A"
        status = "✓" if s["success"] else "✗"
        print(f"{status} {s['config']:<28} {s['steps']:>6} {lr_str:>10} {psnr_ft:>10} {ssim_ft:>10} {lpips_ft:>10} {psnr_bl:>10}")
    
    print("="*100)
    print(f"\nFull results saved to: {summary_file}")
    
    # Find best config
    successful = [s for s in summary if s["success"] and s.get("avg_psnr_finetuned")]
    if successful:
        best_psnr = max(successful, key=lambda x: x["avg_psnr_finetuned"])
        best_ssim = max(successful, key=lambda x: x["avg_ssim_finetuned"])
        best_lpips = min(successful, key=lambda x: x["avg_lpips_finetuned"])
        
        print("\nBest configurations (fine-tuned):")
        print(f"  Best PSNR:  {best_psnr['config']} (PSNR={best_psnr['avg_psnr_finetuned']:.2f})")
        print(f"  Best SSIM:  {best_ssim['config']} (SSIM={best_ssim['avg_ssim_finetuned']:.4f})")
        print(f"  Best LPIPS: {best_lpips['config']} (LPIPS={best_lpips['avg_lpips_finetuned']:.4f})")
        
        # Compare to baseline
        if best_psnr.get('avg_psnr_baseline'):
            print(f"\nComparison to baseline:")
            print(f"  PSNR improvement: {best_psnr['avg_psnr_finetuned'] - best_psnr['avg_psnr_baseline']:+.2f}")
        if best_ssim.get('avg_ssim_baseline'):
            print(f"  SSIM improvement: {best_ssim['avg_ssim_finetuned'] - best_ssim['avg_ssim_baseline']:+.4f}")
        if best_lpips.get('avg_lpips_baseline'):
            print(f"  LPIPS improvement: {best_lpips['avg_lpips_baseline'] - best_lpips['avg_lpips_finetuned']:+.4f} (lower is better)")


if __name__ == "__main__":
    main()

