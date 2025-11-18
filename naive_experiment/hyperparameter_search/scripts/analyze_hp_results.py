#!/usr/bin/env python3
"""
Hyperparameter Search Results Analysis and Visualization

This script analyzes and visualizes the results from hyperparameter searches,
comparing the impact of different learning rates, conditioning frame counts,
and fine-tuning steps on model performance.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_file: Path) -> List[Dict]:
    """Load summary JSON from hyperparameter search."""
    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_lr_search(summary: List[Dict], output_dir: Path):
    """Plot learning rate search results."""
    # Extract data
    lrs = [s['lr'] for s in summary]
    psnr_base = [s['baseline_psnr_mean'] for s in summary]
    psnr_ft = [s['finetuned_psnr_mean'] for s in summary]
    psnr_imp = [s.get('psnr_improvement_pct', 0) for s in summary]
    
    ssim_base = [s['baseline_ssim_mean'] for s in summary]
    ssim_ft = [s['finetuned_ssim_mean'] for s in summary]
    ssim_imp = [s.get('ssim_improvement_pct', 0) for s in summary]
    
    lpips_base = [s['baseline_lpips_mean'] for s in summary]
    lpips_ft = [s['finetuned_lpips_mean'] for s in summary]
    lpips_imp = [s.get('lpips_improvement_pct', 0) for s in summary]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Learning Rate Hyperparameter Search', fontsize=16, fontweight='bold')
    
    # Log scale for x-axis
    lr_labels = [f'{lr:.0e}' for lr in lrs]
    x_pos = np.arange(len(lrs))
    
    # Row 1: Absolute values
    # PSNR
    axes[0, 0].plot(x_pos, psnr_base, 'o-', label='Baseline', linewidth=2, markersize=8, color='#1f77b4')
    axes[0, 0].plot(x_pos, psnr_ft, 's-', label='Fine-tuned', linewidth=2, markersize=8, color='#ff7f0e')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(lr_labels)
    axes[0, 0].set_xlabel('Learning Rate', fontsize=11)
    axes[0, 0].set_ylabel('PSNR (dB)', fontsize=11)
    axes[0, 0].set_title('PSNR vs Learning Rate', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SSIM
    axes[0, 1].plot(x_pos, ssim_base, 'o-', label='Baseline', linewidth=2, markersize=8, color='#1f77b4')
    axes[0, 1].plot(x_pos, ssim_ft, 's-', label='Fine-tuned', linewidth=2, markersize=8, color='#ff7f0e')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(lr_labels)
    axes[0, 1].set_xlabel('Learning Rate', fontsize=11)
    axes[0, 1].set_ylabel('SSIM', fontsize=11)
    axes[0, 1].set_title('SSIM vs Learning Rate', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # LPIPS
    axes[0, 2].plot(x_pos, lpips_base, 'o-', label='Baseline', linewidth=2, markersize=8, color='#1f77b4')
    axes[0, 2].plot(x_pos, lpips_ft, 's-', label='Fine-tuned', linewidth=2, markersize=8, color='#ff7f0e')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(lr_labels)
    axes[0, 2].set_xlabel('Learning Rate', fontsize=11)
    axes[0, 2].set_ylabel('LPIPS (lower better)', fontsize=11)
    axes[0, 2].set_title('LPIPS vs Learning Rate', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Improvements
    axes[1, 0].bar(x_pos, psnr_imp, color='#2ca02c', alpha=0.7, width=0.6)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(lr_labels)
    axes[1, 0].set_xlabel('Learning Rate', fontsize=11)
    axes[1, 0].set_ylabel('Improvement (%)', fontsize=11)
    axes[1, 0].set_title('PSNR Improvement', fontsize=12, fontweight='bold')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(x_pos, ssim_imp, color='#2ca02c', alpha=0.7, width=0.6)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(lr_labels)
    axes[1, 1].set_xlabel('Learning Rate', fontsize=11)
    axes[1, 1].set_ylabel('Improvement (%)', fontsize=11)
    axes[1, 1].set_title('SSIM Improvement', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].bar(x_pos, lpips_imp, color='#2ca02c', alpha=0.7, width=0.6)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(lr_labels)
    axes[1, 2].set_xlabel('Learning Rate', fontsize=11)
    axes[1, 2].set_ylabel('Improvement (%)', fontsize=11)
    axes[1, 2].set_title('LPIPS Improvement', fontsize=12, fontweight='bold')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'lr_search_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.close()


def plot_condition_frames_search(summary: List[Dict], output_dir: Path):
    """Plot conditioning frame count search results."""
    # Extract data
    cfs = [s['condition_frames'] for s in summary]
    psnr_imp = [s.get('psnr_improvement_pct', 0) for s in summary]
    ssim_imp = [s.get('ssim_improvement_pct', 0) for s in summary]
    lpips_imp = [s.get('lpips_improvement_pct', 0) for s in summary]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Conditioning Frame Count Hyperparameter Search', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(cfs))
    cf_labels = [str(cf) for cf in cfs]
    
    # PSNR Improvement
    axes[0].plot(x_pos, psnr_imp, 'o-', linewidth=2, markersize=10, color='#1f77b4')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(cf_labels)
    axes[0].set_xlabel('Conditioning Frames', fontsize=12)
    axes[0].set_ylabel('PSNR Improvement (%)', fontsize=12)
    axes[0].set_title('PSNR Improvement', fontsize=13, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].grid(True, alpha=0.3)
    
    # SSIM Improvement
    axes[1].plot(x_pos, ssim_imp, 'o-', linewidth=2, markersize=10, color='#ff7f0e')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(cf_labels)
    axes[1].set_xlabel('Conditioning Frames', fontsize=12)
    axes[1].set_ylabel('SSIM Improvement (%)', fontsize=12)
    axes[1].set_title('SSIM Improvement', fontsize=13, fontweight='bold')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3)
    
    # LPIPS Improvement
    axes[2].plot(x_pos, lpips_imp, 'o-', linewidth=2, markersize=10, color='#2ca02c')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(cf_labels)
    axes[2].set_xlabel('Conditioning Frames', fontsize=12)
    axes[2].set_ylabel('LPIPS Improvement (%)', fontsize=12)
    axes[2].set_title('LPIPS Improvement', fontsize=13, fontweight='bold')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'condition_frames_search_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.close()


def plot_steps_search(summary: List[Dict], output_dir: Path):
    """Plot fine-tuning steps search results."""
    # Extract data
    steps = [s['finetune_steps'] for s in summary]
    psnr_imp = [s.get('psnr_improvement_pct', 0) for s in summary]
    ssim_imp = [s.get('ssim_improvement_pct', 0) for s in summary]
    lpips_imp = [s.get('lpips_improvement_pct', 0) for s in summary]
    times = [s.get('avg_finetune_time_sec', 0) / 60 for s in summary]  # Convert to minutes
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fine-tuning Steps Hyperparameter Search', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(steps))
    step_labels = [str(s) for s in steps]
    
    # PSNR Improvement
    axes[0, 0].plot(x_pos, psnr_imp, 'o-', linewidth=2, markersize=10, color='#1f77b4')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(step_labels)
    axes[0, 0].set_xlabel('Fine-tuning Steps', fontsize=12)
    axes[0, 0].set_ylabel('PSNR Improvement (%)', fontsize=12)
    axes[0, 0].set_title('PSNR Improvement', fontsize=13, fontweight='bold')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # SSIM Improvement
    axes[0, 1].plot(x_pos, ssim_imp, 'o-', linewidth=2, markersize=10, color='#ff7f0e')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(step_labels)
    axes[0, 1].set_xlabel('Fine-tuning Steps', fontsize=12)
    axes[0, 1].set_ylabel('SSIM Improvement (%)', fontsize=12)
    axes[0, 1].set_title('SSIM Improvement', fontsize=13, fontweight='bold')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # LPIPS Improvement
    axes[1, 0].plot(x_pos, lpips_imp, 'o-', linewidth=2, markersize=10, color='#2ca02c')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(step_labels)
    axes[1, 0].set_xlabel('Fine-tuning Steps', fontsize=12)
    axes[1, 0].set_ylabel('LPIPS Improvement (%)', fontsize=12)
    axes[1, 0].set_title('LPIPS Improvement', fontsize=13, fontweight='bold')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training Time
    axes[1, 1].bar(x_pos, times, color='#9467bd', alpha=0.7, width=0.6)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(step_labels)
    axes[1, 1].set_xlabel('Fine-tuning Steps', fontsize=12)
    axes[1, 1].set_ylabel('Avg Training Time (minutes)', fontsize=12)
    axes[1, 1].set_title('Computational Cost', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'steps_search_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.close()


def print_summary_stats(summary: List[Dict], search_type: str):
    """Print summary statistics."""
    print("\n" + "="*60)
    print(f"Summary Statistics - {search_type.upper()} Search")
    print("="*60)
    
    for config in summary:
        print(f"\nConfiguration: {config['config_name']}")
        print(f"  PSNR: {config['baseline_psnr_mean']:.4f} → {config['finetuned_psnr_mean']:.4f} ({config.get('psnr_improvement_pct', 0):+.2f}%)")
        print(f"  SSIM: {config['baseline_ssim_mean']:.4f} → {config['finetuned_ssim_mean']:.4f} ({config.get('ssim_improvement_pct', 0):+.2f}%)")
        print(f"  LPIPS: {config['baseline_lpips_mean']:.4f} → {config['finetuned_lpips_mean']:.4f} ({config.get('lpips_improvement_pct', 0):+.2f}%)")
        
        if 'avg_finetune_time_sec' in config:
            print(f"  Avg fine-tuning time: {config['avg_finetune_time_sec']:.1f}s")
    
    # Find best configuration
    best_psnr = max(summary, key=lambda x: x.get('psnr_improvement_pct', -999))
    best_ssim = max(summary, key=lambda x: x.get('ssim_improvement_pct', -999))
    best_lpips = max(summary, key=lambda x: x.get('lpips_improvement_pct', -999))
    
    print("\n" + "-"*60)
    print("Best Configurations:")
    print(f"  Best PSNR improvement: {best_psnr['config_name']} ({best_psnr.get('psnr_improvement_pct', 0):+.2f}%)")
    print(f"  Best SSIM improvement: {best_ssim['config_name']} ({best_ssim.get('ssim_improvement_pct', 0):+.2f}%)")
    print(f"  Best LPIPS improvement: {best_lpips['config_name']} ({best_lpips.get('lpips_improvement_pct', 0):+.2f}%)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument(
        "--search-type",
        type=str,
        required=True,
        choices=['lr', 'condition_frames', 'steps'],
        help="Type of hyperparameter search"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots (default: same as results-dir)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summary
    search_map = {
        'lr': 'lr_search',
        'condition_frames': 'condition_frames_search',
        'steps': 'steps_search'
    }
    
    search_name = search_map[args.search_type]
    summary_file = results_dir / search_name / "summary.json"
    
    if not summary_file.exists():
        print(f"ERROR: Summary file not found: {summary_file}")
        return
    
    summary = load_summary(summary_file)
    
    # Print statistics
    print_summary_stats(summary, args.search_type)
    
    # Generate plots
    print(f"Generating plots...")
    
    if args.search_type == 'lr':
        plot_lr_search(summary, output_dir)
    elif args.search_type == 'condition_frames':
        plot_condition_frames_search(summary, output_dir)
    elif args.search_type == 'steps':
        plot_steps_search(summary, output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

