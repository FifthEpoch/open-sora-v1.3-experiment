#!/usr/bin/env python3
"""
Visualize evaluation metrics from the naive fine-tuning experiment.

This script loads metrics.json and creates comparison plots for baseline vs fine-tuned models.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(json_path):
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_metrics(data):
    """Extract baseline and fine-tuned metrics from the data."""
    baseline_psnr = []
    baseline_ssim = []
    baseline_lpips = []
    
    finetuned_psnr = []
    finetuned_ssim = []
    finetuned_lpips = []
    
    for video in data:
        # Baseline metrics (try both 'baseline' and 'baseline_metrics' for compatibility)
        baseline_key = 'baseline' if 'baseline' in video else 'baseline_metrics'
        if baseline_key in video and video[baseline_key]:
            bm = video[baseline_key]
            if 'psnr' in bm:
                baseline_psnr.append(bm['psnr'])
            if 'ssim' in bm:
                baseline_ssim.append(bm['ssim'])
            if 'lpips' in bm:
                baseline_lpips.append(bm['lpips'])
        
        # Fine-tuned metrics (try both 'finetuned' and 'finetuned_metrics' for compatibility)
        finetuned_key = 'finetuned' if 'finetuned' in video else 'finetuned_metrics'
        if finetuned_key in video and video[finetuned_key]:
            fm = video[finetuned_key]
            if 'psnr' in fm:
                finetuned_psnr.append(fm['psnr'])
            if 'ssim' in fm:
                finetuned_ssim.append(fm['ssim'])
            if 'lpips' in fm:
                finetuned_lpips.append(fm['lpips'])
    
    return {
        'baseline': {
            'psnr': baseline_psnr,
            'ssim': baseline_ssim,
            'lpips': baseline_lpips
        },
        'finetuned': {
            'psnr': finetuned_psnr,
            'ssim': finetuned_ssim,
            'lpips': finetuned_lpips
        }
    }


def plot_comparison(metrics, output_dir):
    """Create comparison plots for all metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline = metrics['baseline']
    finetuned = metrics['finetuned']
    
    # Calculate averages
    avg_baseline_psnr = np.mean(baseline['psnr']) if baseline['psnr'] else 0
    avg_baseline_ssim = np.mean(baseline['ssim']) if baseline['ssim'] else 0
    avg_baseline_lpips = np.mean(baseline['lpips']) if baseline['lpips'] else 0
    
    avg_finetuned_psnr = np.mean(finetuned['psnr']) if finetuned['psnr'] else 0
    avg_finetuned_ssim = np.mean(finetuned['ssim']) if finetuned['ssim'] else 0
    avg_finetuned_lpips = np.mean(finetuned['lpips']) if finetuned['lpips'] else 0
    
    # Calculate standard errors
    se_baseline_psnr = np.std(baseline['psnr']) / np.sqrt(len(baseline['psnr'])) if baseline['psnr'] else 0
    se_baseline_ssim = np.std(baseline['ssim']) / np.sqrt(len(baseline['ssim'])) if baseline['ssim'] else 0
    se_baseline_lpips = np.std(baseline['lpips']) / np.sqrt(len(baseline['lpips'])) if baseline['lpips'] else 0
    
    se_finetuned_psnr = np.std(finetuned['psnr']) / np.sqrt(len(finetuned['psnr'])) if finetuned['psnr'] else 0
    se_finetuned_ssim = np.std(finetuned['ssim']) / np.sqrt(len(finetuned['ssim'])) if finetuned['ssim'] else 0
    se_finetuned_lpips = np.std(finetuned['lpips']) / np.sqrt(len(finetuned['lpips'])) if finetuned['lpips'] else 0
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nNumber of videos evaluated: {len(baseline['psnr'])}")
    print(f"\nBaseline Model:")
    print(f"  PSNR:  {avg_baseline_psnr:.4f} ± {se_baseline_psnr:.4f}")
    print(f"  SSIM:  {avg_baseline_ssim:.4f} ± {se_baseline_ssim:.4f}")
    print(f"  LPIPS: {avg_baseline_lpips:.4f} ± {se_baseline_lpips:.4f}")
    print(f"\nFine-tuned Model:")
    print(f"  PSNR:  {avg_finetuned_psnr:.4f} ± {se_finetuned_psnr:.4f}")
    print(f"  SSIM:  {avg_finetuned_ssim:.4f} ± {se_finetuned_ssim:.4f}")
    print(f"  LPIPS: {avg_finetuned_lpips:.4f} ± {se_finetuned_lpips:.4f}")
    
    # Calculate improvements
    if avg_baseline_psnr > 0:
        psnr_improvement = ((avg_finetuned_psnr - avg_baseline_psnr) / avg_baseline_psnr) * 100
        print(f"\nImprovement:")
        print(f"  PSNR:  {psnr_improvement:+.2f}%")
    if avg_baseline_ssim > 0:
        ssim_improvement = ((avg_finetuned_ssim - avg_baseline_ssim) / avg_baseline_ssim) * 100
        print(f"  SSIM:  {ssim_improvement:+.2f}%")
    if avg_baseline_lpips > 0:
        lpips_improvement = ((avg_baseline_lpips - avg_finetuned_lpips) / avg_baseline_lpips) * 100
        print(f"  LPIPS: {lpips_improvement:+.2f}% (lower is better)")
    print("="*60 + "\n")
    
    # Create bar plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Bar width (thinner bars)
    bar_width = 0.5
    x_positions = [0, 1]
    
    # PSNR
    bars1 = ax[0].bar(x_positions,
                       [avg_baseline_psnr, avg_finetuned_psnr],
                       yerr=[se_baseline_psnr, se_finetuned_psnr],
                       capsize=5,
                       width=bar_width,
                       color=['#3498db', '#e74c3c'],
                       alpha=0.8)
    ax[0].set_ylabel('PSNR (dB)', fontsize=12)
    ax[0].set_title('PSNR', fontsize=14, fontweight='bold')
    ax[0].grid(axis='y', alpha=0.3, linestyle='--')
    ax[0].set_ylim(bottom=0)
    ax[0].set_xticks(x_positions)
    ax[0].set_xticklabels(['Baseline', 'Fine-tuned'])
    
    # Add value labels on bars (offset to the right to avoid overlap with error bars)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        error = [se_baseline_psnr, se_finetuned_psnr][i]
        ax[0].text(bar.get_x() + bar.get_width() + 0.05, height,
                  f'{height:.2f}',
                  ha='left', va='center', fontsize=10, fontweight='bold')
    
    # SSIM
    bars2 = ax[1].bar(x_positions,
                       [avg_baseline_ssim, avg_finetuned_ssim],
                       yerr=[se_baseline_ssim, se_finetuned_ssim],
                       capsize=5,
                       width=bar_width,
                       color=['#3498db', '#e74c3c'],
                       alpha=0.8)
    ax[1].set_ylabel('SSIM', fontsize=12)
    ax[1].set_title('SSIM', fontsize=14, fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3, linestyle='--')
    ax[1].set_ylim(0, 1)
    ax[1].set_xticks(x_positions)
    ax[1].set_xticklabels(['Baseline', 'Fine-tuned'])
    
    # Add value labels on bars (offset to the right to avoid overlap with error bars)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        error = [se_baseline_ssim, se_finetuned_ssim][i]
        ax[1].text(bar.get_x() + bar.get_width() + 0.05, height,
                  f'{height:.3f}',
                  ha='left', va='center', fontsize=10, fontweight='bold')
    
    # LPIPS
    bars3 = ax[2].bar(x_positions,
                       [avg_baseline_lpips, avg_finetuned_lpips],
                       yerr=[se_baseline_lpips, se_finetuned_lpips],
                       capsize=5,
                       width=bar_width,
                       color=['#3498db', '#e74c3c'],
                       alpha=0.8)
    ax[2].set_ylabel('LPIPS', fontsize=12)
    ax[2].set_title('LPIPS', fontsize=14, fontweight='bold')
    ax[2].grid(axis='y', alpha=0.3, linestyle='--')
    ax[2].set_ylim(bottom=0)
    ax[2].set_xticks(x_positions)
    ax[2].set_xticklabels(['Baseline', 'Fine-tuned'])
    
    # Add value labels on bars (offset to the right to avoid overlap with error bars)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        error = [se_baseline_lpips, se_finetuned_lpips][i]
        ax[2].text(bar.get_x() + bar.get_width() + 0.05, height,
                  f'{height:.3f}',
                  ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_dir / 'metrics_comparison.png'}")
    
    # Create individual metric plots with distributions
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR distribution
    ax[0].hist([baseline['psnr'], finetuned['psnr']], bins=20, alpha=0.6, 
               label=['Baseline', 'Fine-tuned'], color=['#3498db', '#e74c3c'])
    ax[0].set_xlabel('PSNR (dB)', fontsize=12)
    ax[0].set_ylabel('Frequency', fontsize=12)
    ax[0].set_title('PSNR Distribution', fontsize=12, fontweight='bold')
    ax[0].legend()
    ax[0].grid(alpha=0.3, linestyle='--')
    
    # SSIM distribution
    ax[1].hist([baseline['ssim'], finetuned['ssim']], bins=20, alpha=0.6, 
               label=['Baseline', 'Fine-tuned'], color=['#3498db', '#e74c3c'])
    ax[1].set_xlabel('SSIM', fontsize=12)
    ax[1].set_ylabel('Frequency', fontsize=12)
    ax[1].set_title('SSIM Distribution', fontsize=12, fontweight='bold')
    ax[1].legend()
    ax[1].grid(alpha=0.3, linestyle='--')
    
    # LPIPS distribution
    ax[2].hist([baseline['lpips'], finetuned['lpips']], bins=20, alpha=0.6, 
               label=['Baseline', 'Fine-tuned'], color=['#3498db', '#e74c3c'])
    ax[2].set_xlabel('LPIPS', fontsize=12)
    ax[2].set_ylabel('Frequency', fontsize=12)
    ax[2].set_title('LPIPS Distribution', fontsize=12, fontweight='bold')
    ax[2].legend()
    ax[2].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_dir / 'metrics_distributions.png'}")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation metrics")
    parser.add_argument("--metrics-json", type=str, required=True, 
                       help="Path to metrics.json file")
    parser.add_argument("--output-dir", type=str, default="visualization_outputs",
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Load metrics
    print(f"Loading metrics from: {args.metrics_json}")
    data = load_metrics(args.metrics_json)
    print(f"Loaded {len(data)} video evaluations")
    
    # Extract metrics
    metrics = extract_metrics(data)
    
    # Check if we have data
    if not metrics['baseline']['psnr']:
        print("ERROR: No valid metrics found in the data!")
        return
    
    # Create plots
    plot_comparison(metrics, args.output_dir)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

