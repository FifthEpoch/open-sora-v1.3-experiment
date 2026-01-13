#!/usr/bin/env python3
"""Generate LoRA heatmaps with all 12 configurations including rank16_lr2e4_100steps."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results directory
RESULTS_DIR = Path(__file__).parent / 'lora_experiment' / 'results'
OUTPUT_DIR = Path(__file__).parent / 'lora_experiment' / 'scripts' / 'visualization' / 'plots'

# All 12 configurations
CONFIGS = {
    'rank8_lr1e4_20steps': {'rank': 8, 'lr': 1e-4, 'steps': 20},
    'rank8_lr1e4_50steps': {'rank': 8, 'lr': 1e-4, 'steps': 50},
    'rank8_lr1e4_100steps': {'rank': 8, 'lr': 1e-4, 'steps': 100},
    'rank8_lr2e4_20steps': {'rank': 8, 'lr': 2e-4, 'steps': 20},
    'rank8_lr2e4_50steps': {'rank': 8, 'lr': 2e-4, 'steps': 50},
    'rank8_lr2e4_100steps': {'rank': 8, 'lr': 2e-4, 'steps': 100},
    'rank16_lr1e4_20steps': {'rank': 16, 'lr': 1e-4, 'steps': 20},
    'rank16_lr1e4_50steps': {'rank': 16, 'lr': 1e-4, 'steps': 50},
    'rank16_lr1e4_100steps': {'rank': 16, 'lr': 1e-4, 'steps': 100},
    'rank16_lr2e4_20steps': {'rank': 16, 'lr': 2e-4, 'steps': 20},
    'rank16_lr2e4_50steps': {'rank': 16, 'lr': 2e-4, 'steps': 50},
    'rank16_lr2e4_100steps': {'rank': 16, 'lr': 2e-4, 'steps': 100},
}

def load_results():
    """Load all metrics from results directories."""
    results = {}
    for dir_name, config in CONFIGS.items():
        metrics_file = RESULTS_DIR / dir_name / 'metrics_summary.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            results[dir_name] = {
                'config': config,
                'psnr': data.get('avg_psnr', 0),
                'ssim': data.get('avg_ssim', 0),
                'lpips': data.get('avg_lpips', 1),
            }
            print(f"Loaded {dir_name}: PSNR={results[dir_name]['psnr']:.2f}, SSIM={results[dir_name]['ssim']:.3f}, LPIPS={results[dir_name]['lpips']:.3f}")
        else:
            print(f"WARNING: Missing {metrics_file}")
    return results

def plot_heatmaps(results):
    """Generate heatmap visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    
    # Define grid structure
    ranks = [8, 16]
    lrs = ['1e-4', '2e-4']
    steps = [20, 50, 100]
    
    metrics = ['psnr', 'ssim', 'lpips']
    metric_labels = ['PSNR (↑)', 'SSIM (↑)', 'LPIPS (↓)']
    
    for rank_idx, rank in enumerate(ranks):
        for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[rank_idx, metric_idx]
            
            # Build data matrix (steps x lr)
            data = np.zeros((3, 2))
            for step_idx, step in enumerate(steps):
                for lr_idx, lr in enumerate(lrs):
                    # Format: rank8_lr1e4_20steps or rank16_lr2e4_100steps
                    lr_key = lr.replace('-', '')  # '1e-4' -> '1e4'
                    dir_name = f'rank{rank}_lr{lr_key}_{step}steps'
                    
                    if dir_name in results:
                        data[step_idx, lr_idx] = results[dir_name][metric]
                    else:
                        data[step_idx, lr_idx] = np.nan
            
            # Choose colormap based on metric
            if metric == 'lpips':
                cmap = 'RdYlGn_r'  # Reversed - lower is better
            else:
                cmap = 'RdYlGn'  # Higher is better
            
            im = ax.imshow(data, cmap=cmap, aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Labels
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['LR 1e-4', 'LR 2e-4'])
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['20 steps', '50 steps', '100 steps'])
            
            # Title for top row only
            if rank_idx == 0:
                ax.set_title(label, fontsize=12, fontweight='bold')
            
            # Y-label for left column only
            if metric_idx == 0:
                ax.set_ylabel(f'Rank {rank}', fontsize=12, fontweight='bold')
            
            # Annotate cells with values
            for i in range(3):
                for j in range(2):
                    if not np.isnan(data[i, j]):
                        text = f'{data[i, j]:.3f}' if metric != 'psnr' else f'{data[i, j]:.2f}'
                        ax.text(j, i, text, ha='center', va='center', 
                               fontsize=11, fontweight='bold', color='black')
    
    plt.suptitle('LoRA Hyperparameter Exploration\n(How Rank, LR, and Steps Affect Performance)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / '1_lora_heatmaps.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    # Also save PDF
    plt.savefig(OUTPUT_DIR / '1_lora_heatmaps.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / '1_lora_heatmaps.pdf'}")
    
    plt.close()

if __name__ == '__main__':
    print("Loading LoRA results...")
    results = load_results()
    print(f"\nLoaded {len(results)} configurations")
    
    print("\nGenerating heatmaps...")
    plot_heatmaps(results)
    print("\nDone!")

