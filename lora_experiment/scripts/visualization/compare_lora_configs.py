#!/usr/bin/env python3
"""
Compare LoRA experiment results across different configurations.
Generates plots comparing:
- Effect of steps (20, 50, 100)
- Effect of rank (8, 16)
- Effect of learning rate (1e-4, 2e-4)
- Time vs quality trade-offs
"""

import json
import os
from pathlib import Path
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")


def load_all_results(results_dir: Path) -> dict:
    """Load all metrics_summary.json files from subdirectories."""
    results = {}
    
    # Define expected configurations and their properties
    # Consistent naming format: rank{R}_lr{LR}_{STEPS}steps
    configs = {
        # Rank 8, LR 1e-4
        'rank8_lr1e4_20steps': {'rank': 8, 'lr': 1e-4, 'steps': 20, 'label': 'R8 LR1e-4 20s'},
        'rank8_lr1e4_50steps': {'rank': 8, 'lr': 1e-4, 'steps': 50, 'label': 'R8 LR1e-4 50s'},
        'rank8_lr1e4_100steps': {'rank': 8, 'lr': 1e-4, 'steps': 100, 'label': 'R8 LR1e-4 100s'},
        # Rank 8, LR 2e-4
        'rank8_lr2e4_20steps': {'rank': 8, 'lr': 2e-4, 'steps': 20, 'label': 'R8 LR2e-4 20s'},
        'rank8_lr2e4_50steps': {'rank': 8, 'lr': 2e-4, 'steps': 50, 'label': 'R8 LR2e-4 50s'},
        'rank8_lr2e4_100steps': {'rank': 8, 'lr': 2e-4, 'steps': 100, 'label': 'R8 LR2e-4 100s'},
        # Rank 16, LR 1e-4
        'rank16_lr1e4_20steps': {'rank': 16, 'lr': 1e-4, 'steps': 20, 'label': 'R16 LR1e-4 20s'},
        'rank16_lr1e4_50steps': {'rank': 16, 'lr': 1e-4, 'steps': 50, 'label': 'R16 LR1e-4 50s'},
        'rank16_lr1e4_100steps': {'rank': 16, 'lr': 1e-4, 'steps': 100, 'label': 'R16 LR1e-4 100s'},
        # Rank 16, LR 2e-4
        'rank16_lr2e4_20steps': {'rank': 16, 'lr': 2e-4, 'steps': 20, 'label': 'R16 LR2e-4 20s'},
        'rank16_lr2e4_50steps': {'rank': 16, 'lr': 2e-4, 'steps': 50, 'label': 'R16 LR2e-4 50s'},
    }
    
    for dir_name, config in configs.items():
        metrics_file = results_dir / dir_name / 'metrics_summary.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            # Normalize field names (some have _sec suffix, some don't)
            train_time = data.get('avg_train_time_sec') or data.get('avg_train_time', 0)
            infer_time = data.get('avg_infer_time_sec') or data.get('avg_infer_time', 0)
            
            results[dir_name] = {
                'config': config,
                'psnr': data.get('avg_psnr', 0),
                'ssim': data.get('avg_ssim', 0),
                'lpips': data.get('avg_lpips', 1),
                'train_time': train_time,
                'infer_time': infer_time,
                'total_time': train_time + infer_time,
            }
            print(f"Loaded {dir_name}: PSNR={results[dir_name]['psnr']:.2f}, "
                  f"SSIM={results[dir_name]['ssim']:.4f}, LPIPS={results[dir_name]['lpips']:.4f}")
    
    return results


def plot_metrics_comparison(results: dict, output_dir: Path):
    """Bar chart comparing all metrics across configurations."""
    if not HAS_MATPLOTLIB:
        return
    
    configs = list(results.keys())
    labels = [results[c]['config']['label'] for c in configs]
    
    psnr_vals = [results[c]['psnr'] for c in configs]
    ssim_vals = [results[c]['ssim'] for c in configs]
    lpips_vals = [results[c]['lpips'] for c in configs]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    
    # PSNR (higher is better)
    bars = axes[0].bar(range(len(configs)), psnr_vals, color=colors)
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR ↑ (Higher is Better)')
    axes[0].axhline(y=max(psnr_vals), color='green', linestyle='--', alpha=0.5, label='Best')
    for i, v in enumerate(psnr_vals):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    # SSIM (higher is better)
    bars = axes[1].bar(range(len(configs)), ssim_vals, color=colors)
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM ↑ (Higher is Better)')
    axes[1].axhline(y=max(ssim_vals), color='green', linestyle='--', alpha=0.5)
    for i, v in enumerate(ssim_vals):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # LPIPS (lower is better)
    bars = axes[2].bar(range(len(configs)), lpips_vals, color=colors)
    axes[2].set_xticks(range(len(configs)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].set_ylabel('LPIPS')
    axes[2].set_title('LPIPS ↓ (Lower is Better)')
    axes[2].axhline(y=min(lpips_vals), color='green', linestyle='--', alpha=0.5)
    for i, v in enumerate(lpips_vals):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '1_metrics_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: 1_metrics_comparison.png/pdf")


def plot_effect_of_steps(results: dict, output_dir: Path):
    """Line plot showing how metrics change with number of steps."""
    if not HAS_MATPLOTLIB:
        return
    
    # Filter to rank8_lr1e4 configs only
    step_configs = ['rank8_lr1e4_20steps', 'rank8_lr1e4_50steps', 'rank8_lr1e4_100steps']
    available = [c for c in step_configs if c in results]
    
    if len(available) < 2:
        print("Not enough step configs for comparison")
        return
    
    steps = [results[c]['config']['steps'] for c in available]
    psnr = [results[c]['psnr'] for c in available]
    ssim = [results[c]['ssim'] for c in available]
    lpips = [results[c]['lpips'] for c in available]
    train_time = [results[c]['train_time'] for c in available]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR vs Steps
    axes[0, 0].plot(steps, psnr, 'bo-', linewidth=2, markersize=10)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_title('PSNR vs Training Steps (Rank=8, LR=1e-4)')
    axes[0, 0].grid(True, alpha=0.3)
    for i, (s, p) in enumerate(zip(steps, psnr)):
        axes[0, 0].annotate(f'{p:.2f}', (s, p), textcoords="offset points", xytext=(0, 10), ha='center')
    
    # SSIM vs Steps
    axes[0, 1].plot(steps, ssim, 'go-', linewidth=2, markersize=10)
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_title('SSIM vs Training Steps (Rank=8, LR=1e-4)')
    axes[0, 1].grid(True, alpha=0.3)
    for i, (s, ss) in enumerate(zip(steps, ssim)):
        axes[0, 1].annotate(f'{ss:.3f}', (s, ss), textcoords="offset points", xytext=(0, 10), ha='center')
    
    # LPIPS vs Steps
    axes[1, 0].plot(steps, lpips, 'ro-', linewidth=2, markersize=10)
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('LPIPS')
    axes[1, 0].set_title('LPIPS vs Training Steps (Rank=8, LR=1e-4)')
    axes[1, 0].grid(True, alpha=0.3)
    for i, (s, l) in enumerate(zip(steps, lpips)):
        axes[1, 0].annotate(f'{l:.3f}', (s, l), textcoords="offset points", xytext=(0, -15), ha='center')
    
    # Training Time vs Steps
    axes[1, 1].plot(steps, train_time, 'mo-', linewidth=2, markersize=10)
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('Training Time (sec)')
    axes[1, 1].set_title('Training Time vs Steps (Rank=8, LR=1e-4)')
    axes[1, 1].grid(True, alpha=0.3)
    for i, (s, t) in enumerate(zip(steps, train_time)):
        axes[1, 1].annotate(f'{t:.0f}s', (s, t), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_effect_of_steps.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '2_effect_of_steps.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: 2_effect_of_steps.png/pdf")


def plot_effect_of_rank_and_lr(results: dict, output_dir: Path):
    """Heatmap/grouped bars showing effect of rank and LR at 20 steps."""
    if not HAS_MATPLOTLIB:
        return
    
    # Filter to 20-step configs only
    configs_20s = {
        'rank8_lr1e4_20steps': (8, 1e-4),
        'rank16_lr1e4': (16, 1e-4),
        'rank8_lr2e4': (8, 2e-4),
        'rank16_lr2e4': (16, 2e-4),
    }
    
    available = {k: v for k, v in configs_20s.items() if k in results}
    if len(available) < 2:
        print("Not enough rank/LR configs for comparison")
        return
    
    # Create grouped bar chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    ranks = [8, 16]
    lrs = ['1e-4', '2e-4']
    lr_map = {1e-4: '1e-4', 2e-4: '2e-4'}
    
    x = np.arange(len(ranks))
    width = 0.35
    
    metrics = ['psnr', 'ssim', 'lpips', 'train_time']
    titles = ['PSNR ↑', 'SSIM ↑', 'LPIPS ↓', 'Training Time (s)']
    
    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        lr1_vals = []
        lr2_vals = []
        
        for rank in ranks:
            # Find config for LR=1e-4
            lr1_config = None
            lr2_config = None
            for cname, (r, lr) in available.items():
                if r == rank and lr == 1e-4:
                    lr1_config = cname
                elif r == rank and lr == 2e-4:
                    lr2_config = cname
            
            lr1_vals.append(results[lr1_config][metric] if lr1_config else 0)
            lr2_vals.append(results[lr2_config][metric] if lr2_config else 0)
        
        bars1 = axes[ax_idx].bar(x - width/2, lr1_vals, width, label='LR=1e-4', color='steelblue')
        bars2 = axes[ax_idx].bar(x + width/2, lr2_vals, width, label='LR=2e-4', color='coral')
        
        axes[ax_idx].set_xlabel('LoRA Rank')
        axes[ax_idx].set_ylabel(metric.upper() if metric != 'train_time' else 'Seconds')
        axes[ax_idx].set_title(title)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([f'Rank {r}' for r in ranks])
        axes[ax_idx].legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            fmt = '.2f' if metric in ['psnr', 'lpips'] else '.3f' if metric == 'ssim' else '.0f'
            axes[ax_idx].annotate(f'{height:{fmt}}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            fmt = '.2f' if metric in ['psnr', 'lpips'] else '.3f' if metric == 'ssim' else '.0f'
            axes[ax_idx].annotate(f'{height:{fmt}}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_effect_of_rank_lr.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '3_effect_of_rank_lr.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: 3_effect_of_rank_lr.png/pdf")


def plot_time_vs_quality(results: dict, output_dir: Path):
    """Scatter plot: training time vs quality metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    configs = list(results.keys())
    labels = [results[c]['config']['label'] for c in configs]
    
    train_times = [results[c]['train_time'] for c in configs]
    psnr_vals = [results[c]['psnr'] for c in configs]
    lpips_vals = [results[c]['lpips'] for c in configs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Assign colors based on rank
    colors = ['blue' if results[c]['config']['rank'] == 8 else 'red' for c in configs]
    # Assign markers based on LR
    markers = ['o' if results[c]['config']['lr'] == 1e-4 else 's' for c in configs]
    
    # Time vs PSNR
    for i, c in enumerate(configs):
        axes[0].scatter(train_times[i], psnr_vals[i], c=colors[i], marker=markers[i], s=150, alpha=0.7)
        axes[0].annotate(labels[i], (train_times[i], psnr_vals[i]), 
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    axes[0].set_xlabel('Training Time per Video (seconds)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Training Time vs PSNR')
    axes[0].grid(True, alpha=0.3)
    
    # Time vs LPIPS  
    for i, c in enumerate(configs):
        axes[1].scatter(train_times[i], lpips_vals[i], c=colors[i], marker=markers[i], s=150, alpha=0.7)
        axes[1].annotate(labels[i], (train_times[i], lpips_vals[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    axes[1].set_xlabel('Training Time per Video (seconds)')
    axes[1].set_ylabel('LPIPS (lower is better)')
    axes[1].set_title('Training Time vs LPIPS')
    axes[1].grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Rank 8'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Rank 16'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='LR 1e-4'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='LR 2e-4'),
    ]
    axes[0].legend(handles=legend_elements, loc='lower right')
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_time_vs_quality.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '4_time_vs_quality.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: 4_time_vs_quality.png/pdf")


def plot_summary_table(results: dict, output_dir: Path):
    """Create a summary table as an image."""
    if not HAS_MATPLOTLIB:
        return
    
    configs = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    # Create table data
    headers = ['Configuration', 'Rank', 'LR', 'Steps', 'PSNR↑', 'SSIM↑', 'LPIPS↓', 'Train(s)', 'Total(s)']
    
    table_data = []
    for c in configs:
        r = results[c]
        row = [
            r['config']['label'],
            str(r['config']['rank']),
            f"{r['config']['lr']:.0e}",
            str(r['config']['steps']),
            f"{r['psnr']:.2f}",
            f"{r['ssim']:.4f}",
            f"{r['lpips']:.4f}",
            f"{r['train_time']:.1f}",
            f"{r['total_time']:.1f}",
        ]
        table_data.append(row)
    
    # Sort by LPIPS (lower is better)
    table_data.sort(key=lambda x: float(x[6]))
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best values
    psnr_col = 4
    ssim_col = 5
    lpips_col = 6
    
    best_psnr_row = max(range(len(table_data)), key=lambda i: float(table_data[i][psnr_col])) + 1
    best_ssim_row = max(range(len(table_data)), key=lambda i: float(table_data[i][ssim_col])) + 1
    best_lpips_row = min(range(len(table_data)), key=lambda i: float(table_data[i][lpips_col])) + 1
    
    table[(best_psnr_row, psnr_col)].set_facecolor('#C6EFCE')
    table[(best_ssim_row, ssim_col)].set_facecolor('#C6EFCE')
    table[(best_lpips_row, lpips_col)].set_facecolor('#C6EFCE')
    
    plt.title('LoRA Experiment Results Summary (Sorted by LPIPS)', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / '5_summary_table.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '5_summary_table.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: 5_summary_table.png/pdf")


def plot_radar_chart(results: dict, output_dir: Path):
    """Radar chart comparing all configurations."""
    if not HAS_MATPLOTLIB:
        return
    
    configs = list(results.keys())
    
    # Normalize metrics to 0-1 scale (for radar chart)
    psnr_vals = [results[c]['psnr'] for c in configs]
    ssim_vals = [results[c]['ssim'] for c in configs]
    lpips_vals = [results[c]['lpips'] for c in configs]
    time_vals = [results[c]['train_time'] for c in configs]
    
    # Normalize (higher is better for all after transformation)
    psnr_norm = [(v - min(psnr_vals)) / (max(psnr_vals) - min(psnr_vals) + 1e-8) for v in psnr_vals]
    ssim_norm = [(v - min(ssim_vals)) / (max(ssim_vals) - min(ssim_vals) + 1e-8) for v in ssim_vals]
    lpips_norm = [1 - (v - min(lpips_vals)) / (max(lpips_vals) - min(lpips_vals) + 1e-8) for v in lpips_vals]  # Invert
    time_norm = [1 - (v - min(time_vals)) / (max(time_vals) - min(time_vals) + 1e-8) for v in time_vals]  # Invert (faster is better)
    
    categories = ['PSNR', 'SSIM', 'LPIPS\n(inverted)', 'Speed\n(inverted)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    
    for i, c in enumerate(configs):
        values = [psnr_norm[i], ssim_norm[i], lpips_norm[i], time_norm[i]]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=results[c]['config']['label'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('LoRA Configurations Comparison\n(Higher is Better for All)', fontsize=14, fontweight='bold')
    
    plt.savefig(output_dir / '6_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '6_radar_chart.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: 6_radar_chart.png/pdf")


def main():
    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent.parent / 'results'
    output_dir = script_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    print(f"Saving plots to: {output_dir}")
    print("=" * 60)
    
    # Load all results
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print("=" * 60)
    print(f"Found {len(results)} configurations")
    print("=" * 60)
    
    # Generate all plots
    plot_metrics_comparison(results, output_dir)
    plot_effect_of_steps(results, output_dir)
    plot_effect_of_rank_and_lr(results, output_dir)
    plot_time_vs_quality(results, output_dir)
    plot_summary_table(results, output_dir)
    plot_radar_chart(results, output_dir)
    
    print("=" * 60)
    print("All plots generated!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

