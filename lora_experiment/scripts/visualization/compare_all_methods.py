#!/usr/bin/env python3
"""
Comprehensive comparison of LoRA experiments and Full Fine-tuning.
Generates publication-quality plots comparing:
- All LoRA configurations (rank, LR, steps)
- LoRA vs Full Fine-tuning efficiency
- Time-quality trade-offs
"""

import json
import os
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# Color schemes
COLORS = {
    'lora_r8': '#2ecc71',      # Green for rank 8
    'lora_r16': '#3498db',     # Blue for rank 16
    'full_ft': '#e74c3c',      # Red for full fine-tuning
    'lr1e4': '#9b59b6',        # Purple for LR 1e-4
    'lr2e4': '#f39c12',        # Orange for LR 2e-4
}


def load_lora_results(results_dir: Path) -> dict:
    """Load all LoRA metrics_summary.json files."""
    configs = {
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
    
    results = {}
    for dir_name, config in configs.items():
        metrics_file = results_dir / dir_name / 'metrics_summary.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            train_time = data.get('avg_train_time_sec') or data.get('avg_train_time', 0)
            
            results[dir_name] = {
                'config': config,
                'psnr': data.get('avg_psnr', 0),
                'ssim': data.get('avg_ssim', 0),
                'lpips': data.get('avg_lpips', 1),
                'train_time': train_time,
                'method': 'LoRA',
            }
            print(f"Loaded LoRA {dir_name}: PSNR={results[dir_name]['psnr']:.2f}")
    
    return results


def load_full_ft_results(results_dir: Path) -> dict:
    """Load full fine-tuning results from naive experiment."""
    configs = {
        '50steps_5e5': {'steps': 50, 'lr': 5e-5},
        '100steps_5e5': {'steps': 100, 'lr': 5e-5},
        '50steps_1e5': {'steps': 50, 'lr': 1e-5},
        '15steps_1e4': {'steps': 15, 'lr': 1e-4},
    }
    
    results = {}
    for dir_name, config in configs.items():
        metrics_file = results_dir / dir_name / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                # Compute averages from per-video data
                psnr_vals = [v.get('finetuned', {}).get('psnr', 0) for v in data if 'finetuned' in v]
                ssim_vals = [v.get('finetuned', {}).get('ssim', 0) for v in data if 'finetuned' in v]
                lpips_vals = [v.get('finetuned', {}).get('lpips', 1) for v in data if 'finetuned' in v]
                train_times = [v.get('finetune_time_sec', 0) for v in data]
                
                if psnr_vals:
                    results[f'full_ft_{dir_name}'] = {
                        'config': config,
                        'psnr': np.mean(psnr_vals),
                        'ssim': np.mean(ssim_vals),
                        'lpips': np.mean(lpips_vals),
                        'train_time': np.mean(train_times),
                        'method': 'Full FT',
                        'num_videos': len(psnr_vals),
                    }
                    print(f"Loaded Full FT {dir_name}: PSNR={results[f'full_ft_{dir_name}']['psnr']:.2f}, n={len(psnr_vals)}")
    
    return results


def plot_lora_heatmaps(lora_results: dict, output_dir: Path):
    """Create heatmaps showing how metrics vary with rank, LR, and steps."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Organize data by (rank, lr) for each step count
    step_counts = [20, 50, 100]
    metrics = ['psnr', 'ssim', 'lpips']
    metric_titles = ['PSNR (↑)', 'SSIM (↑)', 'LPIPS (↓)']
    
    for col, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Rank 8 row
        r8_data = []
        for steps in step_counts:
            lr1e4 = lora_results.get(f'rank8_lr1e4_{steps}steps', {}).get(metric, np.nan)
            lr2e4 = lora_results.get(f'rank8_lr2e4_{steps}steps', {}).get(metric, np.nan)
            r8_data.append([lr1e4, lr2e4])
        
        # Rank 16 row
        r16_data = []
        for steps in step_counts:
            lr1e4 = lora_results.get(f'rank16_lr1e4_{steps}steps', {}).get(metric, np.nan)
            lr2e4 = lora_results.get(f'rank16_lr2e4_{steps}steps', {}).get(metric, np.nan)
            r16_data.append([lr1e4, lr2e4])
        
        # Combine into heatmap data: rows = steps, cols = LR, separate plots for ranks
        for row, (rank, data) in enumerate([(8, r8_data), (16, r16_data)]):
            ax = axes[row, col]
            data_arr = np.array(data)
            
            # For LPIPS, invert colormap (lower is better)
            cmap = 'RdYlGn_r' if metric == 'lpips' else 'RdYlGn'
            
            im = ax.imshow(data_arr, cmap=cmap, aspect='auto')
            
            # Add value annotations
            for i in range(len(step_counts)):
                for j in range(2):
                    val = data_arr[i, j]
                    if not np.isnan(val):
                        fmt = '.2f' if metric == 'psnr' else '.3f'
                        text_color = 'white' if abs(val - np.nanmean(data_arr)) > np.nanstd(data_arr) else 'black'
                        ax.text(j, i, f'{val:{fmt}}', ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['LR 1e-4', 'LR 2e-4'])
            ax.set_yticks(range(len(step_counts)))
            ax.set_yticklabels([f'{s} steps' for s in step_counts])
            
            if row == 0:
                ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Rank {rank}', fontsize=12, fontweight='bold')
            
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('LoRA Hyperparameter Exploration\n(How Rank, LR, and Steps Affect Performance)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '1_lora_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '1_lora_heatmaps.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 1_lora_heatmaps.png/pdf")


def plot_steps_progression(lora_results: dict, output_dir: Path):
    """Line plots showing how performance changes with training steps."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    steps = [20, 50, 100]
    configs = [
        ('R8 LR1e-4', 'rank8_lr1e4', '#2ecc71', 'o'),
        ('R8 LR2e-4', 'rank8_lr2e4', '#27ae60', 's'),
        ('R16 LR1e-4', 'rank16_lr1e4', '#3498db', '^'),
        ('R16 LR2e-4', 'rank16_lr2e4', '#2980b9', 'd'),
    ]
    
    metrics = [('psnr', 'PSNR (dB) ↑'), ('ssim', 'SSIM ↑'), ('lpips', 'LPIPS ↓')]
    
    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        
        for label, prefix, color, marker in configs:
            values = []
            valid_steps = []
            for s in steps:
                key = f'{prefix}_{s}steps'
                if key in lora_results:
                    values.append(lora_results[key][metric])
                    valid_steps.append(s)
            
            if values:
                ax.plot(valid_steps, values, marker=marker, color=color, linewidth=2, 
                       markersize=8, label=label, alpha=0.8)
        
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Highlight best direction
        if metric == 'lpips':
            ax.invert_yaxis()  # For LPIPS, lower is better, so invert
    
    plt.suptitle('LoRA Performance vs Training Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '2_steps_progression.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '2_steps_progression.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 2_steps_progression.png/pdf")


def plot_lora_vs_full_ft(lora_results: dict, full_ft_results: dict, output_dir: Path):
    """Compare LoRA vs Full Fine-tuning: quality and efficiency."""
    if not HAS_MATPLOTLIB:
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Prepare comparable data (50 and 100 steps)
    comparisons = []
    
    # LoRA best configs at each step count
    for steps in [50, 100]:
        # Find best LoRA config at this step count (by LPIPS)
        lora_configs = {k: v for k, v in lora_results.items() if v['config']['steps'] == steps}
        if lora_configs:
            best_lora = min(lora_configs.items(), key=lambda x: x[1]['lpips'])
            comparisons.append({
                'method': f'LoRA {steps}s',
                'steps': steps,
                'type': 'LoRA',
                **best_lora[1]
            })
    
    # Full FT configs
    for key, data in full_ft_results.items():
        if data['config']['steps'] in [50, 100]:
            comparisons.append({
                'method': f"Full FT {data['config']['steps']}s",
                'steps': data['config']['steps'],
                'type': 'Full FT',
                **data
            })
    
    if not comparisons:
        print("No comparable data for LoRA vs Full FT")
        return
    
    # Plot 1: Time comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    lora_times = [c['train_time'] for c in comparisons if c['type'] == 'LoRA']
    full_ft_times = [c['train_time'] for c in comparisons if c['type'] == 'Full FT']
    lora_labels = [c['method'] for c in comparisons if c['type'] == 'LoRA']
    full_ft_labels = [c['method'] for c in comparisons if c['type'] == 'Full FT']
    
    x = np.arange(len(lora_times))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, lora_times, width, label='LoRA', color=COLORS['lora_r8'], alpha=0.8)
    if full_ft_times:
        bars2 = ax1.bar(x + width/2, full_ft_times[:len(x)], width, label='Full FT', color=COLORS['full_ft'], alpha=0.8)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lora_labels, rotation=15)
    ax1.legend()
    
    # Add speedup annotations
    for i, (lt, ft) in enumerate(zip(lora_times, full_ft_times[:len(lora_times)])):
        speedup = ft / lt if lt > 0 else 0
        ax1.annotate(f'{speedup:.1f}x faster', xy=(i, max(lt, ft)), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Plot 2: Quality comparison - PSNR (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    lora_psnr = [c['psnr'] for c in comparisons if c['type'] == 'LoRA']
    full_ft_psnr = [c['psnr'] for c in comparisons if c['type'] == 'Full FT']
    
    bars1 = ax2.bar(x - width/2, lora_psnr, width, label='LoRA', color=COLORS['lora_r8'], alpha=0.8)
    if full_ft_psnr:
        bars2 = ax2.bar(x + width/2, full_ft_psnr[:len(x)], width, label='Full FT', color=COLORS['full_ft'], alpha=0.8)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('PSNR Comparison (Higher = Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(lora_labels, rotation=15)
    ax2.legend()
    
    # Plot 3: Quality comparison - LPIPS (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    lora_lpips = [c['lpips'] for c in comparisons if c['type'] == 'LoRA']
    full_ft_lpips = [c['lpips'] for c in comparisons if c['type'] == 'Full FT']
    
    bars1 = ax3.bar(x - width/2, lora_lpips, width, label='LoRA', color=COLORS['lora_r8'], alpha=0.8)
    if full_ft_lpips:
        bars2 = ax3.bar(x + width/2, full_ft_lpips[:len(x)], width, label='Full FT', color=COLORS['full_ft'], alpha=0.8)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('LPIPS')
    ax3.set_title('LPIPS Comparison (Lower = Better)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(lora_labels, rotation=15)
    ax3.legend()
    
    # Plot 4: Efficiency scatter - Time vs Quality (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for c in comparisons:
        color = COLORS['lora_r8'] if c['type'] == 'LoRA' else COLORS['full_ft']
        marker = 'o' if c['type'] == 'LoRA' else 's'
        ax4.scatter(c['train_time'], c['lpips'], c=color, marker=marker, s=150, alpha=0.8, 
                   label=c['method'], edgecolors='black', linewidths=1)
        ax4.annotate(c['method'], (c['train_time'], c['lpips']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('LPIPS (lower = better)')
    ax4.set_title('Efficiency: Time vs Quality Trade-off', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add Pareto frontier indication
    ax4.axhline(y=min([c['lpips'] for c in comparisons]), color='green', linestyle='--', alpha=0.5, label='Best Quality')
    ax4.axvline(x=min([c['train_time'] for c in comparisons]), color='blue', linestyle='--', alpha=0.5, label='Fastest')
    
    plt.suptitle('LoRA vs Full Fine-Tuning Comparison', fontsize=16, fontweight='bold')
    plt.savefig(output_dir / '3_lora_vs_full_ft.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '3_lora_vs_full_ft.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 3_lora_vs_full_ft.png/pdf")


def plot_efficiency_summary(lora_results: dict, full_ft_results: dict, output_dir: Path):
    """Create a summary efficiency plot."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter: x = time, y = LPIPS (inverted for "quality"), size = PSNR
    all_results = []
    
    for key, data in lora_results.items():
        all_results.append({
            'name': key.replace('_', ' ').replace('rank', 'R').replace('lr', 'LR').replace('steps', 's'),
            'time': data['train_time'],
            'lpips': data['lpips'],
            'psnr': data['psnr'],
            'type': 'LoRA',
            'rank': data['config']['rank'],
        })
    
    for key, data in full_ft_results.items():
        all_results.append({
            'name': key.replace('full_ft_', 'Full FT ').replace('_', ' '),
            'time': data['train_time'],
            'lpips': data['lpips'],
            'psnr': data['psnr'],
            'type': 'Full FT',
            'rank': 0,
        })
    
    # Plot
    for r in all_results:
        if r['type'] == 'LoRA':
            color = COLORS['lora_r8'] if r['rank'] == 8 else COLORS['lora_r16']
            marker = 'o'
        else:
            color = COLORS['full_ft']
            marker = 's'
        
        # Size based on PSNR (normalized)
        size = 50 + (r['psnr'] - 10) * 30
        
        ax.scatter(r['time'], r['lpips'], c=color, marker=marker, s=size, alpha=0.7,
                  edgecolors='black', linewidths=0.5)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['lora_r8'], 
                  markersize=10, label='LoRA Rank 8'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['lora_r16'], 
                  markersize=10, label='LoRA Rank 16'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['full_ft'], 
                  markersize=10, label='Full Fine-Tuning'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlabel('Training Time per Video (seconds)', fontsize=12)
    ax.set_ylabel('LPIPS (lower = better quality)', fontsize=12)
    ax.set_title('Time vs Quality: All Methods', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to start at 0.4 to reduce whitespace and spread data points
    ax.set_ylim(0.4, 0.85)
    
    # Highlight the Pareto-optimal region
    ax.fill_between([0, 200], [0.4, 0.4], [0.65, 0.65], alpha=0.1, color='green', label='Efficient Region')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_efficiency_overview.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '4_efficiency_overview.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 4_efficiency_overview.png/pdf")


def plot_summary_table(lora_results: dict, full_ft_results: dict, output_dir: Path):
    """Create a comprehensive summary table as an image."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Combine all results
    all_data = []
    
    for key, data in lora_results.items():
        cfg = data['config']
        all_data.append([
            'LoRA',
            f"R{cfg['rank']}",
            f"{cfg['lr']:.0e}",
            str(cfg['steps']),
            f"{data['psnr']:.2f}",
            f"{data['ssim']:.4f}",
            f"{data['lpips']:.4f}",
            f"{data['train_time']:.0f}",
        ])
    
    for key, data in full_ft_results.items():
        cfg = data['config']
        all_data.append([
            'Full FT',
            '-',
            f"{cfg['lr']:.0e}",
            str(cfg['steps']),
            f"{data['psnr']:.2f}",
            f"{data['ssim']:.4f}",
            f"{data['lpips']:.4f}",
            f"{data['train_time']:.0f}",
        ])
    
    # Sort by LPIPS (lower is better)
    all_data.sort(key=lambda x: float(x[6]))
    
    headers = ['Method', 'Rank', 'LR', 'Steps', 'PSNR↑', 'SSIM↑', 'LPIPS↓', 'Time(s)']
    
    table = ax.table(cellText=all_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color-code by method
    for row_idx, row in enumerate(all_data, start=1):
        if row[0] == 'LoRA':
            for col in range(len(headers)):
                table[(row_idx, col)].set_facecolor('#e8f6f3')
        else:
            for col in range(len(headers)):
                table[(row_idx, col)].set_facecolor('#fdedec')
    
    # Highlight best values
    psnr_vals = [float(r[4]) for r in all_data]
    ssim_vals = [float(r[5]) for r in all_data]
    lpips_vals = [float(r[6]) for r in all_data]
    time_vals = [float(r[7]) for r in all_data]
    
    best_psnr_idx = psnr_vals.index(max(psnr_vals)) + 1
    best_ssim_idx = ssim_vals.index(max(ssim_vals)) + 1
    best_lpips_idx = lpips_vals.index(min(lpips_vals)) + 1
    best_time_idx = time_vals.index(min(time_vals)) + 1
    
    table[(best_psnr_idx, 4)].set_facecolor('#27ae60')
    table[(best_psnr_idx, 4)].set_text_props(color='white', fontweight='bold')
    table[(best_ssim_idx, 5)].set_facecolor('#27ae60')
    table[(best_ssim_idx, 5)].set_text_props(color='white', fontweight='bold')
    table[(best_lpips_idx, 6)].set_facecolor('#27ae60')
    table[(best_lpips_idx, 6)].set_text_props(color='white', fontweight='bold')
    table[(best_time_idx, 7)].set_facecolor('#3498db')
    table[(best_time_idx, 7)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Complete Results Summary (Sorted by LPIPS)\nGreen = Best Quality  |  Blue = Fastest', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / '5_summary_table.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '5_summary_table.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 5_summary_table.png/pdf")


def plot_speedup_chart(lora_results: dict, full_ft_results: dict, output_dir: Path):
    """Bar chart showing speedup of LoRA over Full FT."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compare at 50 and 100 steps
    comparisons = []
    
    for steps in [50, 100]:
        # Get LoRA time (best config at this step)
        lora_configs = {k: v for k, v in lora_results.items() if v['config']['steps'] == steps}
        if lora_configs:
            best_lora = min(lora_configs.items(), key=lambda x: x[1]['lpips'])
            lora_time = best_lora[1]['train_time']
            lora_name = best_lora[0]
        else:
            continue
        
        # Get Full FT time
        ft_key = f'full_ft_{steps}steps_5e5'
        if ft_key in full_ft_results:
            ft_time = full_ft_results[ft_key]['train_time']
            speedup = ft_time / lora_time if lora_time > 0 else 0
            
            comparisons.append({
                'steps': steps,
                'lora_time': lora_time,
                'ft_time': ft_time,
                'speedup': speedup,
            })
    
    if not comparisons:
        print("No speedup data available")
        return
    
    x = np.arange(len(comparisons))
    width = 0.35
    
    lora_times = [c['lora_time'] for c in comparisons]
    ft_times = [c['ft_time'] for c in comparisons]
    
    bars1 = ax.bar(x - width/2, lora_times, width, label='LoRA', color=COLORS['lora_r8'])
    bars2 = ax.bar(x + width/2, ft_times, width, label='Full Fine-Tuning', color=COLORS['full_ft'])
    
    # Add speedup labels
    for i, c in enumerate(comparisons):
        max_height = max(c['lora_time'], c['ft_time'])
        ax.annotate(f"{c['speedup']:.1f}x faster!", 
                   xy=(i, max_height), xytext=(0, 10),
                   textcoords='offset points', ha='center',
                   fontsize=12, fontweight='bold', color='green')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('LoRA Speedup Over Full Fine-Tuning', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c["steps"]} steps' for c in comparisons])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '6_speedup_chart.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '6_speedup_chart.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 6_speedup_chart.png/pdf")


def main():
    script_dir = Path(__file__).parent
    lora_results_dir = script_dir.parent.parent / 'results'
    naive_results_dir = script_dir.parent.parent.parent / 'naive_experiment' / 'scripts' / 'results'
    output_dir = script_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Loading LoRA Results...")
    lora_results = load_lora_results(lora_results_dir)
    
    print("\nLoading Full Fine-Tuning Results...")
    full_ft_results = load_full_ft_results(naive_results_dir)
    
    print("=" * 60)
    print(f"Found {len(lora_results)} LoRA configs")
    print(f"Found {len(full_ft_results)} Full FT configs")
    print("=" * 60)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_lora_heatmaps(lora_results, output_dir)
    plot_steps_progression(lora_results, output_dir)
    plot_lora_vs_full_ft(lora_results, full_ft_results, output_dir)
    plot_efficiency_summary(lora_results, full_ft_results, output_dir)
    plot_summary_table(lora_results, full_ft_results, output_dir)
    plot_speedup_chart(lora_results, full_ft_results, output_dir)
    
    print("=" * 60)
    print("All plots generated!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

