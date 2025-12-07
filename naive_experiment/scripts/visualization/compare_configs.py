#!/usr/bin/env python3
"""
Generate comprehensive visualizations comparing different fine-tuning configurations.
For PI presentation.

Usage:
    python compare_configs.py --results-dir ../results --output-dir ./plots
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Use a clean, professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

# Color palette - professional and colorblind-friendly
COLORS = {
    '15steps_1e4': '#E64B35',   # Red
    '15steps_2e4': '#4DBBD5',   # Cyan  
    '50steps_1e5': '#00A087',   # Teal
    '50steps_5e5': '#3C5488',   # Blue
    '100steps_5e5': '#F39B7F',  # Coral
}

CONFIG_LABELS = {
    '15steps_1e4': '15 steps @ 1e-4',
    '15steps_2e4': '15 steps @ 2e-4',
    '50steps_1e5': '50 steps @ 1e-5',
    '50steps_5e5': '50 steps @ 5e-5',
    '100steps_5e5': '100 steps @ 5e-5',
}


def load_metrics(results_dir):
    """Load metrics from all config directories."""
    results_dir = Path(results_dir)
    all_metrics = {}
    
    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        
        metrics_file = config_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: No metrics.json in {config_dir.name}")
            continue
        
        with open(metrics_file) as f:
            data = json.load(f)
        
        all_metrics[config_dir.name] = data
        print(f"Loaded {len(data)} videos from {config_dir.name}")
    
    return all_metrics


def compute_summary_stats(all_metrics):
    """Compute summary statistics for each config."""
    summary = {}
    
    for config_name, videos in all_metrics.items():
        stats = {
            'n_videos': len(videos),
            'baseline_psnr': [],
            'finetuned_psnr': [],
            'baseline_ssim': [],
            'finetuned_ssim': [],
            'baseline_lpips': [],
            'finetuned_lpips': [],
            'finetune_time': [],
            'inference_time': [],
        }
        
        for v in videos:
            if 'baseline' in v and 'finetuned' in v:
                stats['baseline_psnr'].append(v['baseline'].get('psnr', 0))
                stats['finetuned_psnr'].append(v['finetuned'].get('psnr', 0))
                stats['baseline_ssim'].append(v['baseline'].get('ssim', 0))
                stats['finetuned_ssim'].append(v['finetuned'].get('ssim', 0))
                stats['baseline_lpips'].append(v['baseline'].get('lpips', 1))
                stats['finetuned_lpips'].append(v['finetuned'].get('lpips', 1))
            
            if 'finetune_time_sec' in v:
                stats['finetune_time'].append(v['finetune_time_sec'])
            if 'finetuned_inference_time_sec' in v:
                stats['inference_time'].append(v['finetuned_inference_time_sec'])
        
        # Compute improvements
        if stats['baseline_psnr'] and stats['finetuned_psnr']:
            stats['psnr_improvement'] = np.array(stats['finetuned_psnr']) - np.array(stats['baseline_psnr'])
            stats['ssim_improvement'] = np.array(stats['finetuned_ssim']) - np.array(stats['baseline_ssim'])
            stats['lpips_improvement'] = np.array(stats['baseline_lpips']) - np.array(stats['finetuned_lpips'])  # Lower is better
        
        summary[config_name] = stats
    
    return summary


def plot_1_metrics_comparison_bar(summary, output_dir):
    """Bar chart comparing average metrics across configs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    configs = sorted(summary.keys())
    x = np.arange(len(configs))
    width = 0.35
    
    # PSNR
    ax = axes[0]
    baseline_vals = [np.mean(summary[c]['baseline_psnr']) for c in configs]
    finetuned_vals = [np.mean(summary[c]['finetuned_psnr']) for c in configs]
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#888888', alpha=0.7)
    bars2 = ax.bar(x + width/2, finetuned_vals, width, label='Fine-tuned', 
                   color=[COLORS.get(c, '#333') for c in configs])
    
    ax.set_ylabel('PSNR (dB) ↑')
    ax.set_title('PSNR Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, max(finetuned_vals) * 1.2)
    
    # SSIM
    ax = axes[1]
    baseline_vals = [np.mean(summary[c]['baseline_ssim']) for c in configs]
    finetuned_vals = [np.mean(summary[c]['finetuned_ssim']) for c in configs]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#888888', alpha=0.7)
    ax.bar(x + width/2, finetuned_vals, width, label='Fine-tuned',
           color=[COLORS.get(c, '#333') for c in configs])
    
    ax.set_ylabel('SSIM ↑')
    ax.set_title('SSIM Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # LPIPS (lower is better)
    ax = axes[2]
    baseline_vals = [np.mean(summary[c]['baseline_lpips']) for c in configs]
    finetuned_vals = [np.mean(summary[c]['finetuned_lpips']) for c in configs]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#888888', alpha=0.7)
    ax.bar(x + width/2, finetuned_vals, width, label='Fine-tuned',
           color=[COLORS.get(c, '#333') for c in configs])
    
    ax.set_ylabel('LPIPS ↓')
    ax.set_title('LPIPS Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_metrics_comparison_bar.png', bbox_inches='tight')
    plt.savefig(output_dir / '1_metrics_comparison_bar.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 1_metrics_comparison_bar.png/pdf")


def plot_2_improvement_distribution(summary, output_dir):
    """Box plots showing distribution of improvements."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    configs = sorted(summary.keys())
    
    # PSNR Improvement
    ax = axes[0]
    data = [summary[c]['psnr_improvement'] for c in configs if 'psnr_improvement' in summary[c]]
    bp = ax.boxplot(data, patch_artist=True, labels=[CONFIG_LABELS.get(c, c) for c in configs])
    for patch, c in zip(bp['boxes'], configs):
        patch.set_facecolor(COLORS.get(c, '#888'))
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No improvement')
    ax.set_ylabel('PSNR Improvement (dB)')
    ax.set_title('PSNR Improvement Distribution')
    ax.tick_params(axis='x', rotation=30)
    
    # SSIM Improvement
    ax = axes[1]
    data = [summary[c]['ssim_improvement'] for c in configs if 'ssim_improvement' in summary[c]]
    bp = ax.boxplot(data, patch_artist=True, labels=[CONFIG_LABELS.get(c, c) for c in configs])
    for patch, c in zip(bp['boxes'], configs):
        patch.set_facecolor(COLORS.get(c, '#888'))
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('SSIM Improvement')
    ax.set_title('SSIM Improvement Distribution')
    ax.tick_params(axis='x', rotation=30)
    
    # LPIPS Improvement (positive = better)
    ax = axes[2]
    data = [summary[c]['lpips_improvement'] for c in configs if 'lpips_improvement' in summary[c]]
    bp = ax.boxplot(data, patch_artist=True, labels=[CONFIG_LABELS.get(c, c) for c in configs])
    for patch, c in zip(bp['boxes'], configs):
        patch.set_facecolor(COLORS.get(c, '#888'))
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('LPIPS Improvement (↑ = better)')
    ax.set_title('LPIPS Improvement Distribution')
    ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_improvement_distribution.png', bbox_inches='tight')
    plt.savefig(output_dir / '2_improvement_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 2_improvement_distribution.png/pdf")


def plot_3_time_vs_quality(summary, output_dir):
    """Scatter plot: Fine-tuning time vs Quality improvement (Pareto frontier)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    configs = sorted(summary.keys())
    
    metrics = [
        ('psnr_improvement', 'PSNR Improvement (dB)', axes[0]),
        ('ssim_improvement', 'SSIM Improvement', axes[1]),
        ('lpips_improvement', 'LPIPS Improvement', axes[2]),
    ]
    
    for metric_key, metric_label, ax in metrics:
        for c in configs:
            if metric_key not in summary[c] or not summary[c]['finetune_time']:
                continue
            
            avg_time = np.mean(summary[c]['finetune_time']) / 60  # Convert to minutes
            avg_improvement = np.mean(summary[c][metric_key])
            std_improvement = np.std(summary[c][metric_key])
            
            ax.errorbar(avg_time, avg_improvement, yerr=std_improvement,
                       fmt='o', markersize=12, capsize=5,
                       color=COLORS.get(c, '#888'), label=CONFIG_LABELS.get(c, c),
                       markeredgecolor='white', markeredgewidth=1.5)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('Fine-tuning Time (minutes)')
        ax.set_ylabel(metric_label)
        ax.legend(loc='best', fontsize=9)
        ax.set_title(f'Time vs {metric_label.split()[0]}')
    
    plt.suptitle('Quality-Time Trade-off (Error bars = 1 std)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '3_time_vs_quality.png', bbox_inches='tight')
    plt.savefig(output_dir / '3_time_vs_quality.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 3_time_vs_quality.png/pdf")


def plot_4_runtime_breakdown(summary, output_dir):
    """Stacked bar chart showing runtime breakdown per config."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = sorted(summary.keys())
    x = np.arange(len(configs))
    
    finetune_times = [np.mean(summary[c]['finetune_time'])/60 if summary[c]['finetune_time'] else 0 
                     for c in configs]
    inference_times = [np.mean(summary[c]['inference_time'])/60 if summary[c]['inference_time'] else 0 
                      for c in configs]
    
    bars1 = ax.bar(x, finetune_times, label='Fine-tuning', color='#3C5488')
    bars2 = ax.bar(x, inference_times, bottom=finetune_times, label='Inference', color='#4DBBD5')
    
    # Add total time labels
    for i, (ft, it) in enumerate(zip(finetune_times, inference_times)):
        total = ft + it
        ax.annotate(f'{total:.1f}m', xy=(i, total), ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Time per Video (minutes)')
    ax.set_title('Runtime Breakdown per Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], rotation=30, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_runtime_breakdown.png', bbox_inches='tight')
    plt.savefig(output_dir / '4_runtime_breakdown.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 4_runtime_breakdown.png/pdf")


def plot_5_win_rate(summary, output_dir):
    """Bar chart showing % of videos that improved."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    configs = sorted(summary.keys())
    x = np.arange(len(configs))
    
    metrics = [
        ('psnr_improvement', 'PSNR', axes[0]),
        ('ssim_improvement', 'SSIM', axes[1]),
        ('lpips_improvement', 'LPIPS', axes[2]),
    ]
    
    for metric_key, metric_name, ax in metrics:
        win_rates = []
        for c in configs:
            if metric_key in summary[c]:
                improvements = summary[c][metric_key]
                win_rate = (np.array(improvements) > 0).mean() * 100
            else:
                win_rate = 0
            win_rates.append(win_rate)
        
        bars = ax.bar(x, win_rates, color=[COLORS.get(c, '#888') for c in configs])
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        
        # Add percentage labels
        for i, v in enumerate(win_rates):
            ax.annotate(f'{v:.1f}%', xy=(i, v), ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Videos Improved (%)')
        ax.set_title(f'{metric_name} Win Rate')
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], rotation=30, ha='right')
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_win_rate.png', bbox_inches='tight')
    plt.savefig(output_dir / '5_win_rate.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 5_win_rate.png/pdf")


def plot_6_summary_table(summary, output_dir):
    """Create a summary table as an image."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    configs = sorted(summary.keys())
    
    # Build table data
    headers = ['Config', 'Videos', 'PSNR\n(B→FT)', 'SSIM\n(B→FT)', 'LPIPS\n(B→FT)', 
               'Avg Δ PSNR', 'Avg Δ SSIM', 'Avg Δ LPIPS', 'FT Time\n(min)', 'Win %']
    
    rows = []
    for c in configs:
        s = summary[c]
        n = s['n_videos']
        
        b_psnr = np.mean(s['baseline_psnr']) if s['baseline_psnr'] else 0
        f_psnr = np.mean(s['finetuned_psnr']) if s['finetuned_psnr'] else 0
        b_ssim = np.mean(s['baseline_ssim']) if s['baseline_ssim'] else 0
        f_ssim = np.mean(s['finetuned_ssim']) if s['finetuned_ssim'] else 0
        b_lpips = np.mean(s['baseline_lpips']) if s['baseline_lpips'] else 0
        f_lpips = np.mean(s['finetuned_lpips']) if s['finetuned_lpips'] else 0
        
        d_psnr = np.mean(s['psnr_improvement']) if 'psnr_improvement' in s else 0
        d_ssim = np.mean(s['ssim_improvement']) if 'ssim_improvement' in s else 0
        d_lpips = np.mean(s['lpips_improvement']) if 'lpips_improvement' in s else 0
        
        ft_time = np.mean(s['finetune_time'])/60 if s['finetune_time'] else 0
        
        # Win rate (average across all metrics)
        win_rates = []
        for key in ['psnr_improvement', 'ssim_improvement', 'lpips_improvement']:
            if key in s:
                win_rates.append((np.array(s[key]) > 0).mean() * 100)
        avg_win = np.mean(win_rates) if win_rates else 0
        
        rows.append([
            CONFIG_LABELS.get(c, c),
            str(n),
            f'{b_psnr:.2f}→{f_psnr:.2f}',
            f'{b_ssim:.3f}→{f_ssim:.3f}',
            f'{b_lpips:.3f}→{f_lpips:.3f}',
            f'{d_psnr:+.2f}',
            f'{d_ssim:+.4f}',
            f'{d_lpips:+.4f}',
            f'{ft_time:.1f}',
            f'{avg_win:.1f}%'
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#3C5488')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Color rows by config
    for i, c in enumerate(configs):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(COLORS.get(c, '#fff') + '30')  # 30 = alpha
    
    plt.title('Summary of Fine-tuning Configurations', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '6_summary_table.png', bbox_inches='tight', dpi=200)
    plt.savefig(output_dir / '6_summary_table.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 6_summary_table.png/pdf")


def plot_7_radar_chart(summary, output_dir):
    """Radar chart comparing configs across multiple dimensions."""
    from math import pi
    
    configs = sorted(summary.keys())
    
    # Metrics to compare (normalized to 0-1 scale)
    categories = ['PSNR\nImprove', 'SSIM\nImprove', 'LPIPS\nImprove', 'Speed\n(1/time)', 'Win\nRate']
    N = len(categories)
    
    # Compute normalized values
    all_values = {c: [] for c in configs}
    
    # Get raw values for normalization
    raw_psnr = {c: np.mean(summary[c].get('psnr_improvement', [0])) for c in configs}
    raw_ssim = {c: np.mean(summary[c].get('ssim_improvement', [0])) for c in configs}
    raw_lpips = {c: np.mean(summary[c].get('lpips_improvement', [0])) for c in configs}
    raw_time = {c: np.mean(summary[c]['finetune_time']) if summary[c]['finetune_time'] else 1 for c in configs}
    raw_win = {c: np.mean([(np.array(summary[c].get(k, [0])) > 0).mean() 
                           for k in ['psnr_improvement', 'ssim_improvement', 'lpips_improvement']])
               for c in configs}
    
    # Normalize (min-max to 0-1)
    def normalize(d):
        vals = list(d.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 0.5 for k in d}
        return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}
    
    norm_psnr = normalize(raw_psnr)
    norm_ssim = normalize(raw_ssim)
    norm_lpips = normalize(raw_lpips)
    # For time, lower is better, so invert
    norm_speed = {c: 1 - v for c, v in normalize(raw_time).items()}
    norm_win = normalize(raw_win)
    
    for c in configs:
        all_values[c] = [norm_psnr[c], norm_ssim[c], norm_lpips[c], norm_speed[c], norm_win[c]]
    
    # Create radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for c in configs:
        values = all_values[c]
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=CONFIG_LABELS.get(c, c), color=COLORS.get(c, '#888'))
        ax.fill(angles, values, alpha=0.15, color=COLORS.get(c, '#888'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.title('Multi-Dimensional Config Comparison\n(Normalized 0-1)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '7_radar_chart.png', bbox_inches='tight')
    plt.savefig(output_dir / '7_radar_chart.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 7_radar_chart.png/pdf")


def main():
    parser = argparse.ArgumentParser(description='Compare fine-tuning configurations')
    parser.add_argument('--results-dir', type=str, default='../results',
                        help='Directory containing config subdirectories')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading metrics from all configurations...")
    all_metrics = load_metrics(results_dir)
    
    if not all_metrics:
        print("ERROR: No metrics found!")
        return
    
    print(f"\nComputing summary statistics for {len(all_metrics)} configs...")
    summary = compute_summary_stats(all_metrics)
    
    print("\nGenerating visualizations...")
    print("=" * 50)
    
    plot_1_metrics_comparison_bar(summary, output_dir)
    plot_2_improvement_distribution(summary, output_dir)
    plot_3_time_vs_quality(summary, output_dir)
    plot_4_runtime_breakdown(summary, output_dir)
    plot_5_win_rate(summary, output_dir)
    plot_6_summary_table(summary, output_dir)
    plot_7_radar_chart(summary, output_dir)
    
    print("=" * 50)
    print(f"\n✅ All plots saved to: {output_dir.absolute()}")
    print("\nPlots generated:")
    print("  1. metrics_comparison_bar - Side-by-side baseline vs fine-tuned")
    print("  2. improvement_distribution - Box plots of improvement per config")
    print("  3. time_vs_quality - Pareto frontier (quality vs speed trade-off)")
    print("  4. runtime_breakdown - Stacked bar of FT + inference time")
    print("  5. win_rate - % of videos that improved")
    print("  6. summary_table - Comprehensive table of all stats")
    print("  7. radar_chart - Multi-dimensional comparison")


if __name__ == "__main__":
    main()

