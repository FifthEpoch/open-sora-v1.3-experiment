#!/usr/bin/env python3
"""
Compare LoRA vs Full Fine-tuning using PURE TRAINING TIME only.
Excludes model loading, video encoding, and other overhead.

Based on timing test results:
- LoRA: ~0.744s per gradient step
- Model load overhead: ~50s (one-time per video)
- Video encode overhead: ~1.6s (one-time per video)
"""

import json
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# Timing constants from the timing test
LORA_OVERHEAD_SEC = 51.7  # Model load + video encode (one-time per video)
LORA_SEC_PER_STEP = 0.744  # Pure training time per gradient step

# Full FT timing estimation (from naive experiment data)
# 50 steps: 419s, 100 steps: 760s
# Difference: 341s for 50 extra steps = 6.82s/step
# But this includes some overhead. Let's estimate:
# Assuming similar model load overhead (~50s) + checkpoint save overhead (~30s per video)
FULL_FT_OVERHEAD_SEC = 80.0  # Estimated: model load + checkpoint operations
FULL_FT_SEC_PER_STEP = 6.8   # Estimated from (760-419)/(100-50)

# Colors
COLORS = {
    'lora': '#2ecc71',
    'full_ft': '#e74c3c',
    'overhead': '#95a5a6',
}


def load_lora_results(results_dir: Path) -> dict:
    """Load LoRA results and compute pure training times."""
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
    }
    
    results = {}
    for dir_name, config in configs.items():
        metrics_file = results_dir / dir_name / 'metrics_summary.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            steps = config['steps']
            pure_train_time = steps * LORA_SEC_PER_STEP
            total_time = LORA_OVERHEAD_SEC + pure_train_time
            
            results[dir_name] = {
                'config': config,
                'psnr': data.get('avg_psnr', 0),
                'ssim': data.get('avg_ssim', 0),
                'lpips': data.get('avg_lpips', 1),
                'pure_train_time': pure_train_time,
                'overhead': LORA_OVERHEAD_SEC,
                'total_time': total_time,
                'reported_time': data.get('avg_train_time_sec') or data.get('avg_train_time', 0),
            }
    
    return results


def load_full_ft_results(results_dir: Path) -> dict:
    """Load full fine-tuning results and estimate pure training times."""
    configs = {
        '50steps_5e5': {'steps': 50, 'lr': 5e-5},
        '100steps_5e5': {'steps': 100, 'lr': 5e-5},
        '50steps_1e5': {'steps': 50, 'lr': 1e-5},
        '15steps_2e4': {'steps': 15, 'lr': 2e-4},
        '15steps_1e4': {'steps': 15, 'lr': 1e-4},
    }
    
    results = {}
    for dir_name, config in configs.items():
        metrics_file = results_dir / dir_name / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                psnr_vals = [v.get('finetuned', {}).get('psnr', 0) for v in data if 'finetuned' in v]
                ssim_vals = [v.get('finetuned', {}).get('ssim', 0) for v in data if 'finetuned' in v]
                lpips_vals = [v.get('finetuned', {}).get('lpips', 1) for v in data if 'finetuned' in v]
                total_times = [v.get('finetune_time_sec', 0) for v in data]
                
                if psnr_vals:
                    steps = config['steps']
                    avg_total_time = np.mean(total_times)
                    
                    # Estimate pure training time
                    pure_train_time = steps * FULL_FT_SEC_PER_STEP
                    
                    results[f'full_ft_{dir_name}'] = {
                        'config': config,
                        'psnr': np.mean(psnr_vals),
                        'ssim': np.mean(ssim_vals),
                        'lpips': np.mean(lpips_vals),
                        'pure_train_time': pure_train_time,
                        'overhead': FULL_FT_OVERHEAD_SEC,
                        'total_time': avg_total_time,
                        'num_videos': len(psnr_vals),
                    }
    
    return results


def plot_pure_training_time_comparison(lora_results: dict, full_ft_results: dict, output_dir: Path):
    """Bar chart comparing pure training time (no overhead)."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Time breakdown for different step counts
    ax1 = axes[0]
    step_counts = [20, 50, 100]
    lora_pure = [s * LORA_SEC_PER_STEP for s in step_counts]
    full_ft_pure = [s * FULL_FT_SEC_PER_STEP for s in step_counts]
    
    x = np.arange(len(step_counts))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, lora_pure, width, label='LoRA', color=COLORS['lora'])
    bars2 = ax1.bar(x + width/2, full_ft_pure, width, label='Full Fine-Tuning', color=COLORS['full_ft'])
    
    # Add speedup annotations
    for i, (lt, ft) in enumerate(zip(lora_pure, full_ft_pure)):
        speedup = ft / lt if lt > 0 else 0
        ax1.annotate(f'{speedup:.1f}x', 
                    xy=(i, max(lt, ft)), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold', color='green')
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Pure Training Time (seconds)', fontsize=12)
    ax1.set_title('Pure Training Time Comparison\n(Gradient Steps Only - No Overhead)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s} steps' for s in step_counts])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Plot 2: Time per gradient step
    ax2 = axes[1]
    methods = ['LoRA', 'Full Fine-Tuning']
    times_per_step = [LORA_SEC_PER_STEP, FULL_FT_SEC_PER_STEP]
    colors = [COLORS['lora'], COLORS['full_ft']]
    
    bars = ax2.bar(methods, times_per_step, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, t in zip(bars, times_per_step):
        ax2.annotate(f'{t:.3f}s', xy=(bar.get_x() + bar.get_width() / 2, t),
                    xytext=(0, 5), textcoords="offset points", ha='center', 
                    fontsize=14, fontweight='bold')
    
    # Add speedup annotation
    speedup = FULL_FT_SEC_PER_STEP / LORA_SEC_PER_STEP
    ax2.annotate(f'LoRA is {speedup:.1f}x faster per step!', 
                xy=(0.5, max(times_per_step) * 0.7), xycoords='axes fraction',
                fontsize=12, ha='center', color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.set_ylabel('Time per Gradient Step (seconds)', fontsize=12)
    ax2.set_title('Time per Gradient Step', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '7_pure_training_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '7_pure_training_time.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 7_pure_training_time.png/pdf")


def plot_time_breakdown(output_dir: Path):
    """Stacked bar chart showing overhead vs pure training time."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data for different step counts
    step_counts = [20, 50, 100]
    
    # LoRA times
    lora_overhead = [LORA_OVERHEAD_SEC] * 3
    lora_train = [s * LORA_SEC_PER_STEP for s in step_counts]
    lora_total = [o + t for o, t in zip(lora_overhead, lora_train)]
    
    # Full FT times
    full_ft_overhead = [FULL_FT_OVERHEAD_SEC] * 3
    full_ft_train = [s * FULL_FT_SEC_PER_STEP for s in step_counts]
    full_ft_total = [o + t for o, t in zip(full_ft_overhead, full_ft_train)]
    
    x = np.arange(len(step_counts))
    width = 0.35
    
    # Stacked bars for LoRA
    ax.bar(x - width/2, lora_overhead, width, label='LoRA Overhead', color=COLORS['overhead'], alpha=0.7)
    ax.bar(x - width/2, lora_train, width, bottom=lora_overhead, label='LoRA Training', color=COLORS['lora'])
    
    # Stacked bars for Full FT
    ax.bar(x + width/2, full_ft_overhead, width, label='Full FT Overhead', color=COLORS['overhead'], alpha=0.5, hatch='//')
    ax.bar(x + width/2, full_ft_train, width, bottom=full_ft_overhead, label='Full FT Training', color=COLORS['full_ft'])
    
    # Add total time labels
    for i, (lt, ft) in enumerate(zip(lora_total, full_ft_total)):
        ax.annotate(f'{lt:.0f}s', xy=(x[i] - width/2, lt + 5), ha='center', fontsize=10, fontweight='bold')
        ax.annotate(f'{ft:.0f}s', xy=(x[i] + width/2, ft + 5), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Time Breakdown: Overhead vs Pure Training\n(Per Video)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s} steps' for s in step_counts])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation explaining overhead
    ax.text(0.98, 0.02, 
            f'Overhead = Model loading + LoRA injection + Video encoding\n'
            f'LoRA overhead: {LORA_OVERHEAD_SEC:.1f}s | Full FT overhead: {FULL_FT_OVERHEAD_SEC:.1f}s',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / '8_time_breakdown.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '8_time_breakdown.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 8_time_breakdown.png/pdf")


def plot_efficiency_pure_time(lora_results: dict, full_ft_results: dict, output_dir: Path):
    """Scatter plot: Pure training time vs quality."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot LoRA results
    for key, data in lora_results.items():
        ax.scatter(data['pure_train_time'], data['lpips'], 
                  c=COLORS['lora'], marker='o', s=100, alpha=0.7,
                  edgecolors='black', linewidths=0.5)
    
    # Plot Full FT results
    for key, data in full_ft_results.items():
        ax.scatter(data['pure_train_time'], data['lpips'],
                  c=COLORS['full_ft'], marker='s', s=100, alpha=0.7,
                  edgecolors='black', linewidths=0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['lora'], 
              markersize=10, label='LoRA'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['full_ft'], 
              markersize=10, label='Full Fine-Tuning'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    ax.set_xlabel('Pure Training Time (seconds) - Gradient Steps Only', fontsize=12)
    ax.set_ylabel('LPIPS (lower = better quality)', fontsize=12)
    ax.set_title('Pure Training Time vs Quality\n(Excluding Model Load & Encoding Overhead)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight efficient region
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.5)
    ax.text(75, ax.get_ylim()[0] + 0.02, '← Efficient', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '9_efficiency_pure_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '9_efficiency_pure_time.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 9_efficiency_pure_time.png/pdf")


def plot_summary_comparison(output_dir: Path):
    """Summary comparison table."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Summary data
    headers = ['Metric', 'LoRA', 'Full Fine-Tuning', 'LoRA Advantage']
    data = [
        ['Time per step', f'{LORA_SEC_PER_STEP:.3f}s', f'{FULL_FT_SEC_PER_STEP:.2f}s', f'{FULL_FT_SEC_PER_STEP/LORA_SEC_PER_STEP:.1f}x faster'],
        ['Overhead (per video)', f'{LORA_OVERHEAD_SEC:.1f}s', f'{FULL_FT_OVERHEAD_SEC:.1f}s', '-'],
        ['20 steps (pure train)', f'{20*LORA_SEC_PER_STEP:.1f}s', f'{20*FULL_FT_SEC_PER_STEP:.0f}s', f'{20*FULL_FT_SEC_PER_STEP/(20*LORA_SEC_PER_STEP):.1f}x faster'],
        ['50 steps (pure train)', f'{50*LORA_SEC_PER_STEP:.1f}s', f'{50*FULL_FT_SEC_PER_STEP:.0f}s', f'{50*FULL_FT_SEC_PER_STEP/(50*LORA_SEC_PER_STEP):.1f}x faster'],
        ['100 steps (pure train)', f'{100*LORA_SEC_PER_STEP:.1f}s', f'{100*FULL_FT_SEC_PER_STEP:.0f}s', f'{100*FULL_FT_SEC_PER_STEP/(100*LORA_SEC_PER_STEP):.1f}x faster'],
        ['Trainable params', '~5.2M (0.4%)', '~1.2B (100%)', '230x fewer params'],
    ]
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color advantage column
    for row in range(1, len(data) + 1):
        table[(row, 3)].set_facecolor('#d5f5e3')
        table[(row, 3)].set_text_props(color='green', fontweight='bold')
    
    plt.title('LoRA vs Full Fine-Tuning: Efficiency Summary\n(Based on Pure Training Time)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / '10_efficiency_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '10_efficiency_summary.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: 10_efficiency_summary.png/pdf")


def main():
    script_dir = Path(__file__).parent
    lora_results_dir = script_dir.parent.parent / 'results'
    naive_results_dir = script_dir.parent.parent.parent / 'naive_experiment' / 'scripts' / 'results'
    output_dir = script_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Pure Training Time Analysis")
    print("=" * 60)
    print(f"LoRA: {LORA_SEC_PER_STEP:.3f}s per gradient step")
    print(f"Full FT: {FULL_FT_SEC_PER_STEP:.2f}s per gradient step")
    print(f"Speedup: {FULL_FT_SEC_PER_STEP/LORA_SEC_PER_STEP:.1f}x faster with LoRA")
    print("=" * 60)
    
    # Load results
    print("\nLoading results...")
    lora_results = load_lora_results(lora_results_dir)
    full_ft_results = load_full_ft_results(naive_results_dir)
    
    print(f"Found {len(lora_results)} LoRA configs")
    print(f"Found {len(full_ft_results)} Full FT configs")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_pure_training_time_comparison(lora_results, full_ft_results, output_dir)
    plot_time_breakdown(output_dir)
    plot_efficiency_pure_time(lora_results, full_ft_results, output_dir)
    plot_summary_comparison(output_dir)
    
    print("=" * 60)
    print("All pure training time plots generated!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

