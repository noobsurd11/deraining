"""
Generate publication-quality comparison plots from baseline CSV results.

Usage:
    python results/baselines/generate_plots.py
"""
import os
import glob
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).parent / 'plots'
TABLES_DIR = Path(__file__).parent / 'tables'
PLOTS_DIR.mkdir(exist_ok=True)

# Consistent colors per model
COLORS = {
    'restormer': '#2E86C1',
    'drsformer': '#1A5276',
    'nafnet_w32': '#27AE60',
    'nafnet_w64': '#1E8449',
    'diffmamba': '#E74C3C',
}

MARKERS = {
    'restormer': 'o',
    'drsformer': 's',
    'nafnet_w32': '^',
    'nafnet_w64': 'D',
    'diffmamba': 'v',
}


def load_all_results():
    """Load all CSV result files into a list of dicts."""
    results = []
    for csv_path in glob.glob(str(TABLES_DIR / '*_results.csv')):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ['psnr', 'ssim', 'lpips', 'params_M', 'gmacs',
                            'mean_ms', 'std_ms', 'peak_mem_GB', 'model_size_MB']:
                    if key in row:
                        row[key] = float(row[key])
                results.append(row)
    return results


def plot_bar_comparison(results, metric, ylabel, title, filename):
    """Grouped bar chart: models x datasets."""
    models = sorted(set(r['model'] for r in results))
    datasets = sorted(set(r['dataset'] for r in results))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            match = [r for r in results if r['model'] == model and r['dataset'] == ds]
            vals.append(match[0][metric] if match else 0)
        ax.bar(x + i * width, vals, width, label=model,
               color=COLORS.get(model, '#888'), edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_scatter(results, x_key, y_key, xlabel, ylabel, title, filename):
    """Scatter plot: quality vs efficiency."""
    models = sorted(set(r['model'] for r in results))

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        if not model_results:
            continue
        x_val = model_results[0][x_key]
        y_val = np.mean([r[y_key] for r in model_results])  # avg across datasets
        ax.scatter(x_val, y_val, c=COLORS.get(model, '#888'),
                   marker=MARKERS.get(model, 'o'), s=150, edgecolors='black',
                   linewidth=0.5, zorder=5, label=model)
        ax.annotate(model, (x_val, y_val), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def generate_summary_table(results):
    """Render a summary table as an image."""
    models = sorted(set(r['model'] for r in results))
    datasets = sorted(set(r['dataset'] for r in results))

    # Build table data: rows = models, columns per dataset (PSNR/SSIM)
    col_labels = ['Model', 'Params(M)', 'GMACs', 'Latency(ms)']
    for ds in datasets:
        col_labels.append(f'{ds}\nPSNR/SSIM')

    table_data = []
    for model in models:
        mr = [r for r in results if r['model'] == model]
        if not mr:
            continue
        row = [model, f'{mr[0]["params_M"]:.1f}', f'{mr[0]["gmacs"]:.1f}',
               f'{mr[0]["mean_ms"]:.1f}']
        for ds in datasets:
            match = [r for r in mr if r['dataset'] == ds]
            if match:
                row.append(f'{match[0]["psnr"]:.2f}/{match[0]["ssim"]:.4f}')
            else:
                row.append('-')
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(16, 2 + len(models) * 0.5))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2E86C1')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title('Baseline Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_table.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  Saved summary_table')


def main():
    print('Loading results...')
    results = load_all_results()
    if not results:
        print('No results found in', TABLES_DIR)
        return

    print(f'Found {len(results)} entries from {len(set(r["model"] for r in results))} models\n')
    print('Generating plots...')

    plot_bar_comparison(results, 'psnr', 'PSNR (dB)', 'PSNR Comparison', 'psnr_comparison')
    plot_bar_comparison(results, 'ssim', 'SSIM', 'SSIM Comparison', 'ssim_comparison')
    plot_scatter(results, 'gmacs', 'psnr', 'GMACs', 'Avg PSNR (dB)',
                 'Quality vs Compute', 'psnr_vs_gmacs')
    plot_scatter(results, 'params_M', 'psnr', 'Parameters (M)', 'Avg PSNR (dB)',
                 'Quality vs Model Size', 'psnr_vs_params')
    plot_scatter(results, 'mean_ms', 'psnr', 'Latency (ms)', 'Avg PSNR (dB)',
                 'Quality vs Latency', 'psnr_vs_latency')
    generate_summary_table(results)

    print(f'\nAll plots saved to {PLOTS_DIR}')


if __name__ == '__main__':
    main()
