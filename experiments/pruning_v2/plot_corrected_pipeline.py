"""
Generate plots and summary for the corrected pruning pipeline.

Reads experiments/pruning_v2/results.csv and produces:
  1. Waterfall chart: PSNR at each stage
  2. Scatter: PSNR vs model size
  3. Scatter: PSNR vs latency
  4. Bar chart: GMACs comparison
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK = PROJECT_ROOT / "experiments" / "pruning_v2"
CSV_PATH = WORK / "results.csv"
PLOT_DIR = PROJECT_ROOT / "results" / "baselines" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "restormer": "#2E86C1",
    "nafnet_kd_w32": "#27AE60",
    "broken": "#E67E22",
    "physical": "#E74C3C",
    "physical_fp16": "#C0392B",
    "physical_onnx": "#922B21",
}


def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                if k != "stage":
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        row[k] = float("nan")
            rows.append(row)
    return rows


def get_color(stage):
    if "restormer" in stage:
        return COLORS["restormer"]
    if "broken" in stage:
        return COLORS["broken"]
    if "onnx" in stage:
        return COLORS["physical_onnx"]
    if "fp16" in stage:
        return COLORS["physical_fp16"]
    if "w22" in stage or "physical" in stage:
        return COLORS["physical"]
    return COLORS["nafnet_kd_w32"]


def plot_waterfall(rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    stages = [r["stage"] for r in rows]
    psnrs = [r["psnr_rain100h"] for r in rows]
    colors = [get_color(s) for s in stages]

    x = np.arange(len(stages))
    bars = ax.bar(x, psnrs, color=colors, edgecolor="white", linewidth=0.5)

    # Restormer reference line
    ax.axhline(y=31.48, color=COLORS["restormer"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Restormer (31.48 dB)")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("nafnet_kd_", "").replace("_", "\n")
                        for s in stages], fontsize=8, rotation=0)
    ax.set_ylabel("PSNR (dB) on Rain100H")
    ax.set_title("Corrected Compression Pipeline: PSNR at Each Stage")
    ax.legend(loc="lower right")

    for bar, val in zip(bars, psnrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(min(psnrs) - 2, max(psnrs) + 2)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"corrected_pipeline_waterfall.{ext}", dpi=300)
    plt.close(fig)


def plot_psnr_vs_size(rows):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        s = r["stage"]
        ax.scatter(r["model_size_mb"], r["psnr_rain100h"],
                   c=get_color(s), s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(s.replace("nafnet_kd_", "").replace("_", " "),
                    (r["model_size_mb"], r["psnr_rain100h"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7)

    ax.axhline(y=31.48, color=COLORS["restormer"], linestyle="--",
               linewidth=1, alpha=0.5)
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("PSNR (dB) on Rain100H")
    ax.set_title("PSNR vs Actual Model Size")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"corrected_pipeline_psnr_vs_size.{ext}", dpi=300)
    plt.close(fig)


def plot_psnr_vs_latency(rows):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        s = r["stage"]
        lat = r["gpu_latency_ms"]
        if np.isnan(lat):
            continue
        ax.scatter(lat, r["psnr_rain100h"],
                   c=get_color(s), s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(s.replace("nafnet_kd_", "").replace("_", " "),
                    (lat, r["psnr_rain100h"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7)

    ax.axhline(y=31.48, color=COLORS["restormer"], linestyle="--",
               linewidth=1, alpha=0.5)
    ax.set_xlabel("GPU Latency (ms)")
    ax.set_ylabel("PSNR (dB) on Rain100H")
    ax.set_title("PSNR vs GPU Latency")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"corrected_pipeline_psnr_vs_latency.{ext}", dpi=300)
    plt.close(fig)


def plot_gmacs_bars(rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    stages = [r["stage"] for r in rows]
    gmacs = [r["gmacs"] for r in rows]
    colors = [get_color(s) for s in stages]

    x = np.arange(len(stages))
    bars = ax.bar(x, gmacs, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("nafnet_kd_", "").replace("_", "\n")
                        for s in stages], fontsize=8)
    ax.set_ylabel("GMACs (256x256 input)")
    ax.set_title("Compute Cost Comparison")

    for bar, val in zip(bars, gmacs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"corrected_pipeline_gmacs.{ext}", dpi=300)
    plt.close(fig)


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run pruned_finetune_physical.py first.")
        sys.exit(1)

    rows = load_csv()
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    plot_waterfall(rows)
    print("  Saved: corrected_pipeline_waterfall.{pdf,png}")

    plot_psnr_vs_size(rows)
    print("  Saved: corrected_pipeline_psnr_vs_size.{pdf,png}")

    plot_psnr_vs_latency(rows)
    print("  Saved: corrected_pipeline_psnr_vs_latency.{pdf,png}")

    plot_gmacs_bars(rows)
    print("  Saved: corrected_pipeline_gmacs.{pdf,png}")

    # Print final table
    print("\n" + "=" * 90)
    print(f"  {'Stage':<42} {'PSNR-H':>8} {'Size(MB)':>8} {'GMACs':>8} {'Lat(ms)':>8} {'Params(M)':>9}")
    print("  " + "-" * 85)
    for r in rows:
        print(f"  {r['stage']:<42} {r['psnr_rain100h']:>8.2f} {r['model_size_mb']:>8.1f} "
              f"{r['gmacs']:>8.1f} {r['gpu_latency_ms']:>8.1f} {r['total_params_M']:>9.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
