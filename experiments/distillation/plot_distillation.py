"""
Plot distillation training artifacts.

Produces 4 figures (each as .pdf + .png, dpi=300) under
results/baselines/plots/:

  1. distillation_loss_curve    - total loss vs iteration (log y)
  2. distillation_val_psnr      - validation PSNR vs iteration, with dashed
                                  baselines for NAFNet (30.23) and Restormer (31.48)
  3. distillation_testset_bars  - grouped bars: Restormer / NAFNet / NAFNet-KD
                                  on the 5 test sets
  4. distillation_summary_table - Model / Params / GMACs / Latency /
                                  Rain100H / Rain100L table figure

Reads:
  experiments/distillation/train.log
  results/baselines/tables/{nafnet_w32_results,restormer_results,nafnet_w32_kd_results}.csv
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FILE = PROJECT_ROOT / "experiments/distillation/train.log"
TABLES_DIR = PROJECT_ROOT / "results/baselines/tables"
PLOTS_DIR = PROJECT_ROOT / "results/baselines/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NAFNET_BASELINE = 30.23
RESTORMER_BASELINE = 31.48

COLORS = {
    "restormer": "#2E86C1",
    "nafnet_w32": "#27AE60",
    "nafnet_w32_kd": "#E67E22",
}
LABELS = {
    "restormer": "Restormer (teacher)",
    "nafnet_w32": "NAFNet-w32 (baseline)",
    "nafnet_w32_kd": "NAFNet-w32 (KD)",
}

DATASETS = ["Rain100H", "Rain100L", "Test100", "Test1200", "Test2800"]

METRIC_RE = re.compile(
    r"METRIC iter=(\d+) total=([-\d.eE+]+) l_pix=([-\d.eE+]+) "
    r"l_distill=([-\d.eE+]+) lr=([-\d.eE+]+)"
)
VAL_RE = re.compile(r"VAL iter=(\d+) psnr=([-\d.eE+]+)")


def parse_log(path: Path):
    iters, total, l_pix, l_distill, lrs = [], [], [], [], []
    v_iters, v_psnr = [], []
    if not path.exists():
        return (iters, total, l_pix, l_distill, lrs), (v_iters, v_psnr)
    with open(path) as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                iters.append(int(m.group(1)))
                total.append(float(m.group(2)))
                l_pix.append(float(m.group(3)))
                l_distill.append(float(m.group(4)))
                lrs.append(float(m.group(5)))
                continue
            m = VAL_RE.search(line)
            if m:
                v_iters.append(int(m.group(1)))
                v_psnr.append(float(m.group(2)))
    return (iters, total, l_pix, l_distill, lrs), (v_iters, v_psnr)


def read_csv_by_dataset(path: Path) -> dict:
    by_ds = {}
    if not path.exists():
        return by_ds
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_ds[row["dataset"]] = row
    return by_ds


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.pdf / {stem}.png")


def plot_loss(train_stats) -> None:
    iters, total, l_pix, l_distill, _ = train_stats
    if not iters:
        print("  [loss] no log lines found, skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, total, color=COLORS["nafnet_w32_kd"], linewidth=1.4,
            label="total loss")
    ax.plot(iters, l_pix, color="#888", linewidth=0.8, alpha=0.7, label="pixel L1")
    ax.plot(iters, l_distill, color="#BBB", linewidth=0.8, alpha=0.7,
            label="distill L1")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L1 loss (log)", fontsize=12)
    ax.set_title("Distillation training loss", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, "distillation_loss_curve")


def plot_val_psnr(val_stats) -> None:
    v_iters, v_psnr = val_stats
    fig, ax = plt.subplots(figsize=(8, 5))
    if v_iters:
        ax.plot(v_iters, v_psnr, color=COLORS["nafnet_w32_kd"],
                linewidth=1.8, marker="o", markersize=5,
                label="NAFNet-KD (EMA)")
    ax.axhline(NAFNET_BASELINE, color=COLORS["nafnet_w32"], linestyle="--",
               linewidth=1.2, label=f"NAFNet baseline ({NAFNET_BASELINE:.2f} dB)")
    ax.axhline(RESTORMER_BASELINE, color=COLORS["restormer"], linestyle="--",
               linewidth=1.2, label=f"Restormer ({RESTORMER_BASELINE:.2f} dB)")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Rain100H PSNR (dB)", fontsize=12)
    ax.set_title("Validation PSNR (EMA student) vs training iteration",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    _save(fig, "distillation_val_psnr")


def plot_testset_bars() -> None:
    baseline = read_csv_by_dataset(TABLES_DIR / "nafnet_w32_results.csv")
    teacher = read_csv_by_dataset(TABLES_DIR / "restormer_results.csv")
    kd = read_csv_by_dataset(TABLES_DIR / "nafnet_w32_kd_results.csv")
    if not kd:
        print("  [bars] no nafnet_w32_kd_results.csv yet, skipping")
        return

    x = np.arange(len(DATASETS))
    width = 0.26

    def series(by_ds):
        return [float(by_ds.get(d, {}).get("psnr", "nan") or "nan") for d in DATASETS]

    restormer_vals = series(teacher)
    baseline_vals = series(baseline)
    kd_vals = series(kd)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width, restormer_vals, width, label=LABELS["restormer"],
                color=COLORS["restormer"], edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, baseline_vals, width, label=LABELS["nafnet_w32"],
                color=COLORS["nafnet_w32"], edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + width, kd_vals, width, label=LABELS["nafnet_w32_kd"],
                color=COLORS["nafnet_w32_kd"], edgecolor="white", linewidth=0.5)

    for bars in (b1, b2, b3):
        for rect in bars:
            v = rect.get_height()
            if np.isnan(v):
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("PSNR across test sets: teacher vs student vs distilled student",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    # Give headroom for value labels.
    vmax = max(v for v in restormer_vals + baseline_vals + kd_vals if not np.isnan(v))
    ax.set_ylim(bottom=min(baseline_vals) - 1.0, top=vmax + 1.2)
    plt.tight_layout()
    _save(fig, "distillation_testset_bars")


def plot_summary_table() -> None:
    baseline = read_csv_by_dataset(TABLES_DIR / "nafnet_w32_results.csv")
    teacher = read_csv_by_dataset(TABLES_DIR / "restormer_results.csv")
    kd = read_csv_by_dataset(TABLES_DIR / "nafnet_w32_kd_results.csv")
    if not kd:
        print("  [table] no nafnet_w32_kd_results.csv yet, skipping")
        return

    def pick(by_ds, key, ds="Rain100H"):
        v = by_ds.get(ds, {}).get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    rows = []
    for name, src in (("Restormer (teacher)", teacher),
                      ("NAFNet-w32 (baseline)", baseline),
                      ("NAFNet-w32 (KD)", kd)):
        rows.append([
            name,
            f"{pick(src, 'params_M'):.2f}",
            f"{pick(src, 'gmacs'):.2f}",
            f"{pick(src, 'mean_ms'):.2f}",
            f"{pick(src, 'psnr', 'Rain100H'):.2f}",
            f"{pick(src, 'psnr', 'Rain100L'):.2f}",
        ])

    col_labels = ["Model", "Params (M)", "GMACs", "Latency (ms)",
                  "Rain100H (dB)", "Rain100L (dB)"]

    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    row_colors = [COLORS["restormer"], COLORS["nafnet_w32"], COLORS["nafnet_w32_kd"]]
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2E86C1")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, color in enumerate(row_colors, start=1):
        table[i, 0].set_facecolor(color)
        table[i, 0].set_text_props(color="white", fontweight="bold")

    plt.title("Distillation summary: Restormer / NAFNet / NAFNet-KD",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "distillation_summary_table")


def main() -> None:
    print(f"Reading log: {LOG_FILE}")
    train_stats, val_stats = parse_log(LOG_FILE)
    print(f"  parsed {len(train_stats[0])} METRIC and {len(val_stats[0])} VAL entries")
    print(f"Writing plots to: {PLOTS_DIR}")
    plot_loss(train_stats)
    plot_val_psnr(val_stats)
    plot_testset_bars()
    plot_summary_table()
    print("done.")


if __name__ == "__main__":
    main()
