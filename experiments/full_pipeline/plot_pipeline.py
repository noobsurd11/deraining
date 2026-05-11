"""
Waterfall plot for the full compression pipeline.

Reads:  results/baselines/tables/full_pipeline.csv
Writes: results/baselines/plots/full_pipeline_waterfall.{pdf,png}

Each compression stage is shown as a bar; bar color shifts from green
(less compressed) → orange → red (most compressed). A dashed horizontal
line marks the Restormer FP32 reference.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "results" / "baselines" / "tables" / "full_pipeline.csv"
PLOT_DIR = PROJECT_ROOT / "results" / "baselines" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RESTORMER_FP32_PSNR = 31.48

# Stage order + display labels. Color gradient: green → orange → red.
STAGES = [
    ("nafnet_kd_baseline",          "KD baseline",      "#1E8449"),
    ("nafnet_kd_pruned30",          "Pruned 30%\n(no FT)", "#27AE60"),
    ("nafnet_kd_pruned30_ft",       "Pruned + FT",      "#F39C12"),
    ("nafnet_kd_pruned30_ft_fp16",  "Pruned + FT\n+ FP16", "#E67E22"),
    ("nafnet_kd_pruned30_ft_int8",  "Pruned + FT\n+ INT8", "#E74C3C"),
]


def load_csv(path: Path) -> dict:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if k == "stage":
                    continue
                try:
                    r[k] = float(v) if v not in ("", None) else float("nan")
                except ValueError:
                    r[k] = float("nan")
            out[r["stage"]] = r
    return out


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: missing {CSV_PATH}")
        return
    rows = load_csv(CSV_PATH)
    labels, vals, colors, sizes, lats = [], [], [], [], []
    for key, lbl, color in STAGES:
        if key not in rows:
            continue
        r = rows[key]
        v = r.get("psnr_rain100h", float("nan"))
        if math.isnan(v):
            continue
        labels.append(lbl)
        vals.append(v)
        colors.append(color)
        sizes.append(r.get("model_size_mb", float("nan")))
        lats.append(r.get("gpu_latency_ms", float("nan")))

    if not vals:
        print("ERROR: no usable stages in CSV.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

    # Value labels on top of each bar.
    for i, (bar, v, sz, lt) in enumerate(zip(bars, vals, sizes, lats)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08,
                f"{v:.2f} dB", ha="center", va="bottom", fontsize=10, fontweight="bold")
        sub = ""
        if not math.isnan(sz):
            sub += f"{sz:.1f} MB"
        if not math.isnan(lt):
            sub += f"  {lt:.1f} ms" if sub else f"{lt:.1f} ms"
        if sub:
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.6,
                    sub, ha="center", va="top", fontsize=8, color="white",
                    fontweight="bold")

    ax.axhline(RESTORMER_FP32_PSNR, color="#2E86C1", linestyle="--", linewidth=1.4,
               label=f"Restormer FP32 ({RESTORMER_FP32_PSNR:.2f} dB)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("PSNR on Rain100H (dB)", fontsize=12)
    ax.set_title("Full compression pipeline: NAFNet-KD → Prune → FT → FP16 → INT8",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    # Headroom for value labels.
    finite_vals = [v for v in vals if not math.isnan(v)]
    if finite_vals:
        ymin = min(min(finite_vals), RESTORMER_FP32_PSNR) - 1.5
        ymax = max(max(finite_vals), RESTORMER_FP32_PSNR) + 0.8
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"full_pipeline_waterfall.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PLOT_DIR / 'full_pipeline_waterfall.{pdf,png}'}")


if __name__ == "__main__":
    main()
