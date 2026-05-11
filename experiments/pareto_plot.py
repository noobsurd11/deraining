"""
Master Pareto plots from results/baselines/tables/combined_compression.csv.

Produces three figures (.pdf + .png at dpi=300, bbox_inches='tight'),
saved to results/baselines/plots/:

  1. pareto_psnr_vs_size     - PSNR (Rain100H) vs Model Size (MB)
                               Every row plotted; technique name as label;
                               color by base model; marker by technique kind;
                               dashed Pareto frontier connecting non-dominated
                               points (max PSNR for any given size).
  2. pareto_psnr_vs_latency  - PSNR (Rain100H) vs GPU Latency (ms).
                               GPU-only rows (fp32, fp16, pruned, KD-fp16);
                               CPU INT8 rows excluded.
  3. pareto_summary_table    - figure-only table with columns
                               [Model+Technique, Rain100H, Rain100L, Size(MB),
                                Latency(ms), ΔPSNR vs Restormer FP32]
                               sorted by model_size_mb ascending.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "baselines" / "tables" / "combined_compression.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "baselines" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "restormer":  "#2E86C1",
    "drsformer":  "#1A5276",
    "nafnet_w32": "#27AE60",
    "nafnet_kd":  "#27AE60",  # NAFNet family — same hue as base
}

# Marker by *technique kind*, not exact technique name.
def technique_marker(tech: str) -> str:
    t = tech.lower()
    if "kd" in t and ("fp16" in t or "int8" in t):
        return "p"   # pentagon — KD + quantized
    if t.endswith("kd") or t == "nafnet_kd":
        return "*"   # star — KD only
    if "pruned" in t:
        return "D"   # diamond — pruned (and pruned+fp16)
    if "int8" in t:
        return "^"   # triangle — INT8
    if "fp16" in t:
        return "s"   # square — FP16
    return "o"       # circle — FP32 baseline


def technique_kind(tech: str) -> str:
    """Short label for the legend grouping by marker."""
    t = tech.lower()
    if "kd" in t and ("fp16" in t or "int8" in t):
        return "KD + quantized (pentagon)"
    if t.endswith("kd") or t == "nafnet_kd":
        return "KD (star)"
    if "pruned" in t:
        return "pruned (diamond)"
    if "int8" in t:
        return "static INT8 (triangle)"
    if "fp16" in t:
        return "FP16 (square)"
    return "FP32 (circle)"


def base_model(tech: str) -> str:
    t = tech.lower()
    if t.startswith("restormer"):
        return "restormer"
    if t.startswith("drsformer"):
        return "drsformer"
    if t.startswith("nafnet_kd") or t == "nafnet_kd":
        return "nafnet_kd"
    if t.startswith("nafnet"):
        return "nafnet_w32"
    return "?"


def load_rows() -> list[dict]:
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("psnr_rain100h", "psnr_rain100l", "params_M",
                      "model_size_mb", "latency_ms"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "n/a") else float("nan")
            rows.append(r)
    return rows


def is_gpu_row(tech: str) -> bool:
    """True for rows that ran on GPU. CPU-only static INT8 is excluded
    from the latency plot."""
    t = tech.lower()
    if "int8" in t:
        return False
    return True


def pareto_frontier(points):
    """Given a list of (x, y) where lower x is better and higher y is better,
    return the subset that lies on the Pareto frontier, sorted by x."""
    pts = [p for p in points if not (math.isnan(p[0]) or math.isnan(p[1]))]
    if not pts:
        return []
    pts.sort(key=lambda p: (p[0], -p[1]))
    frontier = []
    best_y = -math.inf
    for x, y in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def _annotate(ax, x, y, text, dx=8, dy=8):
    if math.isnan(x) or math.isnan(y):
        return
    ax.annotate(text, (x, y), textcoords="offset points",
                xytext=(dx, dy), fontsize=8)


def plot_psnr_vs_x(rows, x_key, xlabel, title, filename, gpu_only=False):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    seen_kinds = {}
    seen_models = {}
    used_rows = []
    for r in rows:
        tech = r["technique"]
        if gpu_only and not is_gpu_row(tech):
            continue
        x = r[x_key]
        y = r["psnr_rain100h"]
        if math.isnan(x) or math.isnan(y):
            continue
        used_rows.append(r)
        bm = base_model(tech)
        color = MODEL_COLORS.get(bm, "#888888")
        marker = technique_marker(tech)
        ax.scatter(x, y, c=color, marker=marker, s=140,
                   edgecolors="black", linewidth=0.6, zorder=5)
        _annotate(ax, x, y, tech)
        seen_kinds.setdefault(technique_kind(tech), marker)
        seen_models.setdefault(bm, color)

    # Pareto frontier on (x, y) where smaller x is better.
    pts = [(r[x_key], r["psnr_rain100h"]) for r in used_rows]
    front = pareto_frontier(pts)
    if len(front) >= 2:
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        ax.plot(fx, fy, "--", color="#444444", linewidth=1.4, alpha=0.8,
                label="Pareto frontier", zorder=4)

    # Legends — model colors, technique markers.
    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="black",
                   markersize=10, label=name)
        for name, c in seen_models.items()
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=m, color="w",
                   markerfacecolor="#888", markeredgecolor="black",
                   markersize=10, label=name)
        for name, m in seen_kinds.items()
    ]
    legend1 = ax.legend(handles=color_handles, title="Base model",
                        loc="lower right", fontsize=9, title_fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, title="Technique",
              loc="lower left", fontsize=9, title_fontsize=9)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("PSNR on Rain100H (dB)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {filename}.pdf / {filename}.png")


def plot_summary_table(rows: list[dict]) -> None:
    # Sort by model size ascending; place rows missing size at the end.
    rows = sorted(
        rows,
        key=lambda r: (math.isnan(r["model_size_mb"]), r["model_size_mb"]),
    )
    # ΔPSNR vs Restormer FP32 reference.
    ref = next((r for r in rows if r["technique"] == "restormer_fp32"), None)
    ref_psnr = ref["psnr_rain100h"] if ref else float("nan")

    col_labels = ["Model+Technique", "Rain100H", "Rain100L",
                  "Size (MB)", "Latency (ms)", "ΔPSNR vs Restormer FP32"]

    table_data = []
    for r in rows:
        d_psnr = r["psnr_rain100h"] - ref_psnr if not math.isnan(ref_psnr) else float("nan")
        table_data.append([
            r["technique"],
            f"{r['psnr_rain100h']:.2f}" if not math.isnan(r["psnr_rain100h"]) else "—",
            f"{r['psnr_rain100l']:.2f}" if not math.isnan(r["psnr_rain100l"]) else "—",
            f"{r['model_size_mb']:.2f}" if not math.isnan(r["model_size_mb"]) else "—",
            f"{r['latency_ms']:.2f}" if not math.isnan(r["latency_ms"]) else "—",
            f"{d_psnr:+.2f}" if not math.isnan(d_psnr) else "—",
        ])

    fig_h = 1.5 + len(table_data) * 0.36
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Header style.
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2E86C1")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color the first cell of each data row by base model.
    for i, r in enumerate(rows, start=1):
        bm = base_model(r["technique"])
        c = MODEL_COLORS.get(bm, "#888888")
        table[i, 0].set_facecolor(c)
        table[i, 0].set_text_props(color="white", fontweight="bold")

    plt.title("Compressed Deraining Models — Master Summary",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"pareto_summary_table.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved pareto_summary_table.pdf / pareto_summary_table.png")


def main() -> None:
    print(f"Reading {CSV_PATH}")
    rows = load_rows()
    print(f"  {len(rows)} rows loaded")

    plot_psnr_vs_x(
        rows,
        x_key="model_size_mb",
        xlabel="Model size on disk (MB)",
        title="Quality-Size Pareto Frontier for Compressed Deraining Models",
        filename="pareto_psnr_vs_size",
    )
    plot_psnr_vs_x(
        rows,
        x_key="latency_ms",
        xlabel="GPU latency on 1×3×256×256 (ms)",
        title="Quality-Latency Tradeoff",
        filename="pareto_psnr_vs_latency",
        gpu_only=True,
    )
    plot_summary_table(rows)
    print("done.")


if __name__ == "__main__":
    main()
