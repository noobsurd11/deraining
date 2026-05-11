"""
Quantization comparison plots across Restormer / DRSformer / NAFNet-w32.

Reads:
  results/baselines/tables/{restormer,drsformer,nafnet_w32}_quantization.csv

Produces (PDF + PNG):
  results/baselines/plots/quantization_psnr_rain100h.{pdf,png}
  results/baselines/plots/quantization_psnr_vs_size.{pdf,png}
  results/baselines/plots/quantization_psnr_vs_latency_gpu.{pdf,png}
  results/baselines/plots/quantization_summary_table.{pdf,png}
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "baselines" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "baselines" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["restormer", "drsformer", "nafnet_w32"]
MODEL_LABELS = {"restormer": "Restormer", "drsformer": "DRSformer", "nafnet_w32": "NAFNet-w32"}
COLORS = {"restormer": "#2E86C1", "drsformer": "#1A5276", "nafnet_w32": "#27AE60"}
MARKERS = {"fp32": "o", "fp16": "s", "dynamic_int8": "D", "static_int8": "^"}
VARIANT_LABELS = {"fp32": "FP32", "fp16": "FP16",
                  "dynamic_int8": "Dyn-INT8", "static_int8": "Static-INT8"}
BAR_VARIANTS = ["fp32", "fp16", "static_int8"]
COMMON_DATASETS = ("Rain100H", "Rain100L", "Test100")


def load_results() -> list:
    rows = []
    for m in MODELS:
        csv_path = TABLES_DIR / f"{m}_quantization.csv"
        if not csv_path.exists():
            print(f"[WARN] missing {csv_path.name}")
            continue
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                r["model"] = m
                for k in ("psnr", "ssim", "lpips", "model_size_mb", "latency_ms"):
                    try:
                        r[k] = float(r.get(k, "nan"))
                    except (TypeError, ValueError):
                        r[k] = float("nan")
                rows.append(r)
    return rows


def get(rows, model, variant, dataset):
    for r in rows:
        if r["model"] == model and r["variant"] == variant and r["dataset"] == dataset:
            return r
    return None


def avg_psnr(rows, model, variant):
    vals = [get(rows, model, variant, d) for d in COMMON_DATASETS]
    psnrs = [r["psnr"] for r in vals if r and not np.isnan(r["psnr"])]
    return float(np.mean(psnrs)) if psnrs else float("nan")


# ---------------------------------------------------------------------------
# 1. Grouped bar: PSNR by model × variant on Rain100H
# ---------------------------------------------------------------------------
def plot_bar_rain100h(rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(MODELS))
    width = 0.8 / len(BAR_VARIANTS)
    hatches = {"fp32": "", "fp16": "//", "static_int8": "xx"}

    all_vals = []
    for i, var in enumerate(BAR_VARIANTS):
        vals = []
        for m in MODELS:
            r = get(rows, m, var, "Rain100H")
            vals.append(r["psnr"] if r and not np.isnan(r["psnr"]) else np.nan)
        offset = (i - (len(BAR_VARIANTS) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=[COLORS[m] for m in MODELS],
                      edgecolor="black", linewidth=0.7,
                      hatch=hatches[var], alpha=0.9)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
                all_vals.append(v)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rain100H PSNR by Model × Quantization Variant",
                 fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="white", edgecolor="black",
                            hatch=hatches[v], label=VARIANT_LABELS[v])
                      for v in BAR_VARIANTS]
    legend_handles += [Patch(facecolor=COLORS[m], label=MODEL_LABELS[m]) for m in MODELS]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    if all_vals:
        ax.set_ylim(bottom=min(all_vals) - 1.5)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(PLOTS_DIR / f"quantization_psnr_rain100h.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved quantization_psnr_rain100h")


# ---------------------------------------------------------------------------
# 2. Scatter: PSNR vs Model Size — every model × variant
# ---------------------------------------------------------------------------
def plot_scatter_size(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in MODELS:
        for var, mk in MARKERS.items():
            r = get(rows, m, var, "Rain100H")
            if r is None:
                continue
            size = r["model_size_mb"]
            psnr = avg_psnr(rows, m, var)
            if np.isnan(size) or np.isnan(psnr):
                continue
            ax.scatter(size, psnr, c=COLORS[m], marker=mk, s=170,
                       edgecolors="black", linewidth=0.6, zorder=5)
            ax.annotate(f"{MODEL_LABELS[m]} {VARIANT_LABELS[var]}",
                        (size, psnr), textcoords="offset points",
                        xytext=(9, 6), fontsize=8.5)
    ax.set_xlabel("Model Size (MB)", fontsize=12)
    ax.set_ylabel("Avg PSNR (dB) — Rain100H / Rain100L / Test100", fontsize=12)
    ax.set_title("Quality vs Model Size across Quantization Variants",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=COLORS[m], markeredgecolor="black",
                     markersize=10, label=MODEL_LABELS[m]) for m in MODELS]
    legend += [Line2D([0], [0], marker=mk, color="w",
                      markerfacecolor="gray", markeredgecolor="black",
                      markersize=10, label=VARIANT_LABELS[v])
               for v, mk in MARKERS.items()]
    ax.legend(handles=legend, fontsize=9, loc="best", ncol=2)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(PLOTS_DIR / f"quantization_psnr_vs_size.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved quantization_psnr_vs_size")


# ---------------------------------------------------------------------------
# 3. Scatter: PSNR vs GPU Latency — fp32 vs fp16 (CUDA only)
# ---------------------------------------------------------------------------
def plot_scatter_latency_gpu(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in MODELS:
        xs, ys = [], []
        for var in ("fp32", "fp16"):
            r = get(rows, m, var, "Rain100H")
            if r is None or r.get("device") != "cuda":
                continue
            lat = r["latency_ms"]
            psnr = avg_psnr(rows, m, var)
            if np.isnan(lat) or np.isnan(psnr):
                continue
            ax.scatter(lat, psnr, c=COLORS[m], marker=MARKERS[var], s=170,
                       edgecolors="black", linewidth=0.6, zorder=5)
            ax.annotate(f"{MODEL_LABELS[m]} {VARIANT_LABELS[var]}",
                        (lat, psnr), textcoords="offset points",
                        xytext=(9, 6), fontsize=9)
            xs.append(lat); ys.append(psnr)
        if len(xs) == 2:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order],
                    c=COLORS[m], linestyle="--", linewidth=1.0, alpha=0.6, zorder=3)

    ax.set_xlabel("GPU Latency (ms) @ 256×256", fontsize=12)
    ax.set_ylabel("Avg PSNR (dB) — Rain100H / Rain100L / Test100", fontsize=12)
    ax.set_title("GPU Latency vs Quality — FP32 vs FP16",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=COLORS[m], markeredgecolor="black",
                     markersize=10, label=MODEL_LABELS[m]) for m in MODELS]
    legend += [Line2D([0], [0], marker=MARKERS[v], color="w",
                      markerfacecolor="gray", markeredgecolor="black",
                      markersize=10, label=VARIANT_LABELS[v])
               for v in ("fp32", "fp16")]
    ax.legend(handles=legend, fontsize=9, loc="best")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(PLOTS_DIR / f"quantization_psnr_vs_latency_gpu.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved quantization_psnr_vs_latency_gpu")


# ---------------------------------------------------------------------------
# 4. Summary table as figure
# ---------------------------------------------------------------------------
def plot_summary_table(rows):
    col_labels = ["Model", "Variant", "Avg PSNR\n(3 sets)", "Δ vs FP32",
                  "Size (MB)", "Latency (ms)", "Device"]
    data = []
    for m in MODELS:
        fp32_psnr = avg_psnr(rows, m, "fp32")
        for var in ("fp32", "fp16", "dynamic_int8", "static_int8"):
            r = get(rows, m, var, "Rain100H")
            if r is None:
                continue
            psnr = avg_psnr(rows, m, var)
            delta = psnr - fp32_psnr if not (np.isnan(psnr) or np.isnan(fp32_psnr)) else float("nan")
            size = r["model_size_mb"]
            lat = r["latency_ms"]
            dev = r.get("device", "-") or "-"
            if var == "fp32":
                delta_str = "0.00"
            else:
                delta_str = f"{delta:+.2f}" if not np.isnan(delta) else "—"
            data.append([
                MODEL_LABELS[m],
                VARIANT_LABELS[var],
                f"{psnr:.2f}" if not np.isnan(psnr) else "—",
                delta_str,
                f"{size:.1f}" if not np.isnan(size) else "—",
                f"{lat:.1f}" if not np.isnan(lat) else "—",
                dev.upper(),
            ])

    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.5 * len(data)))
    ax.axis("off")
    table = ax.table(cellText=data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for j in range(len(col_labels)):
        c = table[0, j]
        c.set_facecolor("#2E86C1")
        c.set_text_props(color="white", fontweight="bold")
    model_lookup = {MODEL_LABELS[m]: COLORS[m] for m in MODELS}
    for i, row in enumerate(data, start=1):
        c = table[i, 0]
        c.set_facecolor(model_lookup[row[0]])
        c.set_text_props(color="white", fontweight="bold")

    plt.title("Quantization Summary — PSNR averaged over Rain100H / Rain100L / Test100",
              fontsize=13, fontweight="bold", pad=16)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(PLOTS_DIR / f"quantization_summary_table.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved quantization_summary_table")


def main():
    print("Loading quantization results...")
    rows = load_results()
    if not rows:
        print("No data — aborting.")
        return
    print(f"  {len(rows)} rows across {len(set(r['model'] for r in rows))} models\n")
    print("Generating plots...")
    plot_bar_rain100h(rows)
    plot_scatter_size(rows)
    plot_scatter_latency_gpu(rows)
    plot_summary_table(rows)
    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
