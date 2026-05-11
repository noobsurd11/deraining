"""
Three figures from the pruning experiments, all saved to
results/baselines/plots/ as PDF + PNG:

1. pruning_psnr_vs_ratio            (Rain100H only, line chart)
2. pruning_recovery_bars            (Rain100H only, grouped bars)
3. pruning_psnr_vs_nonzero_params   (scatter + fp32 star)
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES = PROJECT_ROOT / "results/baselines/tables"
PLOTS = PROJECT_ROOT / "results/baselines/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODELS = ["restormer", "drsformer", "nafnet_w32"]
LABEL = {"restormer": "Restormer", "drsformer": "DRSformer", "nafnet_w32": "NAFNet-w32"}
COLORS = {"restormer": "#2E86C1", "drsformer": "#1A5276", "nafnet_w32": "#27AE60"}
RATIOS = [0.3, 0.5, 0.7]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 140,
    "savefig.dpi": 200,
})


def _read(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_pruning():
    rows = {}
    for m in MODELS:
        rows[m] = _read(TABLES / f"{m}_pruning.csv")
    return rows


def load_baselines():
    """fp32 anchor: PSNR on Rain100H/L and total params."""
    anchors = {}
    for m in MODELS:
        for r in _read(TABLES / f"{m}_results.csv"):
            ds = r["dataset"]
            if ds in ("Rain100H", "Rain100L"):
                anchors.setdefault(m, {})[ds] = {
                    "psnr": float(r["psnr"]),
                    "params_M": float(r["params_M"]),
                }
    return anchors


def _by(rows, ratio, stage, dataset):
    for r in rows:
        if (abs(float(r["prune_ratio"]) - ratio) < 1e-6
                and r["stage"] == stage and r["dataset"] == dataset):
            return r
    return None


def plot_psnr_vs_ratio(prun, anchors, out):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    xs = [0.0] + RATIOS
    for m in MODELS:
        rows = prun.get(m, [])
        if not rows or m not in anchors:
            continue
        base = anchors[m]["Rain100H"]["psnr"]
        y_only = [base] + [
            (float(_by(rows, r, "prune_only", "Rain100H")["psnr"])
             if _by(rows, r, "prune_only", "Rain100H") else np.nan)
            for r in RATIOS
        ]
        y_ft = [base] + [
            (float(_by(rows, r, "prune_ft", "Rain100H")["psnr"])
             if _by(rows, r, "prune_ft", "Rain100H") else np.nan)
            for r in RATIOS
        ]
        c = COLORS[m]
        ax.plot(xs, y_only, "--", marker="o", color=c, lw=1.8, ms=7,
                label=f"{LABEL[m]} (prune-only)", alpha=0.75)
        ax.plot(xs, y_ft, "-", marker="^", color=c, lw=2.0, ms=8,
                label=f"{LABEL[m]} (+ fine-tune)")
    ax.set_xlabel("Pruning ratio (fraction of channels zeroed per layer)")
    ax.set_ylabel("Rain100H PSNR (dB)")
    ax.set_title("Structured channel pruning — quality vs. ratio")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.1f}" for x in xs])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_recovery_bars(prun, out):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    w = 0.25
    x = np.arange(len(RATIOS))
    for i, m in enumerate(MODELS):
        rows = prun.get(m, [])
        deltas = []
        for r in RATIOS:
            a = _by(rows, r, "prune_only", "Rain100H")
            b = _by(rows, r, "prune_ft", "Rain100H")
            deltas.append(
                float(b["psnr"]) - float(a["psnr"]) if (a and b) else np.nan
            )
        ax.bar(x + (i - 1) * w, deltas, w, color=COLORS[m],
               label=LABEL[m], edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.1f}" for r in RATIOS])
    ax.set_xlabel("Pruning ratio")
    ax.set_ylabel("Δ PSNR (fine-tuned − prune-only), Rain100H (dB)")
    ax.set_title("Fine-tune recovery vs. pruning ratio")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_psnr_vs_nonzero(prun, anchors, out):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for m in MODELS:
        rows = prun.get(m, [])
        c = COLORS[m]
        for stage, marker in [("prune_only", "o"), ("prune_ft", "^")]:
            xs, ys, labels = [], [], []
            for r in RATIOS:
                row = _by(rows, r, stage, "Rain100H")
                if row is None:
                    continue
                xs.append(float(row["nonzero_params_M"]))
                ys.append(float(row["psnr"]))
                labels.append(f"{r:.1f}")
            if xs:
                ax.scatter(xs, ys, s=80, marker=marker, color=c,
                           edgecolor="black", lw=0.6,
                           label=f"{LABEL[m]} ({stage.replace('_', '-')})",
                           zorder=3)
                if stage == "prune_ft":
                    for xi, yi, li in zip(xs, ys, labels):
                        ax.annotate(li, (xi, yi), textcoords="offset points",
                                    xytext=(6, 4), fontsize=8, color=c)
        # fp32 baseline anchor
        if m in anchors and "Rain100H" in anchors[m]:
            a = anchors[m]["Rain100H"]
            ax.scatter([a["params_M"]], [a["psnr"]], marker="*",
                       color=c, edgecolor="black", lw=0.8, s=260,
                       label=f"{LABEL[m]} (fp32)", zorder=4)
    ax.set_xlabel("Non-zero parameters (M)")
    ax.set_ylabel("Rain100H PSNR (dB)")
    ax.set_title("Quality vs. effective capacity after pruning")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def main():
    prun = load_pruning()
    anchors = load_baselines()
    have = {m: len(v) for m, v in prun.items()}
    print("rows per model:", have)
    plot_psnr_vs_ratio(prun, anchors, PLOTS / "pruning_psnr_vs_ratio")
    plot_recovery_bars(prun, PLOTS / "pruning_recovery_bars")
    plot_psnr_vs_nonzero(prun, anchors, PLOTS / "pruning_psnr_vs_nonzero_params")
    print(f"plots saved in {PLOTS}")


if __name__ == "__main__":
    main()
