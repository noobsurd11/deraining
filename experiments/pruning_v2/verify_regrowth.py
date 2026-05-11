"""
Verify the hypothesis: weights regrow during fine-tuning because
prune.remove() is called before fine-tuning, removing the mask.

Loads three checkpoints and counts nonzero Conv2d/Linear weights.
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.model_wrappers import load_model

DEVICE = "cpu"

CHECKPOINTS = {
    "After pruning, before FT": PROJECT_ROOT / "experiments/pruning/checkpoints/nafnet_w32_pruned_0.3.pth",
    "After pruning + FT": PROJECT_ROOT / "experiments/pruning/checkpoints/nafnet_w32_pruned_0.3_finetuned.pth",
    "Full pipeline (KD+prune+FT)": PROJECT_ROOT / "experiments/full_pipeline/checkpoints/nafnet_kd_pruned30_ft.pth",
}


def count_nonzero(model):
    total = 0
    nonzero = 0
    for name, p in model.named_parameters():
        if 'weight' in name and p.dim() >= 2:
            total += p.numel()
            nonzero += (p != 0).sum().item()
    sparsity = 100.0 * (1 - nonzero / total)
    return total, nonzero, sparsity


def _purge_basicsr():
    for key in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
        del sys.modules[key]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def main():
    print("=" * 70)
    print("  VERIFYING REGROWTH HYPOTHESIS")
    print("=" * 70)

    results = {}
    for label, ckpt in CHECKPOINTS.items():
        if not ckpt.exists():
            print(f"\n  SKIP: {ckpt.name} not found")
            continue
        _purge_basicsr()
        model = load_model("nafnet_w32", str(ckpt), DEVICE)
        total, nonzero, sparsity = count_nonzero(model)
        results[label] = (total, nonzero, sparsity)
        print(f"\n  {label}:")
        print(f"    {nonzero/1e6:.2f} M nonzero out of {total/1e6:.2f} M total ({sparsity:.1f}% sparse)")
        del model

    print("\n" + "=" * 70)
    labels = list(results.keys())
    if len(labels) >= 2:
        _, _, s_before = results[labels[0]]
        _, _, s_after = results[labels[1]]
        if s_after < s_before - 1.0:
            print(f"  CONFIRMED: Sparsity dropped from {s_before:.1f}% to {s_after:.1f}%"
                  f" -- weights regrew during FT")
        else:
            print(f"  DISPROVED: Sparsity maintained at {s_after:.1f}%"
                  f" -- hypothesis was wrong")
    print("=" * 70)


if __name__ == "__main__":
    main()
