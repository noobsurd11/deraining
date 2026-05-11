"""
Verify that the pruning script's eval path and evaluate_model.py's eval path
produce bit-identical PSNR on fp32 checkpoints. If they do, the existing
pruning CSV numbers are trustworthy and the user's suspected gap was a
summary/label error, not a pipeline bug.

Runs two paths per model on first 5 Rain100H images:
  Path A — evaluate_model.py:60-66 style
  Path B — structured_pruning.py::eval_on_testset:212-217 style
Then full 100-image mean via Path B vs the baseline CSV's Rain100H value.
"""
import sys
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset
from evaluation.metrics import compute_psnr
from evaluation.model_wrappers import load_model

DEVICE = "cuda"
PRETRAINED = {
    "nafnet_w32": PROJECT_ROOT / "pretrained/nafnet_w32_deraining.pth",
    "restormer":  PROJECT_ROOT / "pretrained/restormer_deraining.pth",
    "drsformer":  PROJECT_ROOT / "pretrained/drsformer_deraining.pth",
}
RAIN100H = PROJECT_ROOT / "datasets/Rain13K/test/Rain100H"
BASELINE_CSV = PROJECT_ROOT / "results/baselines/tables"


def _purge_basicsr():
    for key in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
        del sys.modules[key]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def path_a_psnr(model, inp, gt):
    """evaluate_model.py:60-66 exact replication."""
    with torch.no_grad():
        pred = model(inp)
        pred = torch.clamp(pred, 0, 1)
    return compute_psnr(pred, gt)


@torch.no_grad()
def path_b_psnr(model, inp, gt):
    """structured_pruning.py::eval_on_testset:213-215 exact replication."""
    pred = torch.clamp(model(inp), 0, 1)
    return compute_psnr(pred, gt)


def baseline_rain100h(name):
    with open(BASELINE_CSV / f"{name}_results.csv") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "Rain100H":
                return float(row["psnr"])
    return None


def run_model(name):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    _purge_basicsr()
    model = load_model(name, str(PRETRAINED[name]), DEVICE)
    # load_model already sets eval, but be explicit:
    model.eval()

    ds = DerainDataset(str(RAIN100H / "input"), str(RAIN100H / "target"))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    a_vals, b_vals = [], []
    print(f"{'idx':>3}  {'filename':<30}  {'path_A':>10}  {'path_B':>10}  {'diff':>10}")
    print("-" * 75)
    for i, (inp, gt, fname) in enumerate(loader):
        inp, gt = inp.to(DEVICE), gt.to(DEVICE)
        pa = path_a_psnr(model, inp, gt)
        pb = path_b_psnr(model, inp, gt)
        a_vals.append(pa)
        b_vals.append(pb)
        if i < 5:
            print(f"{i:>3}  {fname[0]:<30}  {pa:>10.4f}  {pb:>10.4f}  {pa - pb:>+10.2e}")

    mean_a = sum(a_vals) / len(a_vals)
    mean_b = sum(b_vals) / len(b_vals)
    baseline = baseline_rain100h(name)
    max_diff = max(abs(a - b) for a, b in zip(a_vals, b_vals))

    print("-" * 75)
    print(f"{'MEAN':>3}  {'(100 images)':<30}  {mean_a:>10.4f}  {mean_b:>10.4f}  "
          f"{mean_a - mean_b:>+10.2e}")
    print(f"  baseline CSV Rain100H PSNR: {baseline:.4f}")
    print(f"  |Path_B mean - baseline| = {abs(mean_b - baseline):.4f} dB")
    print(f"  max per-image |A - B|    = {max_diff:.2e} dB")

    pass_a_eq_b = max_diff < 1e-4
    pass_b_eq_baseline = abs(mean_b - baseline) < 0.01
    if pass_a_eq_b and pass_b_eq_baseline:
        print(f"  RESULT: OK — pipelines identical and match baseline")
    else:
        print(f"  RESULT: FAIL — A==B: {pass_a_eq_b}, B==baseline: {pass_b_eq_baseline}")

    del model
    torch.cuda.empty_cache()


def main():
    for name in ("nafnet_w32", "restormer", "drsformer"):
        run_model(name)


if __name__ == "__main__":
    main()
