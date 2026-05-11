"""
Aggregate results across compression techniques into a single CSV for the
master Pareto plot.

Pulls existing rows from:
  results/baselines/tables/{restormer,nafnet_w32,drsformer}_results.csv     (fp32)
  results/baselines/tables/{restormer,nafnet_w32,drsformer}_quantization.csv
  results/baselines/tables/{restormer,nafnet_w32}_pruning.csv               (prune_ft @ 0.3)
  results/baselines/tables/nafnet_w32_kd_results.csv
  results/baselines/tables/nafnet_w32_kd_quantization.csv

Runs NEW evals only for combinations not yet measured:
  - restormer_pruned30_ft_fp16  (load 30%-pruned finetuned ckpt, .half(), GPU)
  - nafnet_pruned30_ft_fp16     (same)

Writes results/baselines/tables/combined_compression.csv with columns:
  model, technique, psnr_rain100h, psnr_rain100l, params_M, model_size_mb, latency_ms

The fixed row set produced (in order):
  restormer_fp32, restormer_fp16, restormer_static_int8,
  restormer_pruned30_ft, restormer_pruned30_ft_fp16,
  nafnet_fp32, nafnet_fp16, nafnet_static_int8,
  nafnet_pruned30_ft, nafnet_pruned30_ft_fp16,
  nafnet_kd, nafnet_kd_fp16, nafnet_kd_static_int8,
  drsformer_fp32, drsformer_fp16
"""
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset  # noqa: E402
from evaluation.metrics import compute_psnr  # noqa: E402
from evaluation.model_wrappers import load_model  # noqa: E402

TABLES = PROJECT_ROOT / "results" / "baselines" / "tables"
OUT_CSV = TABLES / "combined_compression.csv"
TEST_ROOT = PROJECT_ROOT / "datasets" / "Rain13K" / "test"

PRUNE_CKPT = {
    "restormer":  PROJECT_ROOT / "experiments/pruning/checkpoints/restormer_pruned_0.3_finetuned.pth",
    "nafnet_w32": PROJECT_ROOT / "experiments/pruning/checkpoints/nafnet_w32_pruned_0.3_finetuned.pth",
}


def _purge_basicsr():
    for k in [m for m in sys.modules if m == "basicsr" or m.startswith("basicsr.")]:
        del sys.modules[k]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find(rows, **filters):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in filters.items()):
            return r
    return None


def to_float(v):
    if v is None or v == "" or v == "n/a":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def measure_fp16_on_pruned(model_name: str) -> dict:
    """Load a pruned-and-finetuned checkpoint, cast to FP16 on GPU,
    measure PSNR on Rain100H + Rain100L, model size, and GPU latency."""
    ckpt = PRUNE_CKPT[model_name]
    print(f"\n[NEW] {model_name}_pruned30_ft_fp16 — loading {ckpt.name}")
    _purge_basicsr()
    device = "cuda"
    model = load_model(model_name, str(ckpt), device).half().eval()

    # Save FP16 state_dict for size-on-disk measurement.
    out_dir = PROJECT_ROOT / "experiments" / "pruning" / "checkpoints" / "fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = out_dir / f"{model_name}_pruned_0.3_finetuned_fp16.pth"
    torch.save(model.state_dict(), fp16_path)
    size_mb = os.path.getsize(fp16_path) / 1e6

    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    # Pure FP16 softmax overflows in Restormer's MDTA (pruned weights amplify
    # this). Wrap the forward in autocast(fp16) so sensitive ops are upcast,
    # matching the recipe used in experiments/quantization/ptq_restormer.py.
    def _forward(t):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(t)

    # Latency: 100 forward passes at 1×3×256×256 (FP16 input).
    x = torch.randn(1, 3, 256, 256, device=device).half()
    with torch.no_grad():
        for _ in range(10):
            _ = _forward(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _forward(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    latency_ms = float(np.mean(times))

    # PSNR on Rain100H + Rain100L (Y-channel, 4px crop — same as compute_psnr).
    psnrs = {}
    for ds_name in ("Rain100H", "Rain100L"):
        ds = DerainDataset(
            input_dir=str(TEST_ROOT / ds_name / "input"),
            target_dir=str(TEST_ROOT / ds_name / "target"),
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        vals = []
        for inp, gt, _ in tqdm(loader, desc=f"  {ds_name}", leave=False):
            inp = inp.to(device).half()
            with torch.no_grad():
                pred = _forward(inp)
            pred = torch.clamp(pred.float().cpu(), 0, 1)
            vals.append(compute_psnr(pred, gt.float().cpu()))
        psnrs[ds_name] = float(np.mean(vals))

    del model
    torch.cuda.empty_cache()
    return {
        "psnr_rain100h": psnrs["Rain100H"],
        "psnr_rain100l": psnrs["Rain100L"],
        "params_M": params_M,
        "model_size_mb": size_mb,
        "latency_ms": latency_ms,
    }


def main():
    base = {
        "restormer":  read_csv(TABLES / "restormer_results.csv"),
        "nafnet_w32": read_csv(TABLES / "nafnet_w32_results.csv"),
        "drsformer":  read_csv(TABLES / "drsformer_results.csv"),
        "nafnet_kd":  read_csv(TABLES / "nafnet_w32_kd_results.csv"),
    }
    quant = {
        "restormer":  read_csv(TABLES / "restormer_quantization.csv"),
        "nafnet_w32": read_csv(TABLES / "nafnet_w32_quantization.csv"),
        "drsformer":  read_csv(TABLES / "drsformer_quantization.csv"),
        "nafnet_kd":  read_csv(TABLES / "nafnet_w32_kd_quantization.csv"),
    }
    prune = {
        "restormer":  read_csv(TABLES / "restormer_pruning.csv"),
        "nafnet_w32": read_csv(TABLES / "nafnet_w32_pruning.csv"),
    }

    out_rows = []

    def from_baseline(model_key, label):
        rows = base[model_key]
        h = find(rows, dataset="Rain100H")
        l = find(rows, dataset="Rain100L")
        return {
            "model": model_key, "technique": label,
            "psnr_rain100h": to_float(h["psnr"]),
            "psnr_rain100l": to_float(l["psnr"]),
            "params_M": to_float(h["params_M"]),
            "model_size_mb": to_float(h["model_size_MB"]),
            "latency_ms": to_float(h["mean_ms"]),
        }

    def from_quant(model_key, variant, label):
        rows = quant[model_key]
        h = find(rows, variant=variant, dataset="Rain100H")
        l = find(rows, variant=variant, dataset="Rain100L")
        if not h:
            return None
        # Quant CSVs lack params_M — pull from the model's results CSV.
        base_rows = base.get("nafnet_w32" if model_key == "nafnet_kd" else model_key, [])
        params = to_float(base_rows[0]["params_M"]) if base_rows else float("nan")
        return {
            "model": model_key, "technique": label,
            "psnr_rain100h": to_float(h["psnr"]),
            "psnr_rain100l": to_float(l["psnr"]) if l else float("nan"),
            "params_M": params,
            "model_size_mb": to_float(h["model_size_mb"]),
            "latency_ms": to_float(h["latency_ms"]),
        }

    def from_prune(model_key, ratio, label):
        rows = prune[model_key]
        h = find(rows, prune_ratio=str(ratio), stage="prune_ft", dataset="Rain100H")
        l = find(rows, prune_ratio=str(ratio), stage="prune_ft", dataset="Rain100L")
        return {
            "model": model_key, "technique": label,
            "psnr_rain100h": to_float(h["psnr"]),
            "psnr_rain100l": to_float(l["psnr"]),
            "params_M": to_float(h["total_params_M"]),
            "model_size_mb": to_float(h["model_size_mb"]),
            "latency_ms": to_float(h["latency_ms"]),
        }

    # Restormer ---------------------------------------------------------------
    out_rows.append(from_baseline("restormer", "restormer_fp32"))
    out_rows.append(from_quant("restormer", "fp16", "restormer_fp16"))
    out_rows.append(from_quant("restormer", "static_int8", "restormer_static_int8"))
    out_rows.append(from_prune("restormer", 0.3, "restormer_pruned30_ft"))

    # NEW: restormer_pruned30_ft_fp16
    r = measure_fp16_on_pruned("restormer")
    out_rows.append({
        "model": "restormer", "technique": "restormer_pruned30_ft_fp16",
        **r,
    })

    # NAFNet ------------------------------------------------------------------
    out_rows.append(from_baseline("nafnet_w32", "nafnet_fp32"))
    out_rows.append(from_quant("nafnet_w32", "fp16", "nafnet_fp16"))
    out_rows.append(from_quant("nafnet_w32", "static_int8", "nafnet_static_int8"))
    out_rows.append(from_prune("nafnet_w32", 0.3, "nafnet_pruned30_ft"))

    r = measure_fp16_on_pruned("nafnet_w32")
    out_rows.append({
        "model": "nafnet_w32", "technique": "nafnet_pruned30_ft_fp16",
        **r,
    })

    # KD ----------------------------------------------------------------------
    out_rows.append(from_baseline("nafnet_kd", "nafnet_kd"))
    kd_fp16 = from_quant("nafnet_kd", "fp16", "nafnet_kd_fp16")
    kd_int8 = from_quant("nafnet_kd", "static_int8", "nafnet_kd_static_int8")
    if kd_fp16 is None or kd_int8 is None:
        print("WARNING: nafnet_w32_kd_quantization.csv missing rows — "
              "run experiments/distillation/quantize_kd.py first.")
    if kd_fp16 is not None:
        out_rows.append(kd_fp16)
    if kd_int8 is not None:
        out_rows.append(kd_int8)

    # DRSformer ---------------------------------------------------------------
    out_rows.append(from_baseline("drsformer", "drsformer_fp32"))
    out_rows.append(from_quant("drsformer", "fp16", "drsformer_fp16"))

    # Write -------------------------------------------------------------------
    fieldnames = ["model", "technique", "psnr_rain100h", "psnr_rain100l",
                  "params_M", "model_size_mb", "latency_ms"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            r = {k: row.get(k) for k in fieldnames}
            for fk in ("psnr_rain100h", "psnr_rain100l", "params_M",
                       "model_size_mb", "latency_ms"):
                v = r.get(fk)
                if isinstance(v, float) and not math.isnan(v):
                    r[fk] = round(v, 4)
            w.writerow(r)
    print(f"\nWrote {OUT_CSV} with {len(out_rows)} rows.")
    print(f"\n{'technique':<32}{'PSNR-H':>9}{'PSNR-L':>9}{'params':>9}{'size':>9}{'lat':>9}")
    for row in out_rows:
        print(f"{row['technique']:<32}"
              f"{row['psnr_rain100h']:>9.2f}"
              f"{row['psnr_rain100l']:>9.2f}"
              f"{row['params_M']:>9.2f}"
              f"{row['model_size_mb']:>9.2f}"
              f"{row['latency_ms']:>9.2f}")


if __name__ == "__main__":
    main()
