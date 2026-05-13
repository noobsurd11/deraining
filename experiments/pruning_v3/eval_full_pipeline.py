"""
Full post-training pipeline for NAFNet-KD-StructPrune30:
  1. Load best EMA checkpoint (FP32) → eval all 5 test sets
  2. Convert to FP16 → eval Rain100H/L to verify lossless
  3. Export ONNX FP16 → benchmark ORT GPU → quality check
  4. Save results CSV + print pipeline waterfall
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_psnr, compute_ssim, LPIPSMetric
from evaluation.efficiency import count_parameters, count_gmacs, measure_latency, get_model_size_mb
from evaluation.dataset import DerainDataset
from torch.utils.data import DataLoader

DEVICE = "cuda"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
BEST_CKPT = CKPT_DIR / "nafnet_kd_structpruned30_best.pth"
ONNX_DIR = Path(__file__).resolve().parent / "onnx"
CSV_PATH = Path(__file__).resolve().parent / "results_full_pipeline.csv"
MASTER_CSV = PROJECT_ROOT / "results" / "baselines" / "tables" / "master_results_v2.csv"

TESTSETS = {
    "Rain100H": PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Rain100H",
    "Rain100L": PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Rain100L",
    "Test100":  PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Test100",
    "Test1200": PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Test1200",
    "Test2800": PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Test2800",
}

PADDER = 16


def _purge_basicsr():
    for k in [m for m in sys.modules if m == "basicsr" or m.startswith("basicsr.")]:
        del sys.modules[k]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def set_seed(s=42):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_pruned_model():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from structured_pruning_physical import _build_pruned_student
    model = _build_pruned_student(device="cpu")
    sd = torch.load(str(BEST_CKPT), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model


@torch.no_grad()
def eval_all_testsets(model, device=DEVICE, use_amp=False):
    lpips_fn = LPIPSMetric(device)
    results = {}
    for name, path in TESTSETS.items():
        ds = DerainDataset(str(path / "input"), str(path / "target"))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        psnrs, ssims, lpipss = [], [], []
        for inp, gt, _ in tqdm(loader, desc=f"  {name}", leave=False):
            inp, gt = inp.to(device), gt.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = torch.clamp(model(inp.half()), 0, 1).float()
            else:
                pred = torch.clamp(model(inp), 0, 1)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim(pred, gt))
            lpipss.append(lpips_fn.compute(pred, gt))
        results[name] = {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "lpips": float(np.mean(lpipss)),
        }
        print(f"  {name}: PSNR={results[name]['psnr']:.2f}  SSIM={results[name]['ssim']:.4f}  LPIPS={results[name]['lpips']:.4f}")
    return results


@torch.no_grad()
def eval_rain100h_l(model, device=DEVICE, use_amp=False):
    results = {}
    for name in ["Rain100H", "Rain100L"]:
        path = TESTSETS[name]
        ds = DerainDataset(str(path / "input"), str(path / "target"))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        psnrs = []
        for inp, gt, _ in tqdm(loader, desc=f"  {name}", leave=False):
            inp, gt = inp.to(device), gt.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = torch.clamp(model(inp.half()), 0, 1).float()
            else:
                pred = torch.clamp(model(inp), 0, 1)
            psnrs.append(compute_psnr(pred, gt))
        results[name] = float(np.mean(psnrs))
        print(f"  {name}: PSNR={results[name]:.4f}")
    return results


def benchmark_pytorch(model, dtype=torch.float32, use_amp=False, warmup=10, runs=100):
    x = torch.randn(1, 3, 256, 256, device=DEVICE).to(dtype)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def export_onnx(model, out_path, half=False):
    dtype = torch.float16 if half else torch.float32
    dummy = torch.randn(1, 3, 256, 256, device=DEVICE).to(dtype)
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"},
                      "output": {2: "height", 3: "width"}},
        dynamo=False,
    )
    print(f"  Exported: {out_path} ({get_model_size_mb(str(out_path)):.1f} MB)")


def benchmark_ort_gpu(onnx_path, dtype=np.float32, warmup=10, runs=100):
    import onnxruntime as ort
    providers = [("CUDAExecutionProvider", {"device_id": 0})]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    in_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, 256, 256).astype(dtype)
    for _ in range(warmup):
        sess.run(None, {in_name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), sess


def quality_check_ort(sess, dtype=np.float16, n=20):
    rain_in = sorted(p for p in (TESTSETS["Rain100H"] / "input").iterdir()
                     if p.suffix.lower() in (".png", ".jpg"))[:n]
    rain_gt = sorted(p for p in (TESTSETS["Rain100H"] / "target").iterdir()
                     if p.suffix.lower() in (".png", ".jpg"))[:n]
    to_t = transforms.ToTensor()
    in_name = sess.get_inputs()[0].name
    psnrs = []
    for fp_in, fp_gt in zip(rain_in, rain_gt):
        inp = to_t(Image.open(fp_in).convert("RGB")).unsqueeze(0).numpy().astype(dtype)
        gt = to_t(Image.open(fp_gt).convert("RGB")).unsqueeze(0)
        _, _, h, w = inp.shape
        mod_h, mod_w = (PADDER - h % PADDER) % PADDER, (PADDER - w % PADDER) % PADDER
        if mod_h or mod_w:
            inp = np.pad(inp, ((0,0),(0,0),(0,mod_h),(0,mod_w)), mode="reflect")
        out = sess.run(None, {in_name: inp})[0][:, :, :h, :w]
        pred = torch.clamp(torch.from_numpy(out.astype(np.float32)), 0, 1)
        psnrs.append(compute_psnr(pred, gt))
    return float(np.mean(psnrs))


def peak_gpu_mem(model, dtype=torch.float32):
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1, 3, 256, 256, device=DEVICE).to(dtype)
    with torch.no_grad():
        model(x)
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    set_seed()
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # ── FP32 ──
    print("=" * 60)
    print("  FP32 Evaluation")
    print("=" * 60)
    model = build_pruned_model().to(DEVICE).eval()
    params_m = count_parameters(model)
    gmacs = count_gmacs(model, device=DEVICE)
    lat_fp32 = benchmark_pytorch(model, torch.float32)
    mem_fp32 = peak_gpu_mem(model, torch.float32)
    fp32_ckpt_size = get_model_size_mb(str(BEST_CKPT))
    print(f"  Params: {params_m:.4f}M  GMACs: {gmacs:.3f}  Latency: {lat_fp32:.2f}ms  Size: {fp32_ckpt_size:.1f}MB  PeakMem: {mem_fp32:.1f}MB")

    res_fp32 = eval_all_testsets(model)

    rows.append({
        "variant": "structpruned30_fp32",
        "rain100h": res_fp32["Rain100H"]["psnr"],
        "rain100l": res_fp32["Rain100L"]["psnr"],
        "test100": res_fp32["Test100"]["psnr"],
        "test1200": res_fp32["Test1200"]["psnr"],
        "test2800": res_fp32["Test2800"]["psnr"],
        "params_m": params_m, "gmacs": gmacs,
        "latency_pytorch_ms": lat_fp32, "latency_ort_ms": float("nan"),
        "size_mb": fp32_ckpt_size, "peak_mem_mb": mem_fp32,
    })

    # ── FP16 ──
    print("\n" + "=" * 60)
    print("  FP16 Evaluation")
    print("=" * 60)
    model_fp16 = model.half()
    fp16_path = CKPT_DIR / "nafnet_kd_structpruned30_best_fp16.pth"
    torch.save(model_fp16.state_dict(), fp16_path)
    fp16_size = get_model_size_mb(str(fp16_path))
    lat_fp16 = benchmark_pytorch(model_fp16, torch.float16, use_amp=True)
    mem_fp16 = peak_gpu_mem(model_fp16, torch.float16)
    print(f"  Size: {fp16_size:.1f}MB  Latency: {lat_fp16:.2f}ms  PeakMem: {mem_fp16:.1f}MB")

    res_fp16 = eval_rain100h_l(model_fp16, use_amp=True)
    print(f"  FP32→FP16 delta: Rain100H={res_fp16['Rain100H'] - res_fp32['Rain100H']['psnr']:+.4f} dB, "
          f"Rain100L={res_fp16['Rain100L'] - res_fp32['Rain100L']['psnr']:+.4f} dB")

    rows.append({
        "variant": "structpruned30_fp16",
        "rain100h": res_fp16["Rain100H"],
        "rain100l": res_fp16["Rain100L"],
        "test100": float("nan"), "test1200": float("nan"), "test2800": float("nan"),
        "params_m": params_m, "gmacs": gmacs,
        "latency_pytorch_ms": lat_fp16, "latency_ort_ms": float("nan"),
        "size_mb": fp16_size, "peak_mem_mb": mem_fp16,
    })

    # ── ONNX FP32 ──
    print("\n" + "=" * 60)
    print("  ONNX FP32 Export + ORT Benchmark")
    print("=" * 60)
    model_fp32_onnx = build_pruned_model().to(DEVICE).eval()
    onnx_path = ONNX_DIR / "nafnet_kd_structpruned30_fp32.onnx"
    export_onnx(model_fp32_onnx, onnx_path, half=False)
    onnx_size = get_model_size_mb(str(onnx_path))
    del model_fp32_onnx
    torch.cuda.empty_cache()

    lat_ort, sess = benchmark_ort_gpu(onnx_path, dtype=np.float32)
    print(f"  ORT GPU latency: {lat_ort:.2f}ms")

    ort_psnr = quality_check_ort(sess, dtype=np.float32)
    print(f"  ORT PSNR (20-img Rain100H): {ort_psnr:.2f} dB")

    rows.append({
        "variant": "structpruned30_onnx_fp32",
        "rain100h": ort_psnr,
        "rain100l": float("nan"),
        "test100": float("nan"), "test1200": float("nan"), "test2800": float("nan"),
        "params_m": params_m, "gmacs": gmacs,
        "latency_pytorch_ms": lat_fp32, "latency_ort_ms": lat_ort,
        "size_mb": onnx_size, "peak_mem_mb": float("nan"),
    })

    # ── Save CSV ──
    fields = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {CSV_PATH}")

    # ── Pipeline waterfall ──
    print("\n" + "=" * 70)
    print("  COMPRESSION PIPELINE WATERFALL")
    print("=" * 70)
    print(f"  {'Stage':<35s} {'PSNR-H':>8s} {'PSNR-L':>8s} {'Params':>8s} {'GMACs':>7s} {'Size':>8s} {'Lat(ms)':>8s}")
    print("  " + "-" * 84)
    stages = [
        ("Restormer FP32 (teacher)",     31.48, 39.15, 26.13, 154.9, 104.7, 46.4),
        ("NAFNet-KD FP32",               30.43, 37.32, 29.16, 16.1, 116.9, 11.3),
        ("+ StructPrune30 FP32",         res_fp32["Rain100H"]["psnr"], res_fp32["Rain100L"]["psnr"], params_m, gmacs, fp32_ckpt_size, lat_fp32),
        ("+ FP16",                       res_fp16["Rain100H"], res_fp16["Rain100L"], params_m, gmacs, fp16_size, lat_fp16),
        ("+ ONNX Runtime GPU",           ort_psnr, float("nan"), params_m, gmacs, onnx_size, lat_ort),
    ]
    for name, ph, pl, p, g, s, l in stages:
        pl_str = f"{pl:.2f}" if not np.isnan(pl) else "-"
        print(f"  {name:<35s} {ph:8.2f} {pl_str:>8s} {p:7.2f}M {g:7.1f} {s:7.1f}MB {l:7.2f}")
    print("=" * 70)

    # Clean up GPU
    del model, model_fp16
    torch.cuda.empty_cache()

    # ── Master CSV ──
    print("\nBuilding master results table...")
    build_master_csv(rows)


def build_master_csv(v3_rows):
    """Combine all existing CSVs into one master table."""
    master = []
    tables_dir = PROJECT_ROOT / "results" / "baselines" / "tables"

    # Restormer baselines
    for r in _read_csv(tables_dir / "restormer_results.csv"):
        master.append(_baseline_row("restormer", r["dataset"], r))
    # NAFNet-w32 baselines
    for r in _read_csv(tables_dir / "nafnet_w32_results.csv"):
        master.append(_baseline_row("nafnet_w32", r["dataset"], r))
    # NAFNet-KD baselines
    for r in _read_csv(tables_dir / "nafnet_w32_kd_results.csv"):
        master.append(_baseline_row("nafnet_kd", r["dataset"], r))
    # DRSformer baselines
    for r in _read_csv(tables_dir / "drsformer_results.csv"):
        master.append(_baseline_row("drsformer", r["dataset"], r))

    # Quantization
    for src in ["restormer_quantization.csv", "nafnet_w32_quantization.csv",
                "nafnet_w32_kd_quantization.csv", "drsformer_quantization.csv"]:
        model_name = src.replace("_quantization.csv", "")
        for r in _read_csv(tables_dir / src):
            master.append({
                "model": model_name, "variant": r.get("variant", ""),
                "dataset": r.get("dataset", ""),
                "psnr": _f(r, "psnr"), "ssim": _f(r, "ssim"), "lpips": _f(r, "lpips"),
                "params_m": "", "gmacs": "",
                "size_mb": _f(r, "model_size_mb"), "latency_ms": _f(r, "latency_ms"),
                "device": r.get("device", ""),
            })

    # ONNX benchmarks
    for r in _read_csv(tables_dir / "onnx_benchmark.csv"):
        master.append({
            "model": r.get("model_variant", ""), "variant": "onnx",
            "dataset": "Rain100H_20img",
            "psnr": _f(r, "psnr_rain100h_20img"), "ssim": "", "lpips": "",
            "params_m": "", "gmacs": "",
            "size_mb": _f(r, "onnx_size_mb"),
            "latency_ms": _f(r, "ort_gpu_latency_ms"),
            "device": "ort_gpu",
        })

    # v2 pruning
    for r in _read_csv(Path(__file__).resolve().parents[1] / "pruning_v2" / "results.csv"):
        stage = r.get("stage", "")
        master.append({
            "model": "nafnet_kd_w22", "variant": stage,
            "dataset": "summary",
            "psnr": _f(r, "psnr_rain100h"), "ssim": "", "lpips": "",
            "params_m": _f(r, "total_params_M"), "gmacs": _f(r, "gmacs"),
            "size_mb": _f(r, "model_size_mb"), "latency_ms": _f(r, "gpu_latency_ms"),
            "device": "gpu",
        })

    # v3 structural pruning (from this run)
    for vr in v3_rows:
        master.append({
            "model": "nafnet_kd_structpruned30", "variant": vr["variant"],
            "dataset": "all",
            "psnr": vr["rain100h"], "ssim": "", "lpips": "",
            "params_m": vr["params_m"], "gmacs": vr["gmacs"],
            "size_mb": vr["size_mb"],
            "latency_ms": vr.get("latency_ort_ms", vr["latency_pytorch_ms"]),
            "device": "gpu",
        })

    fields = ["model", "variant", "dataset", "psnr", "ssim", "lpips",
              "params_m", "gmacs", "size_mb", "latency_ms", "device"]
    with open(MASTER_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(master)
    print(f"Saved master CSV: {MASTER_CSV} ({len(master)} rows)")


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row, key):
    v = row.get(key, "")
    if v in ("", "nan", None):
        return ""
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _baseline_row(model, dataset, r):
    return {
        "model": model, "variant": "fp32_baseline",
        "dataset": dataset,
        "psnr": _f(r, "psnr"), "ssim": _f(r, "ssim"), "lpips": _f(r, "lpips"),
        "params_m": _f(r, "params_M"), "gmacs": _f(r, "gmacs"),
        "size_mb": _f(r, "model_size_MB"), "latency_ms": _f(r, "mean_ms"),
        "device": "gpu",
    }


if __name__ == "__main__":
    main()
