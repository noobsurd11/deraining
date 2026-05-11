"""
Quantize the KD-distilled NAFNet-w32 model.

Applies FP16 (GPU) and static INT8 (CPU, FX graph mode PTQ, x86 backend,
calibrated on 100 Rain13K crops) to pretrained/nafnet_w32_kd.pth.

Evaluates each variant on Rain100H and Rain100L and writes
results/baselines/tables/nafnet_w32_kd_quantization.csv with columns:
  variant, dataset, psnr, ssim, lpips, model_size_mb, latency_ms, device

Mirrors experiments/quantization/ptq_nafnet.py — same calibration set,
same LayerNorm2d non-traceable handling, same check_image_size bypass.
"""
import csv
import os
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset  # noqa: E402
from evaluation.metrics import LPIPSMetric, compute_psnr, compute_ssim  # noqa: E402
from evaluation.model_wrappers import load_model  # noqa: E402

SEED = 42

torch.set_num_threads(48)
torch.set_num_interop_threads(48)

CKPT = PROJECT_ROOT / "pretrained" / "nafnet_w32_kd.pth"
CKPT_DIR = PROJECT_ROOT / "experiments" / "distillation" / "checkpoints" / "quantized"
CSV_PATH = PROJECT_ROOT / "results" / "baselines" / "tables" / "nafnet_w32_kd_quantization.csv"
CALIB_DIR = PROJECT_ROOT / "datasets" / "Rain13K" / "train" / "input"

TEST_ROOT = PROJECT_ROOT / "datasets" / "Rain13K" / "test"
TESTSETS = {
    "Rain100H": TEST_ROOT / "Rain100H",
    "Rain100L": TEST_ROOT / "Rain100L",
}

CKPT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

PADDER_SIZE = 16  # NAFNet downsamples 4x → pad to multiple of 16


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_multiple(x: torch.Tensor, m: int = PADDER_SIZE):
    _, _, h, w = x.shape
    mod_h = (m - h % m) % m
    mod_w = (m - w % m) % m
    if mod_h == 0 and mod_w == 0:
        return x, h, w
    return F.pad(x, (0, mod_w, 0, mod_h)), h, w


def _iter_dataset(testset_path: Path):
    ds = DerainDataset(
        input_dir=str(testset_path / "input"),
        target_dir=str(testset_path / "target"),
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2), len(ds)


_LPIPS_METRIC = None


def _lpips() -> LPIPSMetric:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        _LPIPS_METRIC = LPIPSMetric("cpu")
    return _LPIPS_METRIC


def eval_quality(forward_fn, device: str, testset_path: Path, label: str,
                 manual_pad: bool = False) -> dict:
    loader, n = _iter_dataset(testset_path)
    lpips_metric = _lpips()
    psnrs, ssims, lpipss = [], [], []
    for inp, gt, _ in tqdm(loader, desc=f"  eval[{label}] {testset_path.name} ({n})", leave=False):
        inp = inp.to(device)
        with torch.no_grad():
            if manual_pad:
                padded, H, W = pad_to_multiple(inp)
                pred = forward_fn(padded)
                pred = pred[:, :, :H, :W]
            else:
                pred = forward_fn(inp)
        pred = torch.clamp(pred.float().cpu(), 0, 1)
        gt_cpu = gt.float().cpu()
        psnrs.append(compute_psnr(pred, gt_cpu))
        ssims.append(compute_ssim(pred, gt_cpu))
        lpipss.append(lpips_metric.compute(pred, gt_cpu))
    return {"psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "lpips": float(np.mean(lpipss))}


def measure_latency(forward_fn, device: str, input_size=(1, 3, 256, 256),
                    warmup: int = 10, runs: int = 100, dtype=torch.float32) -> float:
    actual_runs = max(5, runs // 10) if device == "cpu" else runs
    actual_warmup = max(2, warmup // 5) if device == "cpu" else warmup
    x = torch.randn(*input_size, device=device).to(dtype)
    is_cuda = device.startswith("cuda")
    with torch.no_grad():
        for _ in range(actual_warmup):
            _ = forward_fn(x)
        if is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(actual_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = forward_fn(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def save_size_mb(path: Path) -> float:
    return os.path.getsize(path) / 1e6


def run_fp16() -> dict:
    if not torch.cuda.is_available():
        return {"rows": [], "notes": "SKIPPED (no CUDA)."}
    device = "cuda"
    model = load_model("nafnet_w32", str(CKPT), device).half()
    fp16_path = CKPT_DIR / "nafnet_w32_kd_fp16.pth"
    torch.save(model.state_dict(), fp16_path)
    size_mb = save_size_mb(fp16_path)

    def forward(x):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x.half())

    lat = measure_latency(forward, device, dtype=torch.float16)
    rows = []
    for name, p in TESTSETS.items():
        q = eval_quality(forward, device, p, "fp16")
        rows.append({"variant": "fp16", "dataset": name, **q,
                     "model_size_mb": size_mb, "latency_ms": lat,
                     "device": device})
    del model
    torch.cuda.empty_cache()
    return {"rows": rows,
            "notes": f"FP16 saved to {fp16_path.name} ({size_mb:.2f} MB)."}


def _load_calibration_batch(k: int = 100, size: int = 256) -> list:
    rng = random.Random(SEED)
    all_files = sorted(CALIB_DIR.glob("*.jpg"))
    if not all_files:
        all_files = sorted(CALIB_DIR.glob("*.png"))
    picks = rng.sample(all_files, min(k, len(all_files)))
    to_tensor = transforms.ToTensor()
    tensors = []
    for fp in picks:
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        if w < size or h < size:
            img = img.resize((max(w, size), max(h, size)), Image.BILINEAR)
            w, h = img.size
        left = rng.randint(0, w - size)
        top = rng.randint(0, h - size)
        img = img.crop((left, top, left + size, top + size))
        tensors.append(to_tensor(img).unsqueeze(0))
    return tensors


def run_static_int8() -> dict:
    from torch.ao.quantization import (
        QConfigMapping,
        get_default_qconfig,
        get_default_qconfig_mapping,
    )
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "x86"
    device = "cpu"

    sys.path.insert(0, str(PROJECT_ROOT / "models" / "NAFNet"))
    from basicsr.models.archs.arch_util import LayerNorm2d

    model = load_model("nafnet_w32", str(CKPT), device).eval()
    model.check_image_size = lambda x: x
    example_input = (torch.randn(1, 3, 256, 256),)
    calib = _load_calibration_batch(k=100, size=256)

    note_lines = []
    qmodel = None
    used_fallback = False
    for attempt in ("default", "conv_only"):
        try:
            if attempt == "default":
                qmap = get_default_qconfig_mapping("x86")
            else:
                used_fallback = True
                qmap = QConfigMapping().set_global(None).set_object_type(
                    nn.Conv2d, get_default_qconfig("x86")
                )
            pcc = PrepareCustomConfig().set_non_traceable_module_classes([LayerNorm2d])
            prepared = prepare_fx(model, qmap, example_input,
                                  prepare_custom_config=pcc)
            with torch.no_grad():
                for t in tqdm(calib, desc=f"  calibrate[{attempt}]", leave=False):
                    _ = prepared(t)
            qmodel = convert_fx(prepared)
            note_lines.append(f"Static INT8 prepared (qconfig='{attempt}').")
            break
        except Exception as exc:
            note_lines.append(f"Static INT8 qconfig='{attempt}' FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            qmodel = None
            model = load_model("nafnet_w32", str(CKPT), device).eval()
            model.check_image_size = lambda x: x

    if qmodel is None:
        return {"rows": [], "notes": "\n    ".join(note_lines + ["All static INT8 attempts failed."])}

    suffix = "_conv_only" if used_fallback else ""
    ckpt_path = CKPT_DIR / f"nafnet_w32_kd_static_int8{suffix}.pth"
    try:
        torch.save(qmodel.state_dict(), ckpt_path)
    except Exception as exc:
        note_lines.append(f"state_dict save failed ({exc}); pickling full module.")
        torch.save(qmodel, ckpt_path)
    size_mb = save_size_mb(ckpt_path)
    note_lines.append(f"Saved to {ckpt_path.name} ({size_mb:.2f} MB).")

    forward = lambda x: qmodel(x)
    try:
        lat = measure_latency(forward, device, warmup=3, runs=20)
    except Exception as exc:
        note_lines.append(f"Latency measurement failed: {exc}")
        lat = float("nan")

    rows = []
    for name, p in TESTSETS.items():
        try:
            q = eval_quality(forward, device, p, "stat_int8", manual_pad=True)
        except Exception as exc:
            note_lines.append(f"Quality eval failed on {name}: {exc}")
            q = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
        rows.append({"variant": "static_int8", "dataset": name, **q,
                     "model_size_mb": size_mb, "latency_ms": lat, "device": device})
    return {"rows": rows, "notes": "\n    ".join(note_lines)}


def run_with_guard(name: str, fn):
    print(f"\n=== {name} ===")
    try:
        out = fn()
        print(f"  notes: {out['notes']}")
        return out["rows"]
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return [{"variant": name, "dataset": ds, "psnr": float("nan"),
                 "ssim": float("nan"), "lpips": float("nan"),
                 "model_size_mb": float("nan"), "latency_ms": float("nan"),
                 "device": "n/a"} for ds in TESTSETS]


def print_summary(rows: list):
    by_var: dict[str, list] = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)
    print("\n" + "=" * 88)
    print(f"{'variant':<16}{'avg_psnr':>10}{'size_mb':>10}{'latency':>10}{'device':>10}")
    print("-" * 88)
    for var, rs in by_var.items():
        psnrs = [r["psnr"] for r in rs if not np.isnan(r["psnr"])]
        avg_psnr = np.mean(psnrs) if psnrs else float("nan")
        size = rs[0]["model_size_mb"]
        lat = rs[0]["latency_ms"]
        dev = rs[0]["device"]
        print(f"{var:<16}{avg_psnr:>10.3f}{size:>10.2f}{lat:>10.2f}{dev:>10}")
    print("=" * 88)


def main():
    set_seed()
    print(f"KD checkpoint: {CKPT}")
    print(f"CUDA: {torch.cuda.is_available()}")

    all_rows = []
    all_rows += run_with_guard("fp16", run_fp16)
    all_rows += run_with_guard("static_int8", run_static_int8)

    fieldnames = ["variant", "dataset", "psnr", "ssim", "lpips",
                  "model_size_mb", "latency_ms", "device"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nCSV written: {CSV_PATH}")
    print_summary(all_rows)


if __name__ == "__main__":
    main()
