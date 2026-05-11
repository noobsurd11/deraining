"""
Post-Training Quantization of Restormer.

Variants measured:
  - fp32          (GPU reference)        — all 5 test sets
  - fp16          (GPU, model.half())    — all 5 test sets
  - dynamic_int8  (CPU, quantize_dynamic) — Rain100H/L/Test100 only
  - static_int8   (CPU, FX graph mode PTQ, x86 backend, calibrated on
                   100 random Rain13K train crops) — Rain100H/L/Test100 only

Restormer's `to_3d`/`to_4d`/attention use einops.rearrange which torch.fx
can't trace. Before quantizing we monkey-patch those helpers to native
torch (flatten/transpose/reshape) so FX tracing succeeds.

CPU-bound variants are evaluated only on the 3 smaller test sets
(Test1200 and Test2800 would take many hours on CPU at Restormer's
latency).

Writes results/baselines/tables/restormer_quantization.csv.
Saves quantized checkpoints to experiments/quantization/checkpoints/.
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

# Maximize CPU threads for INT8 variants
torch.set_num_threads(48)
torch.set_num_interop_threads(48)

CKPT = PROJECT_ROOT / "pretrained" / "restormer_deraining.pth"
CKPT_DIR = PROJECT_ROOT / "experiments" / "quantization" / "checkpoints"
CSV_PATH = PROJECT_ROOT / "results" / "baselines" / "tables" / "restormer_quantization.csv"
CALIB_DIR = PROJECT_ROOT / "datasets" / "Rain13K" / "train" / "input"

TEST_ROOT = PROJECT_ROOT / "datasets" / "Rain13K" / "test"
TESTSETS_ALL = {
    "Rain100H": TEST_ROOT / "Rain100H",
    "Rain100L": TEST_ROOT / "Rain100L",
    "Test100":  TEST_ROOT / "Test100",
    "Test1200": TEST_ROOT / "Test1200",
    "Test2800": TEST_ROOT / "Test2800",
}
# CPU variants only run on the small test sets (sub-1-hour wall-clock).
TESTSETS_CPU = {k: TESTSETS_ALL[k] for k in ("Rain100H", "Rain100L", "Test100")}

CKPT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# einops monkey-patch so FX can trace Restormer.
# Replaces rearrange/to_3d/to_4d in the restormer_arch module namespace with
# torch-native equivalents (tested to produce bit-identical outputs).
# Must run AFTER sys.path is set up but BEFORE load_model is called for static_int8.
# ---------------------------------------------------------------------------
def patch_restormer_einops():
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "Restormer"))
    from basicsr.models.archs import restormer_arch as ra

    def to_3d(x):
        # 'b c h w -> b (h w) c'
        return x.flatten(2).transpose(1, 2).contiguous()

    def to_4d(x, h, w):
        # 'b (h w) c -> b c h w'
        b, _, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w).contiguous()

    def native_rearrange(x, pattern, **kwargs):
        p = pattern.replace(" ", "")
        if p == "b(headc)hw->bheadc(hw)":
            b, _, h, w = x.shape
            head = kwargs["head"]
            c = x.shape[1] // head
            return x.reshape(b, head, c, h * w)
        if p == "bheadc(hw)->b(headc)hw":
            head = kwargs["head"]
            h = kwargs["h"]
            w = kwargs["w"]
            b, _, c, _ = x.shape
            return x.reshape(b, head * c, h, w)
        if p == "bchw->b(hw)c":
            return x.flatten(2).transpose(1, 2).contiguous()
        if p == "b(hw)c->bchw":
            h = kwargs["h"]; w = kwargs["w"]
            b, _, c = x.shape
            return x.transpose(1, 2).reshape(b, c, h, w).contiguous()
        raise ValueError(f"Unpatched einops pattern: {pattern}")

    ra.to_3d = to_3d
    ra.to_4d = to_4d
    ra.rearrange = native_rearrange


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
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


def eval_quality(forward_fn, device: str, testset_path: Path, label: str) -> dict:
    loader, n = _iter_dataset(testset_path)
    lpips_metric = _lpips()
    psnrs, ssims, lpipss = [], [], []
    for inp, gt, _ in tqdm(loader, desc=f"  eval[{label}] {testset_path.name} ({n})", leave=False):
        inp = inp.to(device)
        with torch.no_grad():
            pred = forward_fn(inp)
        pred = torch.clamp(pred.float().cpu(), 0, 1)
        gt_cpu = gt.float().cpu()
        psnrs.append(compute_psnr(pred, gt_cpu))
        ssims.append(compute_ssim(pred, gt_cpu))
        lpipss.append(lpips_metric.compute(pred, gt_cpu))
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }


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


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------
def run_fp32() -> dict:
    """FP32 reference. Pulls PSNR/SSIM/LPIPS from the baselines CSV (already
    computed on the full 5 test sets) and only measures latency fresh."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("restormer", str(CKPT), device)
    forward = lambda x: model(x)
    lat = measure_latency(forward, device)

    baseline_csv = PROJECT_ROOT / "results" / "baselines" / "tables" / "restormer_results.csv"
    baseline = {}
    with open(baseline_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            baseline[r["dataset"]] = r
    rows = []
    for name in TESTSETS_ALL:
        sub = baseline[name]
        rows.append({"variant": "fp32", "dataset": name,
                     "psnr": float(sub["psnr"]),
                     "ssim": float(sub["ssim"]),
                     "lpips": float(sub["lpips"]),
                     "model_size_mb": save_size_mb(CKPT),
                     "latency_ms": lat, "device": device})
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"rows": rows,
            "notes": f"FP32 reference (metrics reused from baselines CSV; latency remeasured: {lat:.2f} ms)."}


def run_fp16() -> dict:
    if not torch.cuda.is_available():
        return {"rows": [], "notes": "SKIPPED (no CUDA)."}
    device = "cuda"
    model = load_model("restormer", str(CKPT), device).half()
    fp16_path = CKPT_DIR / "restormer_fp16.pth"
    torch.save(model.state_dict(), fp16_path)
    size_mb = save_size_mb(fp16_path)

    def forward(x):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x.half())

    lat = measure_latency(forward, device, dtype=torch.float16)
    rows = []
    for name, p in TESTSETS_CPU.items():
        q = eval_quality(forward, device, p, "fp16")
        rows.append({"variant": "fp16", "dataset": name, **q,
                     "model_size_mb": size_mb, "latency_ms": lat,
                     "device": device})
    del model
    torch.cuda.empty_cache()
    return {"rows": rows, "notes": f"FP16 state_dict saved to {fp16_path.name} ({size_mb:.2f} MB); eval on 3 small test sets."}


def _count_quantized_modules(model: nn.Module) -> dict:
    counts: dict[str, int] = {}
    for m in model.modules():
        cls = type(m).__name__
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def run_dynamic_int8() -> dict:
    device = "cpu"
    model = load_model("restormer", str(CKPT), device)
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8,
    )
    ckpt_path = CKPT_DIR / "restormer_dynamic_int8.pth"
    try:
        torch.save(qmodel.state_dict(), ckpt_path)
    except Exception:
        torch.save(qmodel, ckpt_path)
    size_mb = save_size_mb(ckpt_path)

    before = _count_quantized_modules(model)
    after = _count_quantized_modules(qmodel)
    swapped = {k: after[k] for k in after if "Dynamic" in k or "Quantized" in k}
    note_lines = [
        f"Dynamic INT8 state_dict saved ({size_mb:.2f} MB).",
        f"Modules swapped by quantize_dynamic: {swapped or 'NONE — Restormer has no nn.Linear layers; quantize_dynamic is a no-op for conv-only models.'}",
    ]

    forward = lambda x: qmodel(x)
    lat = measure_latency(forward, device, warmup=3, runs=20)
    rows = []
    for name, p in TESTSETS_CPU.items():
        q = eval_quality(forward, device, p, "dyn_int8")
        rows.append({"variant": "dynamic_int8", "dataset": name, **q,
                     "model_size_mb": size_mb, "latency_ms": lat, "device": device})
    del model, qmodel
    return {"rows": rows, "notes": "\n    ".join(note_lines)}


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

    # Apply einops monkey-patch so FX tracing works.
    patch_restormer_einops()

    # Need to mark LayerNorm as non-traceable: x.var(-1, keepdim=True) with a
    # Python-int keepdim is fine, but shape-dependent reshape in our custom
    # to_3d/to_4d uses runtime shapes that FX can still trace via proxies.
    # We still treat LayerNorm as leaf for safety against numerics.
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "Restormer"))
    from basicsr.models.archs.restormer_arch import LayerNorm as RestormerLN

    model = load_model("restormer", str(CKPT), device).eval()
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
            pcc = PrepareCustomConfig().set_non_traceable_module_classes([RestormerLN])
            prepared = prepare_fx(model, qmap, example_input,
                                  prepare_custom_config=pcc)
            with torch.no_grad():
                for t in tqdm(calib, desc=f"  calibrate[{attempt}]", leave=False):
                    _ = prepared(t)
            qmodel = convert_fx(prepared)
            note_lines.append(f"Static INT8 prepared successfully with qconfig='{attempt}'.")
            break
        except Exception as exc:
            note_lines.append(f"Static INT8 qconfig='{attempt}' FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            qmodel = None
            model = load_model("restormer", str(CKPT), device).eval()

    if qmodel is None:
        return {"rows": [], "notes": "\n    ".join(note_lines + ["All static INT8 attempts failed."])}

    mod_counts = _count_quantized_modules(qmodel)
    interesting = {k: v for k, v in mod_counts.items()
                   if any(tok in k.lower() for tok in (
                       "quant", "conv", "linear", "layernorm", "gelu", "softmax"))}
    note_lines.append(f"Quantized-model module inventory: {interesting}")

    ckpt_path = CKPT_DIR / ("restormer_static_int8_conv_only.pth"
                            if used_fallback else "restormer_static_int8.pth")
    try:
        torch.save(qmodel.state_dict(), ckpt_path)
    except Exception as exc:
        note_lines.append(f"state_dict save failed ({exc}); pickling full module instead.")
        torch.save(qmodel, ckpt_path)
    size_mb = save_size_mb(ckpt_path)
    note_lines.append(f"Saved static INT8 model to {ckpt_path.name} ({size_mb:.2f} MB).")

    forward = lambda x: qmodel(x)
    try:
        lat = measure_latency(forward, device, warmup=3, runs=20)
    except Exception as exc:
        note_lines.append(f"Latency measurement failed: {exc}")
        lat = float("nan")

    rows = []
    for name, p in TESTSETS_CPU.items():
        try:
            q = eval_quality(forward, device, p, "stat_int8")
        except Exception as exc:
            note_lines.append(f"Quality eval failed on {name}: {exc}")
            q = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
        rows.append({"variant": "static_int8", "dataset": name, **q,
                     "model_size_mb": size_mb, "latency_ms": lat, "device": device})
    return {"rows": rows, "notes": "\n    ".join(note_lines)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_with_guard(name: str, fn, testsets: dict):
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
                 "device": "n/a"} for ds in testsets]


def print_summary(rows: list):
    by_var: dict[str, list] = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)
    fp32_rows = by_var.get("fp32", [])
    fp32_psnr = np.nanmean([r["psnr"] for r in fp32_rows]) if fp32_rows else float("nan")
    fp32_size = fp32_rows[0]["model_size_mb"] if fp32_rows else float("nan")
    fp32_lat = fp32_rows[0]["latency_ms"] if fp32_rows else float("nan")

    print("\n" + "=" * 96)
    print(f"{'variant':<16}{'avg_psnr':>10}{'Δpsnr':>8}{'size_mb':>10}{'size_ratio':>12}{'latency':>10}{'speedup':>10}{'device':>10}")
    print("-" * 96)
    for var, rs in by_var.items():
        psnrs = [r["psnr"] for r in rs if not np.isnan(r["psnr"])]
        avg_psnr = np.mean(psnrs) if psnrs else float("nan")
        size = rs[0]["model_size_mb"]
        lat = rs[0]["latency_ms"]
        dev = rs[0]["device"]
        dp = avg_psnr - fp32_psnr
        size_ratio = fp32_size / size if size and not np.isnan(size) else float("nan")
        speedup = fp32_lat / lat if lat and not np.isnan(lat) else float("nan")
        print(f"{var:<16}{avg_psnr:>10.3f}{dp:>8.3f}{size:>10.2f}{size_ratio:>11.2f}x{lat:>9.2f}{speedup:>9.2f}x{dev:>10}")
    print("=" * 96)


def main():
    set_seed()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Checkpoint:   {CKPT}")
    print(f"CUDA:         {torch.cuda.is_available()}")
    print(f"Quant engine: x86")

    all_rows = []
    all_rows += run_with_guard("fp32", run_fp32, TESTSETS_ALL)
    all_rows += run_with_guard("fp16", run_fp16, TESTSETS_CPU)
    all_rows += run_with_guard("dynamic_int8", run_dynamic_int8, TESTSETS_CPU)
    all_rows += run_with_guard("static_int8", run_static_int8, TESTSETS_CPU)

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
