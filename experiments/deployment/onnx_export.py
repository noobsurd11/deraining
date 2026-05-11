"""
ONNX export + ONNX Runtime benchmarking for deraining models.

For each (model, variant) pair the script:
  1. Loads the PyTorch model via evaluation/model_wrappers.py.
  2. Exports to ONNX (opset 17, dynamic H/W).
  3. Verifies the .onnx file with onnx.checker.
  4. Benchmarks ONNX Runtime on CPU (5 warmup, 50 timed runs).
  5. Benchmarks ONNX Runtime on CUDA (10 warmup, 100 timed runs) if available.
  6. Runs a 20-image PSNR sanity check on Rain100H using ORT.
  7. Records ONNX file size on disk and PyTorch GPU latency for comparison.

Failures are caught per-variant — a single export failure (DRSformer is the
likely culprit) does not stop the rest of the run.

Output: results/baselines/tables/onnx_benchmark.csv
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_psnr  # noqa: E402
from evaluation.model_wrappers import load_model  # noqa: E402

ONNX_DIR = PROJECT_ROOT / "experiments" / "deployment" / "onnx"
CSV_PATH = PROJECT_ROOT / "results" / "baselines" / "tables" / "onnx_benchmark.csv"
RAIN100H_IN = PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Rain100H" / "input"
RAIN100H_TGT = PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Rain100H" / "target"

ONNX_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
PADDER_SIZE = 16  # NAFNet+Restormer downsample 4x; 16 is safe.

# (variant_name, model_name, fp32_checkpoint, apply_half)
VARIANTS = [
    ("restormer_fp32",  "restormer",  PROJECT_ROOT / "pretrained" / "restormer_deraining.pth", False),
    ("restormer_fp16",  "restormer",  PROJECT_ROOT / "pretrained" / "restormer_deraining.pth", True),
    ("nafnet_fp32",     "nafnet_w32", PROJECT_ROOT / "pretrained" / "nafnet_w32_deraining.pth", False),
    ("nafnet_fp16",     "nafnet_w32", PROJECT_ROOT / "pretrained" / "nafnet_w32_deraining.pth", True),
    ("nafnet_kd",       "nafnet_w32", PROJECT_ROOT / "pretrained" / "nafnet_w32_kd.pth",        False),
    ("nafnet_kd_fp16",  "nafnet_w32", PROJECT_ROOT / "pretrained" / "nafnet_w32_kd.pth",        True),
    ("drsformer_fp32",  "drsformer",  PROJECT_ROOT / "pretrained" / "drsformer_deraining.pth",  False),
]


def _purge_basicsr():
    for k in [m for m in sys.modules if m == "basicsr" or m.startswith("basicsr.")]:
        del sys.modules[k]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_multiple(x: np.ndarray, m: int = PADDER_SIZE) -> tuple:
    """Pad (1,C,H,W) numpy array to multiples of m. Returns (padded, H, W)."""
    _, _, h, w = x.shape
    mod_h = (m - h % m) % m
    mod_w = (m - w % m) % m
    if mod_h == 0 and mod_w == 0:
        return x, h, w
    return np.pad(x, ((0, 0), (0, 0), (0, mod_h), (0, mod_w)), mode="reflect"), h, w


def export_to_onnx(model: torch.nn.Module, dummy: torch.Tensor, out_path: Path,
                   opset: int = 17) -> None:
    # dynamo=False forces the legacy TorchScript-based exporter, which is
    # well-supported for these convolutional/attention models. The default
    # in torch >= 2.11 is dynamo=True, which adds an onnxscript dep and has
    # gaps for complex ops (e.g. DRSformer's einops calls).
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=opset,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input":  {2: "height", 3: "width"},
                      "output": {2: "height", 3: "width"}},
        dynamo=False,
    )


def benchmark_ort(session, input_np: np.ndarray, warmup: int, runs: int) -> float:
    in_name = session.get_inputs()[0].name
    for _ in range(warmup):
        _ = session.run(None, {in_name: input_np})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(None, {in_name: input_np})
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def benchmark_pytorch_gpu(model: torch.nn.Module, dtype=torch.float32,
                          input_size=(1, 3, 256, 256), warmup=10, runs=100) -> float:
    if not torch.cuda.is_available():
        return float("nan")
    x = torch.randn(*input_size, device="cuda").to(dtype)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def quality_check_ort(session, dtype=np.float32, n_images: int = 20,
                      provider: str = "cpu") -> float:
    """Run first n_images of Rain100H through ORT and report mean PSNR."""
    inputs = sorted(p for p in RAIN100H_IN.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))[:n_images]
    targets = sorted(p for p in RAIN100H_TGT.iterdir()
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))[:n_images]
    to_t = transforms.ToTensor()
    in_name = session.get_inputs()[0].name
    psnrs = []
    for fp_in, fp_tgt in tqdm(list(zip(inputs, targets)),
                              desc=f"  qual[{provider}]", leave=False):
        inp = to_t(Image.open(fp_in).convert("RGB")).unsqueeze(0).numpy().astype(dtype)
        gt  = to_t(Image.open(fp_tgt).convert("RGB")).unsqueeze(0)
        padded, H, W = pad_to_multiple(inp)
        out = session.run(None, {in_name: padded})[0]
        out = out[:, :, :H, :W]
        pred = torch.from_numpy(out.astype(np.float32))
        pred = torch.clamp(pred, 0, 1)
        psnrs.append(compute_psnr(pred, gt))
    return float(np.mean(psnrs)) if psnrs else float("nan")


def run_one(variant: str, model_name: str, ckpt: Path, apply_half: bool) -> dict:
    print(f"\n=== {variant} ({model_name}, half={apply_half}) ===")
    row = {
        "model_variant": variant,
        "onnx_size_mb": float("nan"),
        "ort_cpu_latency_ms": float("nan"),
        "ort_gpu_latency_ms": float("nan"),
        "pytorch_gpu_latency_ms": float("nan"),
        "psnr_rain100h_20img": float("nan"),
        "export_success": False,
        "notes": "",
    }
    notes = []
    onnx_path = ONNX_DIR / f"{variant}.onnx"

    # 1. Load PyTorch model
    _purge_basicsr()
    try:
        model = load_model(model_name, str(ckpt), "cuda")
        if apply_half:
            model = model.half()
        model.eval()
    except Exception as e:
        notes.append(f"LOAD_FAIL: {type(e).__name__}: {e}")
        row["notes"] = " | ".join(notes)
        traceback.print_exc()
        return row

    # 2. Measure PyTorch GPU latency for comparison.
    try:
        py_dtype = torch.float16 if apply_half else torch.float32
        row["pytorch_gpu_latency_ms"] = benchmark_pytorch_gpu(model, dtype=py_dtype)
        print(f"  pytorch_gpu_latency: {row['pytorch_gpu_latency_ms']:.2f} ms")
    except Exception as e:
        notes.append(f"PT_LATENCY_FAIL: {e}")
        traceback.print_exc()

    # 3. Export to ONNX
    try:
        dummy_dtype = torch.float16 if apply_half else torch.float32
        dummy = torch.randn(1, 3, 256, 256, device="cuda").to(dummy_dtype)
        export_to_onnx(model, dummy, onnx_path)
        row["onnx_size_mb"] = os.path.getsize(onnx_path) / 1e6
        # Verify
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        row["export_success"] = True
        print(f"  exported + verified: {onnx_path.name} ({row['onnx_size_mb']:.2f} MB)")
    except Exception as e:
        notes.append(f"EXPORT_FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        row["notes"] = " | ".join(notes)
        del model
        torch.cuda.empty_cache()
        return row

    del model
    torch.cuda.empty_cache()

    # 4. Benchmark ORT CPU
    try:
        import onnxruntime as ort
        sess_cpu = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        np_dtype = np.float16 if apply_half else np.float32
        bench_input = np.random.randn(1, 3, 256, 256).astype(np_dtype)
        row["ort_cpu_latency_ms"] = benchmark_ort(sess_cpu, bench_input, warmup=5, runs=50)
        print(f"  ort_cpu_latency: {row['ort_cpu_latency_ms']:.2f} ms")
    except Exception as e:
        notes.append(f"ORT_CPU_FAIL: {e}")
        traceback.print_exc()

    # 5. Benchmark ORT GPU
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            sess_gpu = ort.InferenceSession(str(onnx_path),
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            np_dtype = np.float16 if apply_half else np.float32
            bench_input = np.random.randn(1, 3, 256, 256).astype(np_dtype)
            row["ort_gpu_latency_ms"] = benchmark_ort(sess_gpu, bench_input, warmup=10, runs=100)
            print(f"  ort_gpu_latency: {row['ort_gpu_latency_ms']:.2f} ms")
        else:
            notes.append("CUDA provider not available in onnxruntime.")
    except Exception as e:
        notes.append(f"ORT_GPU_FAIL: {e}")
        traceback.print_exc()

    # 6. Quality check on first 20 Rain100H images via ORT (CPU is fine; deterministic).
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        np_dtype = np.float16 if apply_half else np.float32
        row["psnr_rain100h_20img"] = quality_check_ort(sess, dtype=np_dtype, n_images=20)
        print(f"  psnr_rain100h_20img: {row['psnr_rain100h_20img']:.3f} dB")
    except Exception as e:
        notes.append(f"QUALITY_FAIL: {e}")
        traceback.print_exc()

    row["notes"] = " | ".join(notes)
    return row


def print_summary(rows: list):
    print("\n" + "=" * 110)
    print(f"{'variant':<20}{'size_MB':>10}{'ort_cpu':>10}{'ort_gpu':>10}{'pt_gpu':>10}"
          f"{'speedup':>10}{'psnr20':>10}{'ok':>6}")
    print("-" * 110)
    for r in rows:
        ort_g = r["ort_gpu_latency_ms"]
        pt_g = r["pytorch_gpu_latency_ms"]
        speedup = (pt_g / ort_g) if (ort_g and pt_g and not np.isnan(ort_g) and not np.isnan(pt_g) and ort_g > 0) else float("nan")
        print(
            f"{r['model_variant']:<20}"
            f"{r['onnx_size_mb']:>10.2f}"
            f"{r['ort_cpu_latency_ms']:>10.2f}"
            f"{r['ort_gpu_latency_ms']:>10.2f}"
            f"{r['pytorch_gpu_latency_ms']:>10.2f}"
            f"{speedup:>9.2f}x"
            f"{r['psnr_rain100h_20img']:>10.2f}"
            f"{str(r['export_success']):>6}"
        )
    print("=" * 110)


def main():
    set_seed()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA: {torch.cuda.is_available()}")
    import onnxruntime as ort
    print(f"ORT providers: {ort.get_available_providers()}")

    rows = []
    for variant, model_name, ckpt, apply_half in VARIANTS:
        try:
            rows.append(run_one(variant, model_name, ckpt, apply_half))
        except Exception as e:
            print(f"  RUN_FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({
                "model_variant": variant,
                "onnx_size_mb": float("nan"),
                "ort_cpu_latency_ms": float("nan"),
                "ort_gpu_latency_ms": float("nan"),
                "pytorch_gpu_latency_ms": float("nan"),
                "psnr_rain100h_20img": float("nan"),
                "export_success": False,
                "notes": f"RUN_FAIL: {e}",
            })

    fieldnames = ["model_variant", "onnx_size_mb", "ort_cpu_latency_ms",
                  "ort_gpu_latency_ms", "pytorch_gpu_latency_ms",
                  "psnr_rain100h_20img", "export_success", "notes"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row_out = dict(r)
            for fk in ("onnx_size_mb", "ort_cpu_latency_ms", "ort_gpu_latency_ms",
                       "pytorch_gpu_latency_ms", "psnr_rain100h_20img"):
                v = row_out.get(fk)
                if isinstance(v, float) and not np.isnan(v):
                    row_out[fk] = round(v, 4)
            w.writerow({k: row_out.get(k) for k in fieldnames})
    print(f"\nCSV written: {CSV_PATH}")
    print_summary(rows)


if __name__ == "__main__":
    main()
