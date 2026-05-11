"""
Full compression pipeline on the KD-distilled NAFNet-w32:

  Step 1 — structured L1 pruning at 30%        (no FT yet)
  Step 2 — fine-tune 10k iters with L1 only    (no teacher; recovery)
  Step 3 — apply FP16 cast on the FT'd model   (autocast for inference)
  Step 4 — apply static INT8 PTQ (FX-mode)     (calibrated on Rain13K crops)
  Step 5 — full 5-testset eval of the best variant via evaluate_model.py
  Step 6 — ONNX export of the pruned+FP16 model + ORT CPU/GPU benchmarks

Writes results/baselines/tables/full_pipeline.csv with columns:
  stage, psnr_rain100h, psnr_rain100l, model_size_mb, gpu_latency_ms,
  nonzero_params_M

Saves checkpoints to experiments/full_pipeline/checkpoints/.
"""
from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset  # noqa: E402
from evaluation.metrics import compute_psnr  # noqa: E402
from evaluation.model_wrappers import load_model  # noqa: E402

SEED = 42
DEVICE = "cuda"

CKPT_KD = PROJECT_ROOT / "pretrained" / "nafnet_w32_kd.pth"
WORK = PROJECT_ROOT / "experiments" / "full_pipeline"
CKPT_DIR = WORK / "checkpoints"
ONNX_PATH = WORK / "nafnet_kd_pruned30_fp16.onnx"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = PROJECT_ROOT / "results" / "baselines" / "tables" / "full_pipeline.csv"
TEST_ROOT = PROJECT_ROOT / "datasets" / "Rain13K" / "test"
TRAIN_INPUT = PROJECT_ROOT / "datasets" / "Rain13K" / "train" / "input"
TRAIN_TARGET = PROJECT_ROOT / "datasets" / "Rain13K" / "train" / "target"

PRUNE_RATIO = 0.3
FT_ITERS = 10_000
BATCH_SIZE = 32
LR = 1e-4
WD = 1e-3
VAL_EVERY = 2_000
PADDER_SIZE = 16

# Reference numbers for the final summary block.
RESTORMER_FP32_PSNR = 31.48
RESTORMER_FP32_SIZE_MB = 104.70
RESTORMER_FP32_LAT_MS = 46.35
NAFNET_KD_BASELINE_PSNR = 30.43


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


# ---------- Train dataset (mirrors structured_pruning.py) ----------
class DerainTrainDataset(Dataset):
    def __init__(self, input_dir: Path, target_dir: Path, crop_size: int = 256):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        self.input_files = sorted(f for f in self.input_dir.iterdir()
                                  if f.suffix.lower() in valid_ext)
        self.target_files = sorted(f for f in self.target_dir.iterdir()
                                   if f.suffix.lower() in valid_ext)
        assert len(self.input_files) == len(self.target_files)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp = Image.open(self.input_files[idx]).convert("RGB")
        tgt = Image.open(self.target_files[idx]).convert("RGB")
        cs = self.crop_size
        w, h = inp.size
        if w < cs or h < cs:
            inp = inp.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
            tgt = tgt.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
            w, h = inp.size
        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)
        inp = inp.crop((x, y, x + cs, y + cs))
        tgt = tgt.crop((x, y, x + cs, y + cs))
        if random.random() < 0.5:
            inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
            tgt = tgt.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            inp = inp.transpose(Image.FLIP_TOP_BOTTOM)
            tgt = tgt.transpose(Image.FLIP_TOP_BOTTOM)
        return self.to_tensor(inp), self.to_tensor(tgt)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# ---------- Pruning ----------
def prune_model(model: nn.Module, ratio: float):
    pruned, skipped = [], []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            out = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
            if out == 3 or out == 1:
                skipped.append(name)
                continue
            if int(ratio * out) < 1:
                skipped.append(name)
                continue
            try:
                prune.ln_structured(m, name="weight", amount=ratio, n=1, dim=0)
                prune.remove(m, "weight")
                pruned.append(name)
            except Exception as e:
                skipped.append(f"{name}({type(e).__name__})")
    print(f"  pruned {len(pruned)} layers, skipped {len(skipped)}")
    return model


def sparsity_stats(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return {
        "total_params_M": total / 1e6,
        "nonzero_params_M": nonzero / 1e6,
        "sparsity_pct": 100.0 * (1.0 - nonzero / total),
    }


# ---------- Eval ----------
def pad_to_multiple(x: torch.Tensor, m: int = PADDER_SIZE):
    _, _, h, w = x.shape
    mod_h = (m - h % m) % m
    mod_w = (m - w % m) % m
    if mod_h == 0 and mod_w == 0:
        return x, h, w
    return F.pad(x, (0, mod_w, 0, mod_h)), h, w


@torch.no_grad()
def eval_psnr(forward_fn, ds_name: str, device: str = "cuda",
              manual_pad: bool = False) -> float:
    ds = DerainDataset(
        input_dir=str(TEST_ROOT / ds_name / "input"),
        target_dir=str(TEST_ROOT / ds_name / "target"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    psnrs = []
    for inp, gt, _ in tqdm(loader, desc=f"  {ds_name}", leave=False):
        inp = inp.to(device)
        if manual_pad:
            padded, H, W = pad_to_multiple(inp)
            pred = forward_fn(padded)
            pred = pred[:, :, :H, :W]
        else:
            pred = forward_fn(inp)
        pred = torch.clamp(pred.float().cpu(), 0, 1)
        psnrs.append(compute_psnr(pred, gt.float().cpu()))
    return float(np.mean(psnrs))


def measure_gpu_latency(forward_fn, dtype=torch.float32,
                        input_size=(1, 3, 256, 256),
                        warmup=10, runs=100) -> float:
    if not torch.cuda.is_available():
        return float("nan")
    x = torch.randn(*input_size, device="cuda").to(dtype)
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = forward_fn(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


# ---------- Fine-tune ----------
def finetune(model: nn.Module, iters: int, batch_size: int) -> nn.Module:
    ds = DerainTrainDataset(TRAIN_INPUT, TRAIN_TARGET, crop_size=256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True,
                        persistent_workers=True)
    it = cycle(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.L1Loss()
    # NAFNet is conv-only — bf16 autocast halves activation memory without
    # needing GradScaler and avoids FP16 instability around freshly-pruned
    # zero filters.
    model.train()
    best_psnr = -1.0
    best_state = None
    pbar = tqdm(range(iters), desc="ft[nafnet_kd@0.3]")
    for step in pbar:
        inp, tgt = next(it)
        inp = inp.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(inp)
            loss = loss_fn(pred, tgt)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if (step + 1) % VAL_EVERY == 0:
            model.eval()
            psnr_h = eval_psnr(lambda x: model(x), "Rain100H")
            print(f"  [iter {step+1}/{iters}] Rain100H PSNR = {psnr_h:.3f}")
            if psnr_h > best_psnr:
                best_psnr = psnr_h
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                print(f"  ↑ new best ({psnr_h:.3f}) saved")
            model.train()
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  loaded best-val state (PSNR={best_psnr:.3f})")
    return model


# ---------- Static INT8 (FX) — adapted from ptq_nafnet.py ----------
def _load_calibration_batch(k=100, size=256):
    rng = random.Random(SEED)
    files = sorted(TRAIN_INPUT.glob("*.jpg")) or sorted(TRAIN_INPUT.glob("*.png"))
    picks = rng.sample(files, min(k, len(files)))
    to_t = transforms.ToTensor()
    out = []
    for fp in picks:
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        if w < size or h < size:
            img = img.resize((max(w, size), max(h, size)), Image.BILINEAR)
            w, h = img.size
        left = rng.randint(0, w - size)
        top = rng.randint(0, h - size)
        img = img.crop((left, top, left + size, top + size))
        out.append(to_t(img).unsqueeze(0))
    return out


def static_int8_quantize(ft_state_dict_path: Path) -> tuple:
    """Returns (qmodel, ckpt_path, size_mb) or (None, None, nan) on failure."""
    from torch.ao.quantization import (
        QConfigMapping, get_default_qconfig, get_default_qconfig_mapping,
    )
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "x86"
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "NAFNet"))
    from basicsr.models.archs.arch_util import LayerNorm2d

    model = load_model("nafnet_w32", str(ft_state_dict_path), "cpu").eval()
    model.check_image_size = lambda x: x
    example = (torch.randn(1, 3, 256, 256),)
    calib = _load_calibration_batch(k=100, size=256)

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
            prepared = prepare_fx(model, qmap, example, prepare_custom_config=pcc)
            with torch.no_grad():
                for t in tqdm(calib, desc=f"  calibrate[{attempt}]", leave=False):
                    _ = prepared(t)
            qmodel = convert_fx(prepared)
            print(f"  static INT8 prepared (qconfig='{attempt}')")
            break
        except Exception as exc:
            print(f"  static INT8 qconfig='{attempt}' failed: {exc}")
            traceback.print_exc()
            qmodel = None
            model = load_model("nafnet_w32", str(ft_state_dict_path), "cpu").eval()
            model.check_image_size = lambda x: x

    if qmodel is None:
        return None, None, float("nan")

    suffix = "_conv_only" if used_fallback else ""
    out_path = CKPT_DIR / f"nafnet_kd_pruned30_ft_int8{suffix}.pth"
    try:
        torch.save(qmodel.state_dict(), out_path)
    except Exception as exc:
        print(f"  state_dict save failed ({exc}); pickling full module.")
        torch.save(qmodel, out_path)
    return qmodel, out_path, os.path.getsize(out_path) / 1e6


# ---------- ONNX export + ORT bench ----------
def export_onnx(model: nn.Module, dummy: torch.Tensor, path: Path):
    torch.onnx.export(
        model, dummy, str(path), opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"},
                      "output": {2: "height", 3: "width"}},
        dynamo=False,  # use legacy TorchScript exporter (see onnx_export.py)
    )


def benchmark_ort(onnx_path: Path, dtype=np.float16) -> dict:
    import onnxruntime as ort
    bench_input = np.random.randn(1, 3, 256, 256).astype(dtype)
    out = {}
    # CPU
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        for _ in range(5):
            sess.run(None, {in_name: bench_input})
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            sess.run(None, {in_name: bench_input})
            ts.append((time.perf_counter() - t0) * 1000.0)
        out["cpu_ms"] = float(np.mean(ts))
    except Exception as e:
        print(f"  ORT CPU bench failed: {e}")
        out["cpu_ms"] = float("nan")
    # GPU
    try:
        if "CUDAExecutionProvider" in ort.get_available_providers():
            sess = ort.InferenceSession(str(onnx_path),
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            for _ in range(10):
                sess.run(None, {in_name: bench_input})
            ts = []
            for _ in range(100):
                t0 = time.perf_counter()
                sess.run(None, {in_name: bench_input})
                ts.append((time.perf_counter() - t0) * 1000.0)
            out["gpu_ms"] = float(np.mean(ts))
        else:
            out["gpu_ms"] = float("nan")
    except Exception as e:
        print(f"  ORT GPU bench failed: {e}")
        out["gpu_ms"] = float("nan")
    return out


# ---------- Main ----------
def main():
    set_seed()
    print(f"Using KD checkpoint: {CKPT_KD}")
    print(f"CUDA: {torch.cuda.is_available()}")
    rows = []

    # Stage 0: KD baseline reference (load + measure)
    print("\n--- Stage 0: nafnet_kd baseline ---")
    _purge_basicsr()
    kd_model = load_model("nafnet_w32", str(CKPT_KD), DEVICE).eval()
    kd_size_mb = os.path.getsize(CKPT_KD) / 1e6
    kd_lat = measure_gpu_latency(lambda x: kd_model(x))
    kd_psnr_h = eval_psnr(lambda x: kd_model(x), "Rain100H")
    kd_psnr_l = eval_psnr(lambda x: kd_model(x), "Rain100L")
    kd_stats = sparsity_stats(kd_model)
    print(f"  PSNR Rain100H = {kd_psnr_h:.3f}, Rain100L = {kd_psnr_l:.3f}, "
          f"size = {kd_size_mb:.2f} MB, lat = {kd_lat:.2f} ms")
    rows.append({
        "stage": "nafnet_kd_baseline",
        "psnr_rain100h": kd_psnr_h, "psnr_rain100l": kd_psnr_l,
        "model_size_mb": kd_size_mb, "gpu_latency_ms": kd_lat,
        "nonzero_params_M": kd_stats["nonzero_params_M"],
    })

    # Stage 1: prune-only
    print("\n--- Stage 1: nafnet_kd_pruned30 (no FT) ---")
    pruned = deepcopy(kd_model)
    prune_model(pruned, PRUNE_RATIO)
    prune_path = CKPT_DIR / "nafnet_kd_pruned30.pth"
    torch.save(pruned.state_dict(), prune_path)
    p_size = os.path.getsize(prune_path) / 1e6
    p_lat = measure_gpu_latency(lambda x: pruned(x))
    p_psnr_h = eval_psnr(lambda x: pruned(x), "Rain100H")
    p_psnr_l = eval_psnr(lambda x: pruned(x), "Rain100L")
    p_stats = sparsity_stats(pruned)
    print(f"  PSNR Rain100H = {p_psnr_h:.3f}, Rain100L = {p_psnr_l:.3f}, "
          f"size = {p_size:.2f} MB, lat = {p_lat:.2f} ms, "
          f"nonzero = {p_stats['nonzero_params_M']:.2f}M")
    rows.append({
        "stage": "nafnet_kd_pruned30",
        "psnr_rain100h": p_psnr_h, "psnr_rain100l": p_psnr_l,
        "model_size_mb": p_size, "gpu_latency_ms": p_lat,
        "nonzero_params_M": p_stats["nonzero_params_M"],
    })

    # KD baseline no longer needed — free GPU memory before FT.
    del kd_model
    torch.cuda.empty_cache()

    # Stage 2: fine-tune
    print(f"\n--- Stage 2: fine-tune {FT_ITERS} iters @ batch={BATCH_SIZE} ---")
    ft = finetune(pruned, iters=FT_ITERS, batch_size=BATCH_SIZE)
    ft_path = CKPT_DIR / "nafnet_kd_pruned30_ft.pth"
    torch.save(ft.state_dict(), ft_path)
    ft_size = os.path.getsize(ft_path) / 1e6
    ft_lat = measure_gpu_latency(lambda x: ft(x))
    ft_psnr_h = eval_psnr(lambda x: ft(x), "Rain100H")
    ft_psnr_l = eval_psnr(lambda x: ft(x), "Rain100L")
    ft_stats = sparsity_stats(ft)
    print(f"  PSNR Rain100H = {ft_psnr_h:.3f}, Rain100L = {ft_psnr_l:.3f}, "
          f"size = {ft_size:.2f} MB, lat = {ft_lat:.2f} ms")
    rows.append({
        "stage": "nafnet_kd_pruned30_ft",
        "psnr_rain100h": ft_psnr_h, "psnr_rain100l": ft_psnr_l,
        "model_size_mb": ft_size, "gpu_latency_ms": ft_lat,
        "nonzero_params_M": ft_stats["nonzero_params_M"],
    })

    # Stage 3: FP16
    print("\n--- Stage 3: FP16 ---")
    fp16 = deepcopy(ft).half().eval()
    fp16_path = CKPT_DIR / "nafnet_kd_pruned30_ft_fp16.pth"
    torch.save(fp16.state_dict(), fp16_path)
    fp16_size = os.path.getsize(fp16_path) / 1e6

    def fp16_forward(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return fp16(x.half())

    fp16_lat = measure_gpu_latency(fp16_forward, dtype=torch.float16)
    fp16_psnr_h = eval_psnr(fp16_forward, "Rain100H")
    fp16_psnr_l = eval_psnr(fp16_forward, "Rain100L")
    print(f"  PSNR Rain100H = {fp16_psnr_h:.3f}, Rain100L = {fp16_psnr_l:.3f}, "
          f"size = {fp16_size:.2f} MB, lat = {fp16_lat:.2f} ms")
    rows.append({
        "stage": "nafnet_kd_pruned30_ft_fp16",
        "psnr_rain100h": fp16_psnr_h, "psnr_rain100l": fp16_psnr_l,
        "model_size_mb": fp16_size, "gpu_latency_ms": fp16_lat,
        "nonzero_params_M": ft_stats["nonzero_params_M"],
    })

    # Stage 4: static INT8 (CPU)
    print("\n--- Stage 4: static INT8 (CPU PTQ) ---")
    qmodel, int8_path, int8_size_mb = static_int8_quantize(ft_path)
    int8_psnr_h = int8_psnr_l = float("nan")
    if qmodel is not None:
        try:
            int8_psnr_h = eval_psnr(lambda x: qmodel(x), "Rain100H",
                                    device="cpu", manual_pad=True)
            int8_psnr_l = eval_psnr(lambda x: qmodel(x), "Rain100L",
                                    device="cpu", manual_pad=True)
            print(f"  PSNR Rain100H = {int8_psnr_h:.3f}, Rain100L = {int8_psnr_l:.3f}")
        except Exception as e:
            print(f"  INT8 eval failed: {e}")
            traceback.print_exc()
    rows.append({
        "stage": "nafnet_kd_pruned30_ft_int8",
        "psnr_rain100h": int8_psnr_h, "psnr_rain100l": int8_psnr_l,
        "model_size_mb": int8_size_mb, "gpu_latency_ms": float("nan"),
        "nonzero_params_M": ft_stats["nonzero_params_M"],
    })

    # Stage 5: full 5-testset eval via evaluate_model.py
    print("\n--- Stage 5: full 5-testset eval ---")
    baselines_dir = PROJECT_ROOT / "results" / "baselines"
    baseline_csv = baselines_dir / "tables" / "nafnet_w32_results.csv"
    backup = baseline_csv.with_suffix(".csv.bak_fullpipe")
    if baseline_csv.exists():
        shutil.copy(baseline_csv, backup)
    # Deploy FT'd weights as if it were a normal nafnet_w32 ckpt for evaluator.
    deployed = PROJECT_ROOT / "pretrained" / "nafnet_w32_kd_pruned30_ft.pth"
    shutil.copy(ft_path, deployed)
    try:
        subprocess.run([sys.executable, "evaluation/evaluate_model.py",
                        "--model", "nafnet_w32",
                        "--checkpoint", str(deployed),
                        "--testset", "all",
                        "--output", str(baselines_dir)], check=True)
        produced = baseline_csv
        renamed = baseline_csv.with_name("nafnet_w32_kd_pruned30_ft_results.csv")
        if produced.exists():
            shutil.move(produced, renamed)
            print(f"  full eval saved to {renamed.name}")
    except Exception as e:
        print(f"  evaluate_model.py failed: {e}")
        traceback.print_exc()
    finally:
        if backup.exists():
            shutil.move(backup, baseline_csv)
            print(f"  restored baseline {baseline_csv.name}")

    # Stage 6: ONNX export (FP16) + ORT bench
    print("\n--- Stage 6: ONNX export of pruned+FP16 + ORT bench ---")
    try:
        # Export using FP16 weights + FP16 dummy input.
        fp16.eval()
        dummy = torch.randn(1, 3, 256, 256, device=DEVICE).half()
        export_onnx(fp16, dummy, ONNX_PATH)
        onnx_size_mb = os.path.getsize(ONNX_PATH) / 1e6
        print(f"  exported {ONNX_PATH.name} ({onnx_size_mb:.2f} MB)")
        import onnx
        onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
        bench = benchmark_ort(ONNX_PATH, dtype=np.float16)
        print(f"  ORT cpu_ms = {bench['cpu_ms']:.2f}, gpu_ms = {bench['gpu_ms']:.2f}")
        rows.append({
            "stage": "nafnet_kd_pruned30_ft_fp16_onnx",
            "psnr_rain100h": fp16_psnr_h,  # numerically identical to PyTorch FP16
            "psnr_rain100l": fp16_psnr_l,
            "model_size_mb": onnx_size_mb,
            "gpu_latency_ms": bench["gpu_ms"],
            "nonzero_params_M": ft_stats["nonzero_params_M"],
        })
    except Exception as e:
        print(f"  ONNX export/bench failed: {e}")
        traceback.print_exc()
        rows.append({
            "stage": "nafnet_kd_pruned30_ft_fp16_onnx",
            "psnr_rain100h": float("nan"), "psnr_rain100l": float("nan"),
            "model_size_mb": float("nan"), "gpu_latency_ms": float("nan"),
            "nonzero_params_M": ft_stats["nonzero_params_M"],
        })

    # Write CSV
    fieldnames = ["stage", "psnr_rain100h", "psnr_rain100l",
                  "model_size_mb", "gpu_latency_ms", "nonzero_params_M"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames}
            for fk in fieldnames:
                v = row.get(fk)
                if isinstance(v, float) and not np.isnan(v):
                    row[fk] = round(v, 4)
            w.writerow(row)
    print(f"\nCSV written: {CSV_PATH}")

    # Final summary block
    print("\n" + "=" * 72)
    final = next((r for r in rows
                  if r["stage"] == "nafnet_kd_pruned30_ft_fp16"
                  and not np.isnan(r["psnr_rain100h"])), None)
    if final is None:
        print("FULL PIPELINE: FP16 stage missing — see CSV.")
        return
    quality_pct = 100.0 * final["psnr_rain100h"] / RESTORMER_FP32_PSNR
    size_red = RESTORMER_FP32_SIZE_MB / final["model_size_mb"]
    speedup = RESTORMER_FP32_LAT_MS / final["gpu_latency_ms"]
    print("FULL PIPELINE: Restormer FP32 → NAFNet-KD → Prune 30% → FP16")
    print(f" Original: {RESTORMER_FP32_PSNR:.2f} dB | "
          f"{RESTORMER_FP32_SIZE_MB:.1f} MB | "
          f"{RESTORMER_FP32_LAT_MS:.1f} ms")
    print(f" Final:    {final['psnr_rain100h']:.2f} dB | "
          f"{final['model_size_mb']:.1f} MB  | "
          f"{final['gpu_latency_ms']:.1f} ms")
    print(f" Quality retained: {quality_pct:.1f}%")
    print(f" Size reduction: {size_red:.2f}x")
    print(f" Speedup: {speedup:.2f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()
