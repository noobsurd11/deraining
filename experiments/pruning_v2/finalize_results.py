"""
Complete Steps 5-9 using the already-trained NAFNet-w22 checkpoint.
Fixes the FP16 latency measurement issue from the main script.
"""
import csv
import logging
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset
from evaluation.efficiency import count_gmacs, count_parameters, get_model_size_mb, measure_latency
from evaluation.metrics import compute_psnr

DEVICE = "cuda"
NEW_WIDTH = 22
WORK = PROJECT_ROOT / "experiments" / "pruning_v2"
CKPT_DIR = WORK / "checkpoints"
CSV_PATH = WORK / "results.csv"
TEST_ROOT = PROJECT_ROOT / "datasets" / "Rain13K" / "test"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(WORK / "run.log", mode="a"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def _purge_basicsr():
    for k in [m for m in sys.modules if m == "basicsr" or m.startswith("basicsr.")]:
        del sys.modules[k]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]


@torch.no_grad()
def eval_psnr(model, ds_name, device=DEVICE):
    ds = DerainDataset(str(TEST_ROOT / ds_name / "input"), str(TEST_ROOT / ds_name / "target"))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    psnrs = []
    for inp, gt, _ in tqdm(loader, desc=f"  eval {ds_name}", leave=False):
        inp = inp.to(device)
        pred = torch.clamp(model(inp), 0, 1)
        psnrs.append(compute_psnr(pred.cpu(), gt))
    return float(np.mean(psnrs))


@torch.no_grad()
def eval_psnr_fp16(model, ds_name, device=DEVICE):
    ds = DerainDataset(str(TEST_ROOT / ds_name / "input"), str(TEST_ROOT / ds_name / "target"))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    psnrs = []
    for inp, gt, _ in tqdm(loader, desc=f"  eval {ds_name}", leave=False):
        inp = inp.to(device).half()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(inp)
        pred = torch.clamp(pred.float().cpu(), 0, 1)
        psnrs.append(compute_psnr(pred, gt))
    return float(np.mean(psnrs))


def measure_latency_fp16(model, input_size=(1, 3, 256, 256), warmup=10, runs=100):
    dummy = torch.randn(*input_size, device=DEVICE, dtype=torch.float16)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def main():
    log.info("=" * 70)
    log.info("  FINALIZING RESULTS (Steps 5-9)")
    log.info("=" * 70)

    # Load NAFNet-w22
    _purge_basicsr()
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "NAFNet"))
    from basicsr.models.archs.NAFNet_arch import NAFNet

    ft_ckpt = CKPT_DIR / f"nafnet_w{NEW_WIDTH}_kd_ft.pth"
    student = NAFNet(img_channel=3, width=NEW_WIDTH, middle_blk_num=12,
                     enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    student.load_state_dict(torch.load(str(ft_ckpt), map_location="cpu", weights_only=False))
    student = student.to(DEVICE).eval()

    # Step 5: FP32 evaluation
    log.info("\n[Step 5] FP32 evaluation")
    psnr_h = eval_psnr(student, "Rain100H")
    psnr_l = eval_psnr(student, "Rain100L")
    params = count_parameters(student)
    gmacs = count_gmacs(student, device=DEVICE)
    lat = measure_latency(student, device=DEVICE)["mean_ms"]
    size_mb = get_model_size_mb(str(ft_ckpt))
    log.info("  Rain100H = %.2f dB, Rain100L = %.2f dB", psnr_h, psnr_l)
    log.info("  Params: %.2f M, GMACs: %.1f, Latency: %.1f ms, Size: %.1f MB",
             params, gmacs, lat, size_mb)

    # Also get original NAFNet-KD reference
    orig_params = 29.16
    orig_gmacs = 16.1
    orig_lat = 11.2

    rows = []
    rows.append({"stage": "restormer_fp32", "psnr_rain100h": 31.48, "psnr_rain100l": 39.15,
                 "model_size_mb": 104.70, "gpu_latency_ms": 46.35, "total_params_M": 26.13,
                 "nonzero_params_M": 26.13, "gmacs": 154.9, "onnx_latency_ms": float("nan")})
    rows.append({"stage": "nafnet_kd_w32", "psnr_rain100h": 30.43, "psnr_rain100l": 37.32,
                 "model_size_mb": 116.88, "gpu_latency_ms": orig_lat, "total_params_M": orig_params,
                 "nonzero_params_M": orig_params, "gmacs": orig_gmacs, "onnx_latency_ms": float("nan")})
    rows.append({"stage": "nafnet_kd_w32_pruned30_broken", "psnr_rain100h": 29.32, "psnr_rain100l": 35.27,
                 "model_size_mb": 116.88, "gpu_latency_ms": 11.24, "total_params_M": 29.16,
                 "nonzero_params_M": 29.16, "gmacs": 16.1, "onnx_latency_ms": 7.96})
    rows.append({"stage": f"nafnet_kd_w{NEW_WIDTH}_ft", "psnr_rain100h": psnr_h, "psnr_rain100l": psnr_l,
                 "model_size_mb": size_mb, "gpu_latency_ms": lat, "total_params_M": params,
                 "nonzero_params_M": params, "gmacs": gmacs, "onnx_latency_ms": float("nan")})

    # Step 6: FP16
    log.info("\n[Step 6] FP16 conversion + evaluation")
    model_fp16 = deepcopy(student).half().to(DEVICE).eval()
    psnr_h_fp16 = eval_psnr_fp16(model_fp16, "Rain100H")
    psnr_l_fp16 = eval_psnr_fp16(model_fp16, "Rain100L")
    fp16_lat = measure_latency_fp16(model_fp16)
    fp16_ckpt = CKPT_DIR / f"nafnet_w{NEW_WIDTH}_kd_ft_fp16.pth"
    torch.save(model_fp16.state_dict(), fp16_ckpt)
    fp16_size = get_model_size_mb(str(fp16_ckpt))
    log.info("  FP16 Rain100H = %.2f dB, Rain100L = %.2f dB", psnr_h_fp16, psnr_l_fp16)
    log.info("  FP16 Size: %.1f MB, Latency: %.1f ms", fp16_size, fp16_lat)

    rows.append({"stage": f"nafnet_kd_w{NEW_WIDTH}_ft_fp16", "psnr_rain100h": psnr_h_fp16,
                 "psnr_rain100l": psnr_l_fp16, "model_size_mb": fp16_size, "gpu_latency_ms": fp16_lat,
                 "total_params_M": params, "nonzero_params_M": params, "gmacs": gmacs,
                 "onnx_latency_ms": float("nan")})

    # Step 7: ONNX
    log.info("\n[Step 7] ONNX export + ORT benchmarking")
    onnx_path = WORK / f"nafnet_w{NEW_WIDTH}_kd_ft_fp16.onnx"
    onnx_lat_gpu = float("nan")
    onnx_size = float("nan")

    try:
        model_export = deepcopy(student).half().to(DEVICE).eval()
        dummy = torch.randn(1, 3, 256, 256, device=DEVICE, dtype=torch.float16)
        torch.onnx.export(
            model_export, dummy, str(onnx_path),
            opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
            dynamo=False,
        )
        onnx_size = os.path.getsize(onnx_path) / 1e6
        log.info("  ONNX exported: %s (%.1f MB)", onnx_path.name, onnx_size)

        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        log.info("  ONNX verified OK")

        import onnxruntime as ort
        try:
            sess = ort.InferenceSession(str(onnx_path),
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            inp_np = np.random.randn(1, 3, 256, 256).astype(np.float16)
            for _ in range(10):
                sess.run(None, {in_name: inp_np})
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                sess.run(None, {in_name: inp_np})
                times.append((time.perf_counter() - t0) * 1000)
            onnx_lat_gpu = float(np.mean(times))
            log.info("  ORT GPU latency: %.2f ms", onnx_lat_gpu)
        except Exception as e:
            log.warning("  ORT GPU benchmark failed: %s", e)
    except Exception as e:
        log.error("  ONNX export failed: %s", e)
        traceback.print_exc()

    rows.append({"stage": f"nafnet_kd_w{NEW_WIDTH}_ft_fp16_onnx", "psnr_rain100h": psnr_h_fp16,
                 "psnr_rain100l": psnr_l_fp16, "model_size_mb": onnx_size, "gpu_latency_ms": onnx_lat_gpu,
                 "total_params_M": params, "nonzero_params_M": params, "gmacs": gmacs,
                 "onnx_latency_ms": onnx_lat_gpu})

    # Step 8: Write CSV
    fields = ["stage", "psnr_rain100h", "psnr_rain100l", "model_size_mb",
              "gpu_latency_ms", "total_params_M", "nonzero_params_M", "gmacs", "onnx_latency_ms"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("\n  Saved: %s", CSV_PATH)

    # Step 9: Summary
    log.info("\n" + "=" * 70)
    log.info("  CORRECTED PIPELINE RESULTS")
    log.info("=" * 70)
    log.info("")
    log.info("  %-42s %8s %8s %8s %8s %8s",
             "Stage", "PSNR-H", "Size(MB)", "GMACs", "Lat(ms)", "Params(M)")
    log.info("  " + "-" * 90)
    for r in rows:
        log.info("  %-42s %8.2f %8.1f %8.1f %8.1f %8.2f",
                 r["stage"], r["psnr_rain100h"], r["model_size_mb"],
                 r["gmacs"], r["gpu_latency_ms"], r["total_params_M"])

    log.info("")
    log.info("  vs NAFNet-KD (w32):")
    log.info("    Quality retained: %.1f%%", 100 * psnr_h / 30.43)
    log.info("    Params reduction: %.1fx", orig_params / params)
    log.info("    Size reduction:   %.1fx", 116.88 / size_mb)
    log.info("    Compute (GMACs):  %.1fx", orig_gmacs / gmacs)
    log.info("")
    log.info("  vs OLD BROKEN pipeline (29.32 dB, 116.9 MB, same GMACs):")
    log.info("    Old pipeline had 0%% real sparsity — weights regrew")
    log.info("    New model is GENUINELY %.1fx smaller (%.1f MB vs 116.9 MB)",
             116.88 / size_mb, size_mb)
    log.info("    Compute genuinely reduced: %.1f GMACs vs 16.1 GMACs", gmacs)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
