# Post-Training Quantization Report

**Project:** `/home/user/noob/deraining` — efficient image deraining for edge deployment
**Scope:** Restormer + DRSformer (NAFNet and Diff-Mamba pending)
**Date:** 2026-04-18 (session `8e919221...`)

## Goal

Compress the two transformer-family deraining baselines with post-training quantization (no retraining) and measure the quality / size / latency tradeoff on the standard test sets.

## Variants attempted

| Variant        | Precision  | Backend            | Scope                                        |
|----------------|------------|--------------------|----------------------------------------------|
| `fp32`         | FP32       | CUDA               | All 5 test sets (reference)                  |
| `fp16`         | FP16       | CUDA (`model.half()`) | Restormer/DRSformer on Rain100H/L/Test100 |
| `dynamic_int8` | INT8 weights, FP32 activations | CPU (`quantize_dynamic`)  | Rain100H/L/Test100                         |
| `static_int8`  | INT8 weights + activations  | CPU (FX graph mode PTQ, x86 backend, 100 Rain13K calibration crops) | Rain100H/L/Test100 — **Restormer only**   |

**Why the limited CPU-variant scope:** Test1200 (1200 imgs) + Test2800 (2800 imgs) at ~30 s/img on CPU with dynamic INT8 would each take >10 hours per model. The 3 smaller sets are sufficient to characterize the quality/size tradeoff.

**FX tracing workaround (Restormer only):** `Restormer`'s `to_3d`/`to_4d` helpers and attention use `einops.rearrange`, which `torch.fx` can't trace. The static-INT8 pipeline monkey-patches those to native `torch.flatten`/`transpose`/`reshape` before tracing.

## Results

### Restormer (FP32: 26.13 M params, 154.9 GMACs)

| Variant         | Rain100H | Rain100L | Test100 | Test1200 | Test2800 | Size (MB) | Latency (ms) | Device |
|-----------------|----------|----------|---------|----------|----------|-----------|--------------|--------|
| fp32            | 31.48    | 39.15    | 32.07   | 33.21    | 34.25    | 104.7     | 100.6        | CUDA   |
| fp16            | 31.48    | 39.15    | 32.07   | —        | —        | **52.4**  | **47.5**     | CUDA   |
| dynamic_int8    | 31.48    | 39.15    | 32.07   | —        | —        | 104.7     | 10 255       | CPU    |
| static_int8     | **29.47**| 36.61    | 31.00   | —        | —        | **29.4**  | 15 110       | CPU    |

*PSNR in dB on Y-channel with 4-px border crop.*

### DRSformer (Rain200H-trained; FP32: 33.66 M params, 243.0 GMACs)

| Variant         | Rain100H | Rain100L | Test100 | Test1200 | Test2800 | Size (MB) | Latency (ms) | Device |
|-----------------|----------|----------|---------|----------|----------|-----------|--------------|--------|
| fp32            | 33.87    | 40.50    | 24.03   | 30.01    | 30.15    | 135.0     | 238.4        | CUDA   |
| fp16            | 33.87    | 40.50    | 24.03   | —        | —        | **67.7**  | **149.4**    | CUDA   |
| dynamic_int8    | 33.87    | 40.50    | 24.03   | —        | —        | 135.0     | 16 728       | CPU    |
| static_int8     | *not run — see caveat* | | | | | | |                        |

### Summary (averages on the 3 common CPU-variant test sets — Rain100H, Rain100L, Test100)

| Model      | Variant       | Avg PSNR | ΔPSNR vs fp32 | Size ratio | Speedup |
|------------|---------------|---------:|--------------:|-----------:|--------:|
| Restormer  | fp32          | 34.23    |    0.00       | 1.00×      | 1.00× (CUDA) |
| Restormer  | fp16          | 34.23    |   **−0.00**   | **1.99×**  | **2.12×** (CUDA) |
| Restormer  | dynamic_int8  | 34.23    |   −0.00       | 1.00×      | 0.01× (CPU) |
| Restormer  | static_int8   | 32.36    |   −1.87       | **3.56×**  | 0.01× (CPU) |
| DRSformer  | fp32          | 32.80    |    0.00       | 1.00×      | 1.00× (CUDA) |
| DRSformer  | fp16          | 32.80    |   **−0.00**   | **1.99×**  | **1.60×** (CUDA) |
| DRSformer  | dynamic_int8  | 32.80    |   −0.00       | 1.00×      | 0.01× (CPU) |

## Key findings

1. **FP16 is a free lunch on GPU.** Both models shrink ~2× and run 1.6–2.1× faster with ΔPSNR ≈ 0 dB. For GPU edge deployment (Jetson-class hardware), this is the recommended default.
2. **Dynamic INT8 is a no-op here.** `quantize_dynamic` only INT8-ifies `nn.Linear` and `nn.LSTM`. Restormer is mostly Conv2d; DRSformer has *no* `nn.Linear` layers. The saved files even keep the same size because only Linear layer weights (if any) get packed. The dynamic INT8 checkpoints are on disk (same size as fp32) but provide no practical benefit and are ~100× slower because they run on CPU.
3. **Static INT8 (Restormer) buys the real compression — with a real quality cost.** 3.56× smaller on disk (29.4 MB vs 104.7 MB) but −1.87 dB average PSNR, driven by the −2 dB drop on Rain100H (heavy rain streaks, the hardest case). The CPU latency (15 s/img) is still dominated by Conv/attention ops that the FBGEMM backend can't fully lower — this is a deployment artifact, not a quality one.
4. **DRSformer static INT8 is blocked.** DRSformer's Mixture-of-Experts-like sparse attention plus nested `einops` ops didn't fit the FX-trace-and-quantize path that worked for Restormer. A torchao / `prepare_pt2e` port is the likely fix.

## Artifacts

```
experiments/quantization/
├── ptq_restormer.py                       # FP16 + dynamic + static INT8 pipeline
├── ptq_drsformer.py                       # FP16 + dynamic INT8 (static incomplete)
├── plot_quantization.py                   # (unused — no plots generated)
├── checkpoints/
│   ├── restormer_fp16.pth              (52.4 MB)
│   ├── restormer_dynamic_int8.pth     (104.7 MB)
│   ├── restormer_static_int8.pth       (29.4 MB)
│   ├── drsformer_fp16.pth              (67.7 MB)
│   └── drsformer_dynamic_int8.pth     (135.0 MB)
├── run.log                                # Restormer run
└── run_drsformer.log                      # DRSformer run

results/baselines/tables/
├── restormer_quantization.csv             # 14 rows (5×fp32, 3×fp16, 3×dyn, 3×static)
└── drsformer_quantization.csv             # 11 rows (5×fp32, 3×fp16, 3×dyn)
```

## Caveats

- **Deprecated APIs.** The scripts use `torch.ao.quantization.{quantize_dynamic, quantize_fx}` — the PyTorch runtime prints a migration warning pointing at `torchao` / `prepare_pt2e`. Nothing is broken, but any re-run on a newer PyTorch (≥2.12) should migrate.
- **Latency measurement asymmetry.** FP32/FP16 are measured on the same RTX PRO 4500 GPU; dynamic/static INT8 are CPU-only (no INT8 CUDA kernels for these ops in standard PyTorch). The speedup column is directly comparable only within the same device column.
- **Static INT8 calibration is small (100 Rain13K crops).** Larger calibration sets could reduce the Rain100H −2 dB drop. Not tested.
- **No edge-target artifact.** These are PyTorch INT8 checkpoints, not ONNX / TensorRT / CoreML. For actual edge deployment, an ONNX Runtime INT8 or TensorRT INT8 conversion is the next step.

## Gaps / next steps

1. **Fill DRSformer Test1200 + Test2800** for FP16 (GPU-cheap, should take ~15 min).
2. **Static INT8 for DRSformer** via `torchao.quantization.quantize_` (pt2e flow) — the FX-trace path we used for Restormer won't work.
3. **NAFNet PTQ** (all 4 variants, full 5 test sets). NAFNet finished training 2026-04-19 (`net_g_300000.pth`, Rain100H PSNR 30.23 dB). It's the cleanest candidate — pure Conv2d + SimpleGate + LayerNorm — and should quantize without FX patches.
4. **Diff-Mamba** deferred (no usable pretrained weight; diffusion sampler not wired up).
5. **Publication plots** — `plot_quantization.py` exists but wasn't run; add PSNR-vs-size and PSNR-vs-latency scatter plots for the report figure.
6. **ONNX Runtime / TensorRT INT8** as the actual edge-deployment artifact.
