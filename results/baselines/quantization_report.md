# Post-Training Quantization Report

Deraining models evaluated under FP32 / FP16 / INT8 post-training
quantization (PTQ). *(Draft — numbers filled in by the runner script;
see `experiments/quantization/ptq_{restormer,drsformer}.py`.)*

## Setup

| | |
|---|---|
| Models | Restormer (26.1 M params), DRSformer (33.7 M params) |
| Test sets | Rain100H, Rain100L, Test100 (all variants); plus Test1200 & Test2800 for FP32 reference (from baselines CSV) |
| Calibration set | 100 random 256×256 crops from Rain13K train inputs |
| Quantization API | `torch.ao.quantization` (`quantize_dynamic` for dynamic INT8, FX graph mode PTQ for static INT8) |
| Backend | `x86` (FBGEMM) |
| GPU | CUDA (for FP32 and FP16 inference) |
| CPU threads | 48 intra + 48 inter |
| Latency protocol | 10 warmup + 100 measured forwards on random 1×3×256×256 input (5 warmup + 20 measured on CPU) |
| Metrics | PSNR, SSIM, LPIPS (VGG, v0.1) |
| Seed | 42 |

## Variants

| Variant | Weights | Activations | Device | Notes |
|---|---|---|---|---|
| FP32 | fp32 | fp32 | GPU | Reference. |
| FP16 | fp16 | fp16 (autocast) | GPU | `model.half()` + `torch.autocast`; ~2× size reduction. |
| Dynamic INT8 | int8 (on-the-fly) | fp32 | CPU | `torch.ao.quantization.quantize_dynamic({nn.Linear, nn.Conv2d})`. Effectively a **no-op** on these models: neither Restormer nor DRSformer contains any `nn.Linear` layers, and the x86 dynamic-quant kernel only supports Linear — Conv2d falls through unchanged. |
| Static INT8 | int8 | int8 (PTQ, calibrated) | CPU | FX graph-mode PTQ (`prepare_fx` → calibrate → `convert_fx`) with `x86` qconfig. Only attempted for Restormer; skipped for DRSformer because its sparse-attention block uses `topk`+`scatter_`+`where(mask>0,…)` which FX cannot symbolically trace. |

## Restormer results

CSV: `results/baselines/tables/restormer_quantization.csv`.
Checkpoints: `experiments/quantization/checkpoints/restormer_{fp16,dynamic_int8,static_int8}.pth`.

*(Table auto-generated from the CSV — see `TABLES.md` section below.)*

### FX-tracing fix for static INT8

Restormer's `to_3d`/`to_4d` (plus the Q/K/V/out rearranges inside the
MDTA attention) are expressed as `einops.rearrange(...)`. `einops`
dispatches on runtime tensor type and rejects `torch.fx.proxy.Proxy`,
so `prepare_fx` failed with `RuntimeError: Tensor type unknown to einops`.

`ptq_restormer.patch_restormer_einops()` replaces those calls with
native-torch equivalents (`flatten` / `transpose` / `reshape`) at
runtime. Verified bit-equivalence vs. the einops original:

- FP32 forward on a random 256×256 image: `max |Δ| ≈ 7e-4`,
  `mean |Δ| ≈ 1e-4` — PSNR(native, einops) ≈ **80 dB** (i.e. the
  two are identical up to floating-point reassociation noise).
- Forward produces the same output shape and clamping behavior.

Handled einops patterns:

| Pattern | Native replacement |
|---|---|
| `b c h w -> b (h w) c` | `x.flatten(2).transpose(1,2).contiguous()` |
| `b (h w) c -> b c h w` | `x.transpose(1,2).reshape(b, c, h, w).contiguous()` |
| `b (head c) h w -> b head c (h w)` | `x.reshape(b, head, c, h*w)` |
| `b head c (h w) -> b (head c) h w` | `x.reshape(b, head*c, h, w)` |

## DRSformer results

CSV: `results/baselines/tables/drsformer_quantization.csv`.
Checkpoints: `experiments/quantization/checkpoints/drsformer_{fp16,dynamic_int8}.pth`.

Static INT8 is **not** attempted for DRSformer. The Top-K Sparse
Attention (TKSA) block builds four `torch.zeros` mask buffers per
forward, calls `torch.topk` and `mask.scatter_(…)`, then uses
`torch.where(mask > 0, attn, -inf)` — the `>` comparison and in-place
scatter on a freshly-allocated tensor are not symbolically traceable
in FX graph mode.

## TABLES (populated after run completes)

See `results/baselines/tables/{restormer,drsformer}_quantization.csv`
and plots under `results/baselines/plots/quantization_*`.

## Takeaways

*(Filled in after the run finishes.)*
