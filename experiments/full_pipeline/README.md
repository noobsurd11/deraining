# Full Compression Pipeline

**Status:** Complete, but pruning stage was ineffective (uses buggy v1 pruning).

## What

End-to-end compression pipeline applied to NAFNet-KD: unstructured pruning (30%) followed
by fine-tuning, FP16 conversion, INT8 quantization, and ONNX export.

## Why

Demonstrate a multi-stage compression pipeline combining pruning, quantization, and format
conversion. Intended to show cumulative compression gains.

## Pipeline Stages

```
NAFNet-KD (FP32) --> Prune 30% --> Fine-tune --> FP16 --> INT8 --> ONNX FP16
```

## Known Issue

This pipeline uses the v1 unstructured pruning approach (`prune.remove()` before
fine-tuning), which allows full weight regrowth during SGD. The pruning stage contributed
**zero actual compression** -- see `experiments/pruning/README.md` for details.

## Scripts

| File | Purpose |
|---|---|
| `compress_nafnet_kd.py` | Full pipeline execution |
| `plot_pipeline.py` | Stage-by-stage results visualization |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
python compress_nafnet_kd.py
```

## Results (Rain100H)

| Stage | PSNR (dB) | Size | Actual Sparsity | Notes |
|---|---|---|---|---|
| NAFNet-KD baseline | 30.43 | 116.8 MB | 0% | Starting point |
| + Prune 30% + FT | 30.39 | 116.8 MB | **0%** | Weights regrew |
| + FP16 | 30.39 | 58.6 MB | 0% | Lossless size halving |
| + INT8 | 29.32 | 31.9 MB | 0% | ~1 dB quality drop |
| + ONNX FP16 | 29.32* | 59 MB | 0% | 8.0 ms ORT GPU |

*INT8 was applied before ONNX in the pipeline; ONNX FP16 is a separate branch.

Best practical result: **FP16 ONNX at 59 MB, 8.0 ms GPU latency, ~30.4 dB** (skipping the
ineffective pruning and quality-costly INT8 stages).

Results: `results/baselines/tables/full_pipeline.csv`

## Lessons Learned

1. The pruning stage was a no-op due to the `prune.remove()` bug -- pipeline results reflect only quantization and format conversion.
2. FP16 is the most effective single compression step: 2x size reduction with no quality loss.
3. INT8 adds further size reduction but at a steep quality cost (~1 dB).
4. A corrected pipeline using structural pruning (v3) + FP16 would be the recommended approach.
