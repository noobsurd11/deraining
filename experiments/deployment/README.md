# ONNX Export and Benchmarking

**Status:** Complete.

## What

Export deraining models to ONNX format and benchmark inference latency using ONNX Runtime
(ORT) on both CPU and GPU, compared to native PyTorch inference.

## Why

ONNX provides a framework-agnostic deployment path with potential latency improvements
through ORT's graph optimizations (operator fusion, memory planning). This experiment
measures whether ORT actually improves throughput for each architecture.

## Method

1. Export PyTorch models to ONNX via `torch.onnx.export` (opset 17, dynamic batch)
2. Optimize with ORT graph optimizations (level 99)
3. Benchmark: 100 warmup + 100 timed iterations, 256x256 input patches
4. Produce FP16 ONNX variants for size reduction

## Scripts

| File | Purpose |
|---|---|
| `onnx_export.py` | Export, optimize, and benchmark all models |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
python onnx_export.py
```

## Results

### Model Size

| Model | PyTorch FP32 | ONNX FP16 |
|---|---|---|
| NAFNet-w32 | 116.8 MB | 59 MB |
| Restormer | 105.2 MB | 54.9 MB |
| DRSformer | 268 MB | 138 MB |

### Inference Latency (256x256, GPU)

| Model | PyTorch GPU | ORT GPU | Speedup |
|---|---|---|---|
| NAFNet-w32 | 11.3 ms | **8.0 ms** | 1.41x |
| Restormer | **46.4 ms** | 103.8 ms | 0.45x (slower) |
| DRSformer | **85.0 ms** | 175.2 ms | 0.49x (slower) |

### Key Observations

- **NAFNet benefits from ORT** -- pure convolution ops fuse well under ORT's graph optimizer.
- **Restormer and DRSformer are slower in ORT** -- attention ops (multi-head self-attention, softmax, reshape) do not fuse efficiently and incur overhead from suboptimal ONNX subgraph partitioning.

Results: `results/baselines/tables/onnx_benchmark.csv`

## Lessons Learned

1. ONNX/ORT is not universally faster -- transformer-based models can be 2x slower due to poor attention op fusion.
2. For convolution-heavy architectures like NAFNet, ORT provides a meaningful 1.4x GPU speedup.
3. FP16 ONNX export is straightforward and halves model size with no quality impact.
4. Deployment strategy should be architecture-aware: ORT for CNNs, native PyTorch (or TensorRT) for transformers.
