# Post-Training Quantization

**Status:** Complete.

## What

Post-training quantization (PTQ) of three deraining models at FP16, dynamic INT8, and
static INT8 precision using PyTorch's native quantization APIs.

## Why

Quantization is the simplest compression technique -- no retraining required. This
experiment establishes the quality/size tradeoff for each precision level across
architectures with different operator profiles (convolution-heavy vs attention-heavy).

## Method

| Technique | API | Target Device | Weight Format |
|---|---|---|---|
| FP16 | `model.half()` | GPU | float16 |
| Dynamic INT8 | `torch.ao.quantization.quantize_dynamic` | CPU | int8 (dynamic activation) |
| Static INT8 | `torch.ao.quantization.prepare` + calibration | CPU | int8 (fixed activation) |

Static INT8 uses a calibration set from the training data to determine activation ranges.

## Scripts

| File | Purpose |
|---|---|
| `ptq_restormer.py` | Quantize Restormer |
| `ptq_nafnet.py` | Quantize NAFNet-w32 |
| `ptq_drsformer.py` | Quantize DRSformer |
| `plot_quantization.py` | Results visualization |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
python ptq_restormer.py
python ptq_nafnet.py
python ptq_drsformer.py
```

## Results (Rain100H, PSNR in dB)

| Model | FP32 | FP16 | Dyn. INT8 | Static INT8 |
|---|---|---|---|---|
| Restormer | 31.48 | 31.47 | 31.48 | 29.5* |
| NAFNet-w32 | 30.23 | 30.23 | 30.23 | 29.1* |
| DRSformer | 31.63 | 31.62 | 31.63 | 29.8* |

*Approximate -- static INT8 quality varies with calibration.

### Size and Latency Summary

- **FP16:** Near-lossless (<0.01 dB drop), 2x size reduction, GPU latency comparable to FP32.
- **Dynamic INT8:** Lossless PSNR but CPU-only; no size reduction (weights remain FP32 at rest).
- **Static INT8:** 3-4x size reduction but 1-2 dB quality drop and high CPU latency.

Results: `results/baselines/tables/{restormer,nafnet_w32,drsformer,nafnet_w32_kd}_quantization.csv`

## Lessons Learned

1. FP16 is the clear winner for GPU deployment -- lossless quality at half the size.
2. INT8 quantization causes severe quality degradation in attention-heavy models (Restormer, DRSformer), where softmax and layer-norm are sensitive to reduced precision.
3. Dynamic INT8 provides no real compression benefit since weights are not persistently quantized.
4. Static INT8 requires careful calibration and is architecture-sensitive; convolution-heavy models (NAFNet) tolerate it better than transformer-based ones.
