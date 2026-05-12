# Knowledge Distillation: Restormer to NAFNet

**Status:** Complete.

## What

Response-based knowledge distillation from Restormer (teacher) to NAFNet-w32 (student)
for single-image deraining.

## Why

NAFNet-w32 is 9.6x cheaper in GMACs than Restormer (16.1 vs 154.9) but 1.25 dB worse on
Rain100H (30.23 vs 31.48 dB). KD transfers the teacher's output distribution to close
this gap without increasing student complexity.

## Method

- **Loss:** `L = 1.0 * L1(pred, gt) + 0.5 * L1(pred, teacher_output)`
- **Optimizer:** AdamW, lr=2e-4, weight_decay=1e-3, betas=(0.9, 0.9)
- **Schedule:** Cosine decay to 1e-6 over 50K iterations
- **EMA:** decay=0.999
- **Teacher:** Restormer (frozen, pre-trained)

## Scripts

| File | Purpose |
|---|---|
| `distill_restormer_to_nafnet.py` | Main KD training script |
| `plot_distillation.py` | Training curves visualization |
| `quantize_kd.py` | Post-training quantization of the KD model |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
python distill_restormer_to_nafnet.py
```

## Results (Rain100H)

| Model | PSNR (dB) | Params | GMACs |
|---|---|---|---|
| Restormer (teacher) | 31.48 | 26.13M | 154.9 |
| NAFNet-w32 (scratch) | 30.23 | 29.16M | 16.1 |
| **NAFNet-w32 KD** | **30.43** | 29.16M | 16.1 |

KD gain: **+0.20 dB** over scratch training at zero additional inference cost.

### Quantization of KD Model

| Precision | PSNR (dB) | Model Size |
|---|---|---|
| FP32 | 30.43 | 116.8 MB |
| FP16 | 30.43 | 58.6 MB |
| INT8 | 29.60 | 31.9 MB |

Full 5-testset results: `results/baselines/tables/nafnet_w32_kd_results.csv`

## Lessons Learned

1. Response-based KD (output-level L1) yields a modest but consistent improvement (+0.20 dB).
2. The distillation loss weight (0.5) balances ground-truth fidelity with teacher guidance.
3. The KD model serves as a stronger starting point for downstream compression (pruning, quantization).
