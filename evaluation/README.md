# Evaluation Pipeline

Unified evaluation framework for deraining models across all 5 Rain13K test sets.

## Components

| File | Purpose |
|------|---------|
| `metrics.py` | PSNR/SSIM on Y-channel (4px border crop), LPIPS |
| `efficiency.py` | Parameter count, GMACs (fvcore), GPU latency, peak memory, model size |
| `dataset.py` | `DerainDataset` — paired input/target image loader |
| `model_wrappers.py` | Unified `load_model()` for Restormer, DRSformer, NAFNet-w32/w64, Diff-Mamba |
| `evaluate_model.py` | CLI entrypoint for single-model evaluation |

## Supported Models

```python
from evaluation.model_wrappers import load_model
model = load_model('restormer', 'pretrained/restormer_deraining.pth')
model = load_model('nafnet_w32', 'pretrained/nafnet_w32_kd.pth')
model = load_model('drsformer', 'pretrained/drsformer_deraining.pth')
```

## Usage

```bash
# Single model, all test sets
python evaluation/evaluate_model.py --model restormer \
    --checkpoint pretrained/restormer_deraining.pth --testset all

# Single test set
python evaluation/evaluate_model.py --model nafnet_w32 \
    --checkpoint pretrained/nafnet_w32_kd.pth --testset Rain100H
```

## Test Sets

| Dataset | Images | Description |
|---------|--------|-------------|
| Rain100H | 100 | Heavy synthetic rain streaks |
| Rain100L | 100 | Light synthetic rain streaks |
| Test100 | 98 | Mixed rain types |
| Test1200 | 1200 | Large-scale diverse test |
| Test2800 | 2800 | Largest test split |

## Important Notes

- Each model repo has its own `basicsr/` package. Only one can be loaded per Python process — `evaluate_model.py` loads exactly one model per invocation.
- Metrics match published papers: Y-channel conversion, 4-pixel border crop for PSNR/SSIM.
- Latency is measured with 100 warm-up + 100 timed iterations at 1x3x256x256.
