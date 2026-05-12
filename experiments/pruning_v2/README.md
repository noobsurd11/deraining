# Manual Width Reduction (Physical Pruning Workaround)

**Status:** Complete. Superseded by `pruning_v3/` (automated structural pruning).

## What

Physical model compression by manually reducing NAFNet channel width from 32 to 22,
creating a smaller architecture (NAFNet-w22) and fine-tuning with knowledge distillation
from Restormer.

## Why

After the v1 pruning bug showed that unstructured pruning with `prune.remove()` does not
yield real compression, this experiment tests whether manually shrinking the architecture
and re-training can produce a genuinely smaller model.

## Method

1. Instantiate NAFNet with `width=22` (vs baseline `width=32`)
2. Fine-tune with KD loss: `L = 1.0 * L1(pred, gt) + 0.5 * L1(pred, teacher_out)`
3. Teacher: Restormer (frozen, FP16 inference)
4. Export to FP16 PyTorch and ONNX FP16

## Scripts

| File | Purpose |
|---|---|
| `pruned_finetune_physical.py` | Training with KD from Restormer |
| `verify_regrowth.py` | Sparsity verification (sanity check from v1) |
| `finalize_results.py` | Aggregate final metrics |
| `plot_corrected_pipeline.py` | Pipeline visualization |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
python pruned_finetune_physical.py
```

## Results (Rain100H)

| Variant | PSNR (dB) | Params | GMACs | Size |
|---|---|---|---|---|
| NAFNet-w32 (baseline) | 30.23 | 29.16M | 16.1 | 116.8 MB |
| NAFNet-w22 KD | 27.63 | 13.85M | 7.7 | 55.6 MB |
| NAFNet-w22 KD FP16 | 27.63 | 13.85M | 7.7 | 27.94 MB |
| NAFNet-w22 ONNX FP16 | 27.63 | 13.85M | 7.7 | 28.4 MB |

ONNX FP16 ORT GPU latency: **7.1 ms**

Full results: `experiments/pruning_v2/results.csv`

## Lessons Learned

1. Manual width reduction works but is a blunt instrument -- all channels are reduced uniformly regardless of importance.
2. The 2.6 dB PSNR drop (30.23 -> 27.63) is substantial, suggesting important capacity was removed.
3. Automated structural pruning (v3) with importance-based channel selection should yield better quality at similar compression ratios.
