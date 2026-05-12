# Unstructured Pruning (v1) -- BUGGY

**Status:** Abandoned due to critical bug. See `pruning_v3/` for the corrected approach.

## What

Unstructured L1-norm pruning of three deraining models at 30/50/70% sparsity targets,
followed by fine-tuning to recover quality.

## Why

Test whether magnitude-based weight pruning can compress deraining networks without
significant PSNR loss.

## Bug: Weight Regrowth

`prune.remove()` was called **before** fine-tuning, which strips the pruning mask and
converts the reparameterized weight back to a plain `nn.Parameter`. SGD then freely
updates all weights, allowing pruned weights to regrow from zero.

| Model | Target Sparsity | Actual Sparsity After FT |
|---|---|---|
| NAFNet-w32 | 30% | **0%** (full regrowth) |
| Restormer | 30% | 12.5% |
| DRSformer | 30% | 16.6% |

No actual compression was achieved -- model sizes and GMACs remained unchanged.

## Scripts

| File | Purpose |
|---|---|
| `structured_pruning.py` | Main script (uses `torch.nn.utils.prune.ln_structured` despite filename) |
| `plot_pruning.py` | Visualization of pruning results |
| `verify_eval.py` | Post-hoc sparsity and PSNR verification |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
python structured_pruning.py
```

## Results

CSV files: `results/baselines/tables/{restormer,nafnet_w32,drsformer}_pruning.csv`

## Lessons Learned

1. `prune.remove()` before fine-tuning defeats the purpose -- masks must persist during training.
2. Unstructured pruning (zero-masking) does not reduce GMACs or model file size without sparse format support.
3. For real compression, use **structural pruning** (physically removing channels/filters) or keep masks active throughout training.
