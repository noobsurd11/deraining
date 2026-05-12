# Structural Pruning with torch-pruning DepGraph

**Status:** In progress (fine-tuning at ~24K/100K iterations).

## What

Automated structural (physical) pruning of NAFNet-KD using torch-pruning v1.6.0's
DepGraph and MagnitudePruner. Physically removes channels based on L1-norm importance,
then fine-tunes with KD from Restormer.

## Why

Addresses limitations of both v1 (mask-based, no real compression) and v2 (manual,
uniform width reduction). DepGraph traces inter-layer dependencies to prune coupled
channels correctly, yielding a genuinely smaller model with importance-aware selection.

## Method

### Pruning Phase
- Importance: L1-norm magnitude (global pruning)
- `round_to=8` for hardware-friendly channel counts
- `ignored_layers=[model.intro, model.ending]` -- 3-channel I/O layers tied by global residual skip
- Backbone widths cannot be pruned due to U-Net structural constraints (skip connections, PixelShuffle, downs)
- Pruning targets internal expansion convs (`conv1`/`conv4` in NAFBlocks)

### Fine-tuning Phase
- KD from Restormer (FP16 teacher)
- AdamW, lr=2e-4, cosine decay, 100K iters, batch 24, EMA
- `_build_pruned_student()` deterministically reconstructs the pruned architecture (avoids pickle issues with BasicSR's `_purge` pattern)

## Scripts

| File | Purpose |
|---|---|
| `structured_pruning_physical.py` | 3-phase pipeline: prune, finetune, eval |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining
python structured_pruning_physical.py
```

## Results So Far

| Stage | PSNR (dB) | Params | GMACs | Reduction |
|---|---|---|---|---|
| NAFNet-KD (baseline) | 30.43 | 29.16M | 16.1 | 1.0x |
| After pruning (no FT) | -- | 20.13M | 12.16 | 1.45x params / 1.32x GMACs |
| Fine-tuning @ 24K iter | **30.59** | 20.13M | 12.16 | Already exceeds baseline |

## Key Technical Details

- Backbone widths are structurally locked by the U-Net: skip connections between encoder/decoder stages, PixelShuffle upsampling, and strided downsampling all impose channel-count constraints that DepGraph correctly identifies and respects.
- The `_build_pruned_student()` function reads pruned channel counts from the state dict and reconstructs a clean NAFNet config, avoiding issues with pickling pruned modules.

## Lessons Learned

1. DepGraph correctly handles complex architectural dependencies (skip connections, PixelShuffle).
2. Even with structural constraints limiting prunable layers, 1.45x parameter reduction is achievable.
3. PSNR recovery during fine-tuning is rapid -- exceeding the unpruned KD baseline by 24K iterations suggests the pruned channels were genuinely redundant.
