# Structural Pruning with torch-pruning DepGraph

**Status:** Complete. Best checkpoint at iter 50K (30.60 dB), trained for 100K total (28.8 hours).

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
| `eval_full_pipeline.py` | Post-training eval: FP16, ONNX export, ORT benchmark |
| `generate_qualitative.py` | Visual comparison strips (5 challenging + 1 failure case) |

## How to Run

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining

# Full pipeline
python structured_pruning_physical.py --phase prune
python structured_pruning_physical.py --phase finetune \
    --pruned_ckpt checkpoints/nafnet_kd_structpruned30_raw.pth
python structured_pruning_physical.py --phase eval \
    --checkpoint checkpoints/nafnet_kd_structpruned30_best.pth

# Post-training evaluation (FP16, ONNX, ORT benchmark)
python eval_full_pipeline.py

# Qualitative comparison figures
python generate_qualitative.py
```

## Results

### Full 5-testset evaluation (best EMA checkpoint, iter 50K)

| Dataset | NAFNet-KD | KD + StructPrune30 | Delta |
|---------|-----------|-------------------|-------|
| Rain100H | 30.43 | **30.60** | +0.17 |
| Rain100L | 37.32 | **37.45** | +0.13 |
| Test100 | **30.81** | 30.68 | -0.13 |
| Test1200 | 33.46 | **33.54** | +0.08 |
| Test2800 | 33.63 | **33.76** | +0.13 |

### Efficiency

| Metric | NAFNet-KD | KD + StructPrune30 | Reduction |
|--------|-----------|-------------------|-----------|
| Params | 29.16M | 20.13M | 1.45x |
| GMACs | 16.1 | 12.2 | 1.32x |
| Checkpoint (FP32) | 116.9 MB | 80.8 MB | 1.45x |
| Checkpoint (FP16) | 58.6 MB | 40.5 MB | 1.45x |
| GPU latency | 11.3 ms | 11.2 ms | ~1.0x |
| ONNX ORT GPU | 8.0 ms | 9.0 ms | ~1.0x |

### Validation PSNR trajectory (Rain100H)

| Iteration | 10K | 20K | 30K | 40K | **50K** | 60K | 70K | 80K | 90K | 100K |
|-----------|-----|-----|-----|-----|---------|-----|-----|-----|-----|------|
| PSNR (dB) | 30.57 | 30.59 | 30.59 | 30.59 | **30.60** | 30.59 | 30.56 | 30.55 | 30.53 | 30.52 |

Model converged by ~20K iterations, best at 50K, mild overfitting after 50K.

## Key Technical Details

- Backbone widths are structurally locked by the U-Net: skip connections between encoder/decoder stages, PixelShuffle upsampling, and strided downsampling all impose channel-count constraints that DepGraph correctly identifies and respects.
- The `_build_pruned_student()` function deterministically reconstructs the pruned architecture by re-running the same pruner config (same weights + same seed = same pruned shapes), avoiding pickle/serialization issues.
- FP16 inference requires `torch.autocast` (mixed precision) — pure `model.half()` causes 2.6 dB quality loss due to LayerNorm precision issues in the pruned model.
- ONNX export must be FP32 — FP16 ONNX export produces numerically degraded output (17.7 dB).

## Lessons Learned

1. DepGraph correctly handles complex architectural dependencies (skip connections, PixelShuffle).
2. Even with structural constraints limiting prunable layers, 1.45x parameter reduction is achievable.
3. PSNR recovery during fine-tuning is rapid -- exceeding the unpruned KD baseline by 20K iterations suggests the pruned channels were genuinely redundant.
4. The apparent PSNR gain (+0.17 dB) is from additional training iterations with a fresh optimizer, not from pruning itself.
5. Early stopping at 50K iterations is optimal; continued training overfits mildly.
