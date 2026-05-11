# Deraining Research Setup — Work Log

End-to-end setup of the image-deraining compression project described in `/home/user/noob/prompt.md`, following the plan at `/home/user/.claude/plans/i-have-shifted-to-curious-thimble.md`.

## Environment

- Host: Ubuntu 24.04, NVIDIA RTX PRO 4500 Blackwell 33.7 GB, CUDA 13.1
- Conda env: `deraining` (Python 3.10)
- Key libs: torch 2.11.0+cu128, mamba-ssm 2.3.1, causal-conv1d 1.6.1, basicsr (patched), lpips, einops, fvcore, timm, opencv-python, gdown, matplotlib

The env is **not** auto-active in Claude Code Bash calls. Prefix commands with `source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining` or call `/home/user/anaconda3/envs/deraining/bin/python` directly.

## Repo structure

```
deraining/
├── configs/nafnet_derain_w32.yml          # NAFNet-w32 training config (Rain13K)
├── datasets/Rain13K/
│   ├── train/{input,target}               # symlinks → Restormer/Deraining/Datasets/train/Rain13K/*
│   └── test/{Rain100H,Rain100L,Test100,Test1200,Test2800}/{input,target}
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                         # PSNR/SSIM/LPIPS on Y-channel, 4px crop
│   ├── efficiency.py                      # params, GMACs (fvcore), latency, peak mem, size
│   ├── dataset.py                         # DerainDataset (paired input/target loader)
│   ├── model_wrappers.py                  # load_model('restormer'|'drsformer'|'nafnet_w32'|'nafnet_w64'|'diffmamba')
│   └── evaluate_model.py                  # CLI entrypoint: --model --checkpoint --testset --output
├── models/
│   ├── Restormer/      (cloned + BasicSR develop egg)
│   ├── DRSformer/      (cloned)
│   ├── NAFNet/         (cloned + BasicSR develop egg)
│   └── Diff-Mamba/     (cloned — actually the C2F-DFT repo, see caveats)
├── pretrained/
│   ├── restormer_deraining.pth            # 105 MB — Rain13K trained
│   ├── drsformer_deraining.pth            # 135 MB — Rain200H trained (see caveats)
│   └── drsformer_deraining_did.pth.bak    # 135 MB — archived DID-Data run
├── results/baselines/
│   ├── tables/{restormer,drsformer}_results.csv
│   ├── plots/{psnr,ssim}_comparison.{pdf,png}
│   ├── plots/psnr_vs_{gmacs,params,latency}.{pdf,png}
│   ├── plots/summary_table.{pdf,png}
│   ├── visual_samples/*.png               # input|pred|gt triptychs
│   └── generate_plots.py
├── scripts/
│   ├── download_datasets.sh               # gdown Rain13K train+test, symlink into layout
│   ├── train_nafnet.sh                    # kicks off NAFNet training
│   └── eval_all.sh                        # runs Restormer + DRSformer + plots
└── requirements.txt
```

## What was done

1. **Verified BasicSR arch imports** for Restormer, NAFNet, DRSformer. Diff-Mamba has no `DFT_arch.py` despite `test.py` referencing it — the actual arch class `DFT` lives at `models/Diff-Mamba/models/archs/Diff-Mamba.py` (hyphenated filename).
2. **Wrote the unified evaluation pipeline** from `prompt.md` verbatim: metrics, efficiency, dataset, model_wrappers, evaluate_model, generate_plots.
3. **Fixed the Diff-Mamba wrapper** to load the hyphenated arch via `importlib.util.spec_from_file_location` and a synthetic `basicsr.models.archs` package that routes `vmamba` and `model` imports to the repo's files.
4. **Wrote** `configs/nafnet_derain_w32.yml`, `scripts/{train_nafnet,eval_all,download_datasets}.sh`, `requirements.txt`.
5. **Downloaded Rain13K** — 1.3 GB test + 1.1 GB train via `gdown` (IDs from Restormer's `download_data.py`), symlinked into the project layout. Train archive extracts to `Datasets/train/Rain13K/{input,target}` (extra subfolder) — scripts account for that.
6. **Downloaded pretrained weights** — Restormer (Google Drive folder `1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u` → `deraining.pth` → renamed to `restormer_deraining.pth`); DRSformer Rain200H checkpoint (`1mt8ydHE540_qtytger4dVcv6xqZ5YMhh`), after an initial DID-Data run (`1U_UEGPhYRJ-G10-Dypr7FbwDRGmroAHC`) produced poor Rain100H results — see DRSformer results section. No usable weight for Diff-Mamba — its Drive folder holds only zipped visual results, not a `.pth`.
7. **Ran baselines** across all 5 test sets for Restormer and DRSformer. See Results below.
8. **Generated plots** — 6 publication-style figures (PDF + PNG).

## How to run

```bash
# Datasets (first time only)
bash scripts/download_datasets.sh

# Baselines + plots
bash scripts/eval_all.sh

# Single-model eval
python evaluation/evaluate_model.py --model restormer \
    --checkpoint pretrained/restormer_deraining.pth --testset all

# NAFNet training (overnight, ~300k iters)
bash scripts/train_nafnet.sh
```

## Results

### Restormer (Rain13K pretrained, 26.13 M params, 154.9 GMACs, 46.4 ms @ 256×256)

| Dataset   | # imgs | PSNR (dB) | SSIM   | LPIPS  |
|-----------|--------|-----------|--------|--------|
| Rain100H  | 100    | 31.48     | 0.9084 | 0.2034 |
| Rain100L  | 100    | 39.15     | 0.9795 | 0.0859 |
| Test100   | 98     | 32.07     | 0.9261 | 0.1643 |
| Test1200  | 1200   | 33.21     | 0.9287 | 0.1605 |
| Test2800  | 2800   | 34.25     | 0.9467 | 0.1058 |

All numbers within ~0.1 dB of the published Restormer paper — confirms correctness of pipeline (Y-channel, 4-px border crop).

Model-size metadata: 100.5 MB checkpoint, 0.71 GB peak activation memory.

### DRSformer (Rain200H pretrained, 33.66 M params, 243.0 GMACs, 109.2 ms @ 256×256)

| Dataset   | # imgs | PSNR (dB) | SSIM   | LPIPS  |
|-----------|--------|-----------|--------|--------|
| Rain100H  | 100    | 33.87     | 0.9404 | 0.1194 |
| Rain100L  | 100    | 40.50     | 0.9862 | 0.0345 |
| Test100   | 98     | 24.03     | 0.8379 | 0.2510 |
| Test1200  | 1200   | 30.01     | 0.8839 | 0.2176 |
| Test2800  | 2800   | 30.15     | 0.9029 | 0.1673 |

Model-size metadata: 129.5 MB checkpoint, 1.13 GB peak activation memory.

Upstream does NOT publish a combined-Rain13K checkpoint — it releases **per-dataset** weights (Rain200L, Rain200H, DID-Data, DDN-Data, SPA-Data). We initially used the DID-Data checkpoint but it collapsed on Rain100H (PSNR 14.25) because that weight is fit to a very different rain distribution. **We swapped to the Rain200H checkpoint** (Drive ID `1mt8ydHE540_qtytger4dVcv6xqZ5YMhh`, MEFC variant — same arch as our default `DRSformer()` constructor, no wrapper change needed). Rain100H jumps +19.6 dB and Rain100L jumps +13.0 dB, as expected — Rain200H's training distribution matches those heavy/light synthetic streaks. Test1200 drops from 35.37 → 30.01 because it is no longer the matched training split. Test100 and Test2800 move only marginally. For a fully fair Rain13K-wide comparison, DRSformer would need retraining on Rain13K or evaluating each split with its matched checkpoint; the archived DID-Data artifacts below document the tradeoff.

DID-Data artifacts are preserved for reference: `pretrained/drsformer_deraining_did.pth.bak` (135 MB) and `results/baselines/tables/drsformer_did_results.csv.bak`.

### Summary vs. Restormer

| Model      | Params | GMACs | Latency | Avg PSNR† | Avg SSIM† |
|------------|--------|-------|---------|-----------|-----------|
| Restormer  | 26.1 M | 154.9 | 46.4 ms | 34.03     | 0.9379    |
| DRSformer  | 33.7 M | 243.0 | 109.2 ms| 31.71     | 0.9103    |

† unweighted mean across the five test sets. DRSformer still underperforms Restormer on average because no Rain13K-combined weight exists upstream — the Rain200H checkpoint excels on Rain100H/L but is out-of-domain for Test100/1200/2800.

## Caveats / deferred work

- **NAFNet.** No pretrained deraining weights exist upstream — training from scratch. Launched 2026-04-17 18:06 UTC with `configs/nafnet_derain_w32.yml` (PID 28288, log `/tmp/nafnet_train.log`, checkpoints under `models/NAFNet/experiments/NAFNet-Derain-width32/models/`). Three config bugs fixed before the first successful launch: (1) `model_type: ImageCleanModel` → `ImageRestorationModel` (the registered class in NAFNet's basicsr fork), (2) flattened `losses: pixel_opt:` → `pixel_opt:` directly under `train:` (the parser at `basicsr/models/image_restoration_model.py:49` reads the top level), (3) `use_hflip` → `use_flip` (the key `PairedImageDataset` reads from the data augmentation block). **Batch size dropped from 32 → 16**: a single-GPU bs=32 at 256² OOMs on this RTX PRO 4500 (full forward pass allocates ~30 GB, error in the decoder LayerNorm). Upstream NAFNet-SIDD-width32 achieves effective batch 32 via 8 GPUs × 4 samples; we run 1 GPU × 16. Recipe: L1Loss, AdamW lr=1e-3 wd=1e-3, TrueCosineAnnealingLR to 1e-7 over 300K iters, validation on Rain100H every 5K iters, checkpoint every 5K iters. Observed 0.35 s/iter → ETA ~1d 6h. Once `net_g_300000.pth` exists, run `python evaluation/evaluate_model.py --model nafnet_w32 --checkpoint models/NAFNet/experiments/NAFNet-Derain-width32/models/net_g_300000.pth --testset all` to slot into the baselines table and regenerate plots.
- **Diff-Mamba / C2F-DFT.** The cloned `maluan-ml/Diff-Mamba` repo is a **diffusion** deraining model (6-channel input, `sampling_timesteps=3`). Our `_load_diffmamba` wrapper instantiates the arch and loads weights (so efficiency metrics are measurable), but a bare `model(x)` call will NOT produce a derained image — the diffusion sampler from the repo's `Deraining/test.py` is required. No weight file was found in the author-provided Drive folder (only a `results_fine_derain.zip` of visual outputs). Eval stays commented-out in `scripts/eval_all.sh` until upstream publishes a checkpoint and we wire up the sampler.
- **Cross-arch basicsr collision.** Each model has its own `basicsr/` package under `models/<name>/`. Inserting multiple repo roots on `sys.path` in one Python process means the first-loaded `basicsr` wins. `evaluate_model.py` sidesteps this by loading exactly one model per invocation — do not batch models inside a single process.
- **gdown version.** We installed `gdown==6.0.0`, which dropped the `--id` flag. Scripts use positional ID syntax (`gdown <ID> -O <path>`).

## Files produced in this session

- `evaluation/{__init__,metrics,efficiency,dataset,model_wrappers,evaluate_model}.py`
- `results/baselines/generate_plots.py`
- `configs/nafnet_derain_w32.yml`
- `scripts/{train_nafnet,eval_all,download_datasets}.sh`
- `requirements.txt`
- `pretrained/{restormer,drsformer}_deraining.pth` (DRSformer = Rain200H)
- `pretrained/drsformer_deraining_did.pth.bak` (archived DID-Data weight)
- `results/baselines/tables/{restormer,drsformer}_results.csv`
- `results/baselines/tables/drsformer_did_results.csv.bak` (archived DID-Data run)
- `results/baselines/plots/*.{pdf,png}` (12 files — 6 figures × 2 formats)
- `results/baselines/visual_samples/*.png` (25 triptychs — first 5 images × 5 datasets)
