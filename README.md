# Image Deraining Model Compression

Systematic study of model compression techniques for image deraining, comparing quality-efficiency tradeoffs across knowledge distillation, structured pruning, quantization, and ONNX deployment.

## Models

| Model | PSNR (Rain100H) | Params | GMACs | Latency (GPU) | Source |
|-------|-----------------|--------|-------|---------------|--------|
| Restormer | 31.48 dB | 26.13M | 154.9 | 46.4 ms | [Paper](https://arxiv.org/abs/2111.09881) |
| DRSformer | 33.87 dB | 33.66M | 243.0 | 109.2 ms | [Paper](https://arxiv.org/abs/2307.05915) |
| NAFNet-w32 | 30.23 dB | 29.16M | 16.1 | 11.2 ms | [Paper](https://arxiv.org/abs/2204.04676) |

Restormer is the quality anchor. NAFNet is the efficiency anchor (10x fewer GMACs, 4x lower latency). DRSformer provides an additional comparison point.

## Compression Pipeline

```
Restormer (teacher, 31.48 dB)
    |
    | Knowledge Distillation (L1 pixel + L1 distill)
    v
NAFNet-KD (30.43 dB, 29.16M params, 16.1 GMACs)
    |
    | Structural Pruning (torch-pruning DepGraph, 30% ratio)
    v
NAFNet-KD-Pruned (30.59 dB*, 20.13M params, 12.16 GMACs)   *in-progress
    |
    | FP16 / ONNX Export
    v
Deployment-ready model (~60 MB, ~8 ms ORT GPU)
```

## Key Results

| Stage | PSNR-H | PSNR-L | Params | GMACs | Size | Latency |
|-------|--------|--------|--------|-------|------|---------|
| Restormer FP32 | 31.48 | 39.15 | 26.13M | 154.9 | 104.7 MB | 46.4 ms |
| NAFNet-w32 FP32 | 30.23 | 36.94 | 29.16M | 16.1 | 116.9 MB | 11.2 ms |
| NAFNet-KD | 30.43 | 37.32 | 29.16M | 16.1 | 116.9 MB | 11.3 ms |
| NAFNet-KD FP16 | 30.43 | 37.32 | 29.16M | 16.1 | 58.6 MB | 12.4 ms |
| NAFNet-KD StructPruned30 | 30.59* | TBD | 20.13M | 12.2 | 80.8 MB | 11.1 ms |
| NAFNet-w22 (v2) | 27.63 | 32.02 | 13.85M | 7.7 | 55.6 MB | 11.0 ms |
| NAFNet-w22 ONNX FP16 | 27.63 | 32.02 | 13.85M | 7.7 | 28.4 MB | 7.1 ms |

*In-progress (iter 24K/100K). Already exceeds KD baseline.

## Experiments

Each experiment has its own README with methodology, results, and reproduction steps:

| Directory | Technique | Status |
|-----------|-----------|--------|
| [`experiments/distillation/`](experiments/distillation/) | Restormer-to-NAFNet knowledge distillation | Done |
| [`experiments/pruning/`](experiments/pruning/) | Unstructured pruning (v1, buggy -- weight regrowth) | Done (deprecated) |
| [`experiments/pruning_v2/`](experiments/pruning_v2/) | Manual width reduction to NAFNet-w22 | Done |
| [`experiments/pruning_v3/`](experiments/pruning_v3/) | Structural pruning via torch-pruning DepGraph | Training |
| [`experiments/quantization/`](experiments/quantization/) | FP16 / INT8 post-training quantization | Done |
| [`experiments/deployment/`](experiments/deployment/) | ONNX export and ORT benchmarking | Done |
| [`experiments/full_pipeline/`](experiments/full_pipeline/) | End-to-end compression pipeline (v1 pruning) | Done (deprecated) |

## Evaluation Pipeline

Unified evaluation across all 5 Rain13K test sets (Rain100H, Rain100L, Test100, Test1200, Test2800):

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate deraining

# Single model
python evaluation/evaluate_model.py --model restormer \
    --checkpoint pretrained/restormer_deraining.pth --testset all

# All baselines + generate plots
bash scripts/eval_all.sh
```

Metrics: PSNR and SSIM on Y-channel with 4px border crop (matches published papers), plus LPIPS.

## Repository Structure

```
deraining/
├── configs/                    # Training configs (NAFNet YAML)
├── datasets/                   # Symlinks to Rain13K train/test (not tracked)
├── evaluation/                 # Unified eval pipeline (metrics, efficiency, wrappers)
├── experiments/
│   ├── distillation/           # KD: Restormer → NAFNet
│   ├── pruning/                # v1: unstructured (buggy)
│   ├── pruning_v2/             # v2: manual width reduction
│   ├── pruning_v3/             # v3: structural (torch-pruning)
│   ├── quantization/           # PTQ: FP16 / INT8
│   ├── deployment/             # ONNX export + ORT benchmarks
│   ├── full_pipeline/          # End-to-end pipeline
│   ├── combined_compression.py # Master comparison script
│   └── pareto_plot.py          # Pareto frontier plots
├── models/                     # Cloned repos: Restormer, NAFNet, DRSformer, Diff-Mamba (not tracked)
├── pretrained/                 # Model weights (not tracked, see below)
├── results/baselines/
│   ├── tables/*.csv            # All result CSVs
│   └── plots/                  # Generated figures (PDF/PNG)
├── scripts/                    # Shell scripts (train, eval, download)
└── requirements.txt
```

## Setup

```bash
# Environment
conda create -n deraining python=3.10
conda activate deraining
pip install -r requirements.txt

# Clone model repos
git clone https://github.com/swz30/Restormer.git models/Restormer
git clone https://github.com/VILAB-git/NAFNet.git models/NAFNet
git clone https://github.com/dongliangchang/DRSformer.git models/DRSformer

# Download datasets
bash scripts/download_datasets.sh

# Download pretrained weights (Google Drive)
# restormer_deraining.pth  → pretrained/
# drsformer_deraining.pth  → pretrained/
# nafnet_w32_kd.pth        → pretrained/ (or train via experiments/distillation/)
```

## Hardware

- GPU: NVIDIA RTX PRO 4500 Blackwell (33.7 GB VRAM)
- CUDA 13.1, PyTorch 2.11.0+cu128
- All latency measurements at 1x3x256x256 resolution

## Key Findings

1. **Knowledge distillation works**: NAFNet-KD gains +0.20 dB over scratch training by learning from Restormer, at zero inference cost.

2. **Unstructured pruning is broken for fine-tuning**: Calling `prune.remove()` strips masks, allowing SGD to regrow weights. NAFNet regrew 100% of pruned weights. Use structural pruning instead.

3. **Structural pruning preserves quality**: torch-pruning DepGraph removes internal expansion channels physically. 1.45x param reduction with PSNR exceeding the unpruned KD model (30.59 vs 30.43 dB at iter 24K).

4. **FP16 is free compression**: <0.01 dB quality loss, 2x size reduction, no latency penalty on GPU.

5. **INT8 hurts attention models**: Static INT8 drops Restormer by 2.0 dB and runs 300x slower (CPU-only). Only viable for pure CNNs with small quality loss.

6. **ONNX helps CNNs, hurts transformers**: NAFNet gets 1.4x speedup from ORT. Restormer gets 0.45x (slower than PyTorch) due to custom attention ops falling back to CPU.
