"""
Structured L1-norm channel pruning + short fine-tune recovery
for Restormer, DRSformer, and NAFNet-w32.

For each (model, ratio) we run:
  1. prune every Conv2d/Linear's output channels via prune.ln_structured(n=1, dim=0),
     skipping the RGB tail (out_channels == 3) which would zero a color channel.
  2. prune.remove() to bake zeros into weights.
  3. Eval "prune_only" on Rain100H/L.
  4. Fine-tune 10K iters on Rain13K (AdamW lr=1e-4, L1 loss).
  5. Eval "prune_ft" on Rain100H/L.

Notes
-----
- ln_structured zeros filters but does NOT shrink shapes, so GMACs/latency
  are unchanged. That is intentional and documented.
- Fine-tune does not re-apply the mask; zeros are free to re-grow.
  nonzero_params_M in the "prune_ft" rows captures this drift.
- Run order: nafnet_w32 → restormer → drsformer (fastest-first).
"""

import argparse
import copy
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import prune
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset
from evaluation.efficiency import (
    count_gmacs,
    count_parameters,
    get_model_size_mb,
    measure_latency,
)
from evaluation.metrics import LPIPSMetric, compute_psnr, compute_ssim
from evaluation.model_wrappers import load_model


def _purge_basicsr():
    """Clear cached basicsr modules so the next load_model picks up the right repo.

    NAFNet, Restormer, and DRSformer all ship their own `basicsr` — Python caches
    the first import, so loading a second model would re-use the first model's arch.
    Purging before each load forces a fresh import from the correct sys.path[0].
    """
    for key in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
        del sys.modules[key]
    # Also drop entries the wrappers inserted into sys.path so the order doesn't invert.
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]

SEED = 42
DEVICE = "cuda"

PRETRAINED = {
    "nafnet_w32": PROJECT_ROOT / "pretrained/nafnet_w32_deraining.pth",
    "restormer":  PROJECT_ROOT / "pretrained/restormer_deraining.pth",
    "drsformer":  PROJECT_ROOT / "pretrained/drsformer_deraining.pth",
}
BATCH = {"nafnet_w32": 16, "restormer": 2, "drsformer": 2}  # reduced from 32/8/8: competing finllm_chat process holds ~6GB VRAM
RATIOS = [0.3, 0.5, 0.7]
FT_ITERS = 10_000
LR = 1e-4
VAL_EVERY = 2_000

CKPT_DIR = PROJECT_ROOT / "experiments/pruning/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = PROJECT_ROOT / "results/baselines/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TESTSETS = {
    "Rain100H": PROJECT_ROOT / "datasets/Rain13K/test/Rain100H",
    "Rain100L": PROJECT_ROOT / "datasets/Rain13K/test/Rain100L",
}
TRAIN_INPUT = PROJECT_ROOT / "datasets/Rain13K/train/input"
TRAIN_TARGET = PROJECT_ROOT / "datasets/Rain13K/train/target"

CSV_FIELDS = [
    "model", "prune_ratio", "stage", "dataset",
    "psnr", "ssim", "lpips",
    "total_params_M", "nonzero_params_M", "sparsity_pct",
    "model_size_mb", "gmacs", "latency_ms",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- data ----------

class DerainTrainDataset(Dataset):
    """Rain13K pairs with random 256x256 crop + random horizontal flip."""

    def __init__(self, input_dir: Path, target_dir: Path, crop_size: int = 256):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

        valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        self.input_files = sorted(
            f for f in self.input_dir.iterdir() if f.suffix.lower() in valid_ext
        )
        self.target_files = sorted(
            f for f in self.target_dir.iterdir() if f.suffix.lower() in valid_ext
        )
        assert len(self.input_files) == len(self.target_files), \
            f"mismatch: {len(self.input_files)} vs {len(self.target_files)}"

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int):
        inp = Image.open(self.input_files[idx]).convert("RGB")
        tgt = Image.open(self.target_files[idx]).convert("RGB")
        w, h = inp.size
        cs = self.crop_size
        if w < cs or h < cs:
            inp = inp.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
            tgt = tgt.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
            w, h = inp.size
        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)
        inp = inp.crop((x, y, x + cs, y + cs))
        tgt = tgt.crop((x, y, x + cs, y + cs))
        if random.random() < 0.5:
            inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
            tgt = tgt.transpose(Image.FLIP_LEFT_RIGHT)
        return self.to_tensor(inp), self.to_tensor(tgt)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# ---------- pruning ----------

def prune_model(model: nn.Module, ratio: float, verbose: bool = True):
    """Apply L1-norm structured pruning to output channels of every Conv2d/Linear.

    Skips layers where out_channels / out_features == 3 (RGB tail), since
    pruning there would zero whole color channels.
    """
    pruned_layers, skipped = [], []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            out = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
            if out == 3 or out == 1:
                skipped.append((name, out, "rgb-tail"))
                continue
            # ln_structured needs amount such that floor(amount * out) >= 1.
            if int(ratio * out) < 1:
                skipped.append((name, out, "too-small"))
                continue
            try:
                prune.ln_structured(m, name="weight", amount=ratio, n=1, dim=0)
                prune.remove(m, "weight")
                pruned_layers.append((name, out))
            except Exception as e:
                skipped.append((name, out, f"err:{type(e).__name__}"))
    if verbose:
        print(f"  pruned {len(pruned_layers)} layers at ratio={ratio}")
        print(f"  skipped {len(skipped)} layers: {[s[0] for s in skipped][:5]}"
              + (" ..." if len(skipped) > 5 else ""))
    return model


def sparsity_stats(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return {
        "total_params_M": total / 1e6,
        "nonzero_params_M": nonzero / 1e6,
        "sparsity_pct": 100.0 * (1.0 - nonzero / total),
    }


# ---------- eval ----------

@torch.no_grad()
def eval_on_testset(model: nn.Module, testset_path: Path,
                    lpips_metric: LPIPSMetric) -> dict:
    ds = DerainDataset(
        input_dir=str(testset_path / "input"),
        target_dir=str(testset_path / "target"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for inp, gt, _ in tqdm(loader, desc=f"  eval {testset_path.name}", leave=False):
        inp, gt = inp.to(DEVICE), gt.to(DEVICE)
        pred = torch.clamp(model(inp), 0, 1)
        psnrs.append(compute_psnr(pred, gt))
        ssims.append(compute_ssim(pred, gt))
        lpipss.append(lpips_metric.compute(pred, gt))
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }


@torch.no_grad()
def quick_rain100h_psnr(model: nn.Module) -> float:
    ds = DerainDataset(
        input_dir=str(TESTSETS["Rain100H"] / "input"),
        target_dir=str(TESTSETS["Rain100H"] / "target"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    psnrs = []
    for inp, gt, _ in loader:
        inp, gt = inp.to(DEVICE), gt.to(DEVICE)
        pred = torch.clamp(model(inp), 0, 1)
        psnrs.append(compute_psnr(pred, gt))
    return float(np.mean(psnrs))


def measure_stage(model: nn.Module, ckpt_path: Path) -> dict:
    eff = {
        "gmacs": count_gmacs(model),
        "latency_ms": measure_latency(model)["mean_ms"],
        "model_size_mb": get_model_size_mb(str(ckpt_path)) if ckpt_path.exists() else -1.0,
    }
    eff.update(sparsity_stats(model))
    return eff


# ---------- fine-tune ----------

def finetune(model: nn.Module, model_name: str, ratio: float,
             iters: int, batch_size: int) -> nn.Module:
    ds = DerainTrainDataset(TRAIN_INPUT, TRAIN_TARGET, crop_size=256)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    torch.cuda.empty_cache()
    it = cycle(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()
    # bfloat16 autocast: halves activation memory, has FP32 exponent range so
    # no GradScaler needed and no instability on freshly-pruned weights.
    # FP16 caused Restormer@0.3 PSNR to oscillate 17→12→14 during an earlier run.
    use_amp = model_name in ("restormer", "drsformer")
    model.train()
    pbar = tqdm(range(iters), desc=f"ft[{model_name}@{ratio}]")
    for step in pbar:
        inp, tgt = next(it)
        inp = inp.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            pred = model(inp)
            loss = loss_fn(pred, tgt)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if (step + 1) % VAL_EVERY == 0:
            psnr_h = quick_rain100h_psnr(model)
            print(f"  [{model_name} ratio={ratio}] iter {step+1}/{iters}  "
                  f"Rain100H PSNR = {psnr_h:.3f}  loss = {loss.item():.4f}")
            model.train()
    model.eval()
    return model


# ---------- main per-model runner ----------

def run_model(model_name: str, lpips_metric: LPIPSMetric) -> None:
    print(f"\n{'='*70}\n  {model_name}  (batch={BATCH[model_name]})\n{'='*70}")
    _purge_basicsr()
    fp32 = load_model(model_name, str(PRETRAINED[model_name]), DEVICE)

    rows = []
    for ratio in RATIOS:
        print(f"\n--- {model_name}  ratio={ratio} ---")
        set_seed(SEED)
        pruned = copy.deepcopy(fp32).to(DEVICE)
        prune_model(pruned, ratio, verbose=True)

        prune_ckpt = CKPT_DIR / f"{model_name}_pruned_{ratio}.pth"
        torch.save(pruned.state_dict(), prune_ckpt)

        eff_p = measure_stage(pruned, prune_ckpt)
        print(f"  prune_only: params={eff_p['total_params_M']:.2f}M  "
              f"nonzero={eff_p['nonzero_params_M']:.2f}M  "
              f"sparsity={eff_p['sparsity_pct']:.1f}%  size={eff_p['model_size_mb']:.1f}MB")

        for ds_name, ds_path in TESTSETS.items():
            q = eval_on_testset(pruned, ds_path, lpips_metric)
            rows.append(dict(
                model=model_name, prune_ratio=ratio, stage="prune_only",
                dataset=ds_name, **q, **eff_p,
            ))
            print(f"  prune_only {ds_name}: PSNR={q['psnr']:.3f} "
                  f"SSIM={q['ssim']:.4f} LPIPS={q['lpips']:.4f}")

        ft = finetune(pruned, model_name, ratio,
                      iters=FT_ITERS, batch_size=BATCH[model_name])
        ft_ckpt = CKPT_DIR / f"{model_name}_pruned_{ratio}_finetuned.pth"
        torch.save(ft.state_dict(), ft_ckpt)

        eff_ft = measure_stage(ft, ft_ckpt)
        print(f"  prune_ft:  params={eff_ft['total_params_M']:.2f}M  "
              f"nonzero={eff_ft['nonzero_params_M']:.2f}M  "
              f"sparsity={eff_ft['sparsity_pct']:.1f}%  size={eff_ft['model_size_mb']:.1f}MB")

        for ds_name, ds_path in TESTSETS.items():
            q = eval_on_testset(ft, ds_path, lpips_metric)
            rows.append(dict(
                model=model_name, prune_ratio=ratio, stage="prune_ft",
                dataset=ds_name, **q, **eff_ft,
            ))
            print(f"  prune_ft   {ds_name}: PSNR={q['psnr']:.3f} "
                  f"SSIM={q['ssim']:.4f} LPIPS={q['lpips']:.4f}")

        del pruned, ft
        torch.cuda.empty_cache()

    # write CSV
    csv_path = TABLE_DIR / f"{model_name}_pruning.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"\nCSV written: {csv_path}  ({len(rows)} rows)")


# ---------- entry ----------

def main() -> None:
    global FT_ITERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["nafnet_w32", "restormer", "drsformer"],
                        help="run these models in order")
    parser.add_argument("--ft_iters", type=int, default=FT_ITERS)
    args = parser.parse_args()
    FT_ITERS = args.ft_iters

    set_seed(SEED)
    lpips_metric = LPIPSMetric(DEVICE)
    t0 = time.time()
    for m in args.models:
        csv_path = TABLE_DIR / f"{m}_pruning.csv"
        if csv_path.exists():
            print(f"\n[skip] {m} — {csv_path.name} already complete")
            continue
        run_model(m, lpips_metric)
        print(f"[elapsed {(time.time()-t0)/60:.1f} min after {m}]")
    print(f"\nAll done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
