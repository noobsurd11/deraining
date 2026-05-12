"""
Physical structural pruning of NAFNet-KD via torch-pruning DepGraph.

Three-phase pipeline:
  --phase prune     Step 1: L1-magnitude global pruning, save raw checkpoint
  --phase finetune  Step 2: KD fine-tune pruned model (100K iters)
  --phase eval      Step 3: Evaluate, FP16, ONNX export

Pruning uses torch_pruning.MagnitudePruner which resolves PixelShuffle,
skip-connection, and depthwise-conv dependencies automatically via the
dependency graph — channels are physically removed, not just zeroed.

KD training matches experiments/distillation/distill_restormer_to_nafnet.py:
  L = 1.0 * L1(pred, gt) + 0.5 * L1(pred, teacher(inp))
  AdamW(lr=2e-4, wd=1e-3, betas=(0.9, 0.9))
  Linear warmup 2K iters -> cosine decay to 1e-6
  EMA decay=0.999, fp16 autocast + GradScaler
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset
from evaluation.efficiency import count_gmacs, count_parameters, get_model_size_mb, measure_latency
from evaluation.metrics import compute_psnr, compute_ssim, LPIPSMetric

SEED = 42
DEVICE = "cuda"

WORK_DIR = Path(__file__).resolve().parent
CKPT_DIR = WORK_DIR / "checkpoints"

STUDENT_CKPT = PROJECT_ROOT / "pretrained" / "nafnet_w32_kd.pth"
TEACHER_CKPT = PROJECT_ROOT / "pretrained" / "restormer_deraining.pth"

TRAIN_INPUT = PROJECT_ROOT / "datasets" / "Rain13K" / "train" / "input"
TRAIN_TARGET = PROJECT_ROOT / "datasets" / "Rain13K" / "train" / "target"
VAL_INPUT = PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Rain100H" / "input"
VAL_TARGET = PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Rain100H" / "target"
TEST_ROOT = PROJECT_ROOT / "datasets" / "Rain13K" / "test"

TESTSETS = {
    "Rain100H": TEST_ROOT / "Rain100H",
    "Rain100L": TEST_ROOT / "Rain100L",
    "Test100": TEST_ROOT / "Test100",
    "Test1200": TEST_ROOT / "Test1200",
    "Test2800": TEST_ROOT / "Test2800",
}

# Training hyperparams — matched from distill_restormer_to_nafnet.py
TOTAL_ITERS = 100_000
WARMUP_ITERS = 2_000
LR_PEAK = 2e-4
LR_FLOOR = 1e-6
WEIGHT_DECAY = 1e-3
BETAS = (0.9, 0.9)
BATCH = 24
NUM_WORKERS = 8
CROP = 256
EMA_DECAY = 0.999
W_PIXEL = 1.0
W_DISTILL = 0.5
VAL_EVERY = 10_000
CKPT_EVERY = 25_000
LOG_EVERY = 500


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _purge_basicsr() -> None:
    for key in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
        del sys.modules[key]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def _load_nafnet(ckpt_path: str, device: str = DEVICE) -> nn.Module:
    _purge_basicsr()
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "NAFNet"))
    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(
        img_channel=3, width=32, middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt_data.get("params", ckpt_data.get("state_dict", ckpt_data))
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _load_teacher(device: str = DEVICE) -> nn.Module:
    _purge_basicsr()
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "Restormer"))
    from basicsr.models.archs.restormer_arch import Restormer
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
    )
    ckpt_data = torch.load(str(TEACHER_CKPT), map_location=device, weights_only=False)
    sd = ckpt_data.get("params", ckpt_data.get("state_dict", ckpt_data))
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    model.load_state_dict(sd, strict=True)
    model.to(device).half().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_pruned_student(ratio: float = 0.3, round_to: int = 8,
                          device: str = "cpu") -> nn.Module:
    """Deterministically reconstruct the pruned NAFNet architecture.

    Loads KD weights, applies the same MagnitudePruner config used in
    phase_prune. Because pruning is deterministic (same weights + same
    config = same result), the returned model has identical layer shapes
    to the one produced by phase_prune, so any state_dict saved from
    that model (or from fine-tuning) can be loaded with strict=True.
    """
    import torch_pruning as tp

    _purge_basicsr()
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "NAFNet"))
    from basicsr.models.archs.NAFNet_arch import NAFNet, LayerNorm2d

    model = NAFNet(
        img_channel=3, width=32, middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt_data = torch.load(str(STUDENT_CKPT), map_location="cpu", weights_only=False)
    sd = ckpt_data.get("params", ckpt_data.get("state_dict", ckpt_data))
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    model.load_state_dict(sd, strict=True)

    set_seed()
    example_input = torch.randn(1, 3, 256, 256)
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored_layers = [model.intro, model.ending]

    ln_modules = [m for m in model.modules() if isinstance(m, LayerNorm2d)]
    unwrapped_parameters = []
    for ln in ln_modules:
        unwrapped_parameters.append((ln.weight, 0))
        unwrapped_parameters.append((ln.bias, 0))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pruner = tp.pruner.MagnitudePruner(
            model, example_input, importance=imp,
            pruning_ratio=ratio, global_pruning=True, round_to=round_to,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )
        pruner.step()

    return model.to(device).eval()


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pruning_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# =====================================================================
#  STEP 1: PRUNE
# =====================================================================

def phase_prune(ratio: float, round_to: int) -> None:
    import torch_pruning as tp

    log = setup_logging(WORK_DIR / "prune.log")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    log.info("=" * 70)
    log.info("  PHASE: PRUNE  (ratio=%.2f, round_to=%d)", ratio, round_to)
    log.info("=" * 70)

    # Load original model
    log.info("Loading NAFNet-KD from %s", STUDENT_CKPT)
    model = _load_nafnet(str(STUDENT_CKPT), device="cpu")

    orig_params = count_parameters(model)
    model_on_gpu = copy.deepcopy(model).to(DEVICE)
    orig_gmacs = count_gmacs(model_on_gpu, device=DEVICE)
    orig_lat = measure_latency(model_on_gpu, device=DEVICE)
    del model_on_gpu
    torch.cuda.empty_cache()

    log.info("Original model:")
    log.info("  Params:  %.4f M", orig_params)
    log.info("  GMACs:   %.3f", orig_gmacs)
    log.info("  Latency: %.2f ms (GPU, 256x256)", orig_lat["mean_ms"])

    # Print original channel widths
    log.info("\nOriginal channel widths:")
    _print_channel_widths(model, log)

    # Build pruner
    example_input = torch.randn(1, 3, 256, 256)
    imp = tp.importance.MagnitudeImportance(p=1)

    # Ignore intro and ending (3-channel I/O, tied by global residual)
    ignored_layers = [model.intro, model.ending]

    # Custom handling for LayerNorm2d: tell torch-pruning about the
    # unwrapped gamma/beta parameters so they get pruned consistently.
    # Collect all LayerNorm2d modules.
    from basicsr.models.archs.NAFNet_arch import LayerNorm2d
    ln_modules = [m for m in model.modules() if isinstance(m, LayerNorm2d)]

    # Map unwrapped parameters to their pruning functions
    unwrapped_parameters = []
    for ln in ln_modules:
        unwrapped_parameters.append((ln.weight, 0))  # prune dim 0
        unwrapped_parameters.append((ln.bias, 0))

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_input,
        importance=imp,
        pruning_ratio=ratio,
        global_pruning=True,
        round_to=round_to,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )

    log.info("\nDepGraph built. Executing pruning...")
    pruner.step()
    log.info("Pruning applied.")

    # Verify forward pass
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256, device=DEVICE)
        out = model(dummy)
    assert out.shape == (1, 3, 256, 256), f"Bad output shape: {out.shape}"
    log.info("Forward pass OK, output shape: %s", tuple(out.shape))

    # New stats
    new_params = count_parameters(model)
    new_gmacs = count_gmacs(model, device=DEVICE)
    new_lat = measure_latency(model, device=DEVICE)

    log.info("\nPruned model:")
    log.info("  Params:  %.4f M (%.1f%% of original, %.1fx reduction)",
             new_params, 100 * new_params / orig_params, orig_params / new_params)
    log.info("  GMACs:   %.3f (%.1f%% of original, %.1fx reduction)",
             new_gmacs, 100 * new_gmacs / orig_gmacs, orig_gmacs / new_gmacs)
    log.info("  Latency: %.2f ms (%.1f%% of original, %.1fx speedup)",
             new_lat["mean_ms"], 100 * new_lat["mean_ms"] / orig_lat["mean_ms"],
             orig_lat["mean_ms"] / new_lat["mean_ms"])

    log.info("\nPruned channel widths:")
    _print_channel_widths(model, log)

    # Quick PSNR check on Rain100H (pruned, no FT yet)
    log.info("\nEvaluating pruned model (no fine-tuning) on Rain100H...")
    psnr_raw = _eval_psnr(model, "Rain100H")
    log.info("  Rain100H PSNR (raw pruned): %.2f dB", psnr_raw)

    # Save
    ratio_tag = f"{int(ratio * 100)}"
    ckpt_name = f"nafnet_kd_structpruned{ratio_tag}_raw.pth"
    ckpt_path = CKPT_DIR / ckpt_name
    torch.save(model.state_dict(), ckpt_path)
    ckpt_size = get_model_size_mb(str(ckpt_path))
    log.info("\nSaved: %s (%.1f MB)", ckpt_path, ckpt_size)

    log.info("\n" + "=" * 70)
    log.info("  PRUNE SUMMARY")
    log.info("=" * 70)
    log.info("  %-25s %12s %12s %10s", "", "Original", "Pruned", "Ratio")
    log.info("  " + "-" * 62)
    log.info("  %-25s %10.4f M %10.4f M %9.2fx", "Parameters",
             orig_params, new_params, orig_params / new_params)
    log.info("  %-25s %10.3f   %10.3f   %9.2fx", "GMACs",
             orig_gmacs, new_gmacs, orig_gmacs / new_gmacs)
    log.info("  %-25s %10.2f ms %9.2f ms %8.2fx", "Latency (GPU)",
             orig_lat["mean_ms"], new_lat["mean_ms"],
             orig_lat["mean_ms"] / new_lat["mean_ms"])
    log.info("  %-25s %10.1f MB %9.1f MB %8.2fx", "Checkpoint size",
             get_model_size_mb(str(STUDENT_CKPT)), ckpt_size,
             get_model_size_mb(str(STUDENT_CKPT)) / ckpt_size)
    log.info("  %-25s %10.2f dB %9.2f dB %+8.2f dB", "Rain100H PSNR",
             30.43, psnr_raw, psnr_raw - 30.43)
    log.info("=" * 70)

    del model
    torch.cuda.empty_cache()


def _get_arch_info(model: nn.Module) -> dict:
    """Extract channel widths from a pruned NAFNet so we can reconstruct it."""
    info = {}
    info["intro_out"] = model.intro.out_channels
    info["ending_in"] = model.ending.in_channels

    info["enc_widths"] = []
    for i, enc in enumerate(model.encoders):
        info["enc_widths"].append(enc[0].conv1.in_channels)

    info["down_widths"] = []
    for d in model.downs:
        info["down_widths"].append((d.in_channels, d.out_channels))

    info["middle_width"] = model.middle_blks[0].conv1.in_channels
    info["middle_num"] = len(model.middle_blks)

    info["up_widths"] = []
    for u in model.ups:
        info["up_widths"].append((u[0].in_channels, u[0].out_channels))

    info["dec_widths"] = []
    for dec in model.decoders:
        info["dec_widths"].append(dec[0].conv1.in_channels)

    info["enc_blk_nums"] = [len(enc) for enc in model.encoders]
    info["dec_blk_nums"] = [len(dec) for dec in model.decoders]
    return info


def _print_channel_widths(model: nn.Module, log: logging.Logger) -> None:
    log.info("  intro: %d -> %d", model.intro.in_channels, model.intro.out_channels)
    for i, (enc, down) in enumerate(zip(model.encoders, model.downs)):
        w = enc[0].conv1.in_channels
        log.info("  encoder[%d]: width=%d (%d blocks), down: %d->%d",
                 i, w, len(enc), down.in_channels, down.out_channels)
    w = model.middle_blks[0].conv1.in_channels
    log.info("  middle: width=%d (%d blocks)", w, len(model.middle_blks))
    for i, (dec, up) in enumerate(zip(model.decoders, model.ups)):
        w = dec[0].conv1.in_channels
        log.info("  decoder[%d]: width=%d (%d blocks), up: %d->%d",
                 i, w, len(dec), up[0].in_channels, up[0].out_channels)
    log.info("  ending: %d -> %d", model.ending.in_channels, model.ending.out_channels)


@torch.no_grad()
def _eval_psnr(model: nn.Module, testset: str) -> float:
    ds = DerainDataset(str(TESTSETS[testset] / "input"), str(TESTSETS[testset] / "target"))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    psnrs = []
    for inp, gt, _ in tqdm(loader, desc=f"  eval {testset}", leave=False):
        inp = inp.to(DEVICE)
        pred = torch.clamp(model(inp), 0, 1)
        psnrs.append(compute_psnr(pred, gt.to(DEVICE)))
    return float(np.mean(psnrs))


# =====================================================================
#  STEP 2: FINE-TUNE WITH KD
# =====================================================================

class DistillTrainDataset(Dataset):
    VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, input_dir: Path, target_dir: Path, crop_size: int = CROP):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.crop = crop_size
        self.to_tensor = transforms.ToTensor()
        self.inputs = sorted(
            f for f in self.input_dir.iterdir() if f.suffix.lower() in self.VALID_EXT
        )
        self.targets = sorted(
            f for f in self.target_dir.iterdir() if f.suffix.lower() in self.VALID_EXT
        )
        assert len(self.inputs) == len(self.targets)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        inp = self.to_tensor(Image.open(self.inputs[idx]).convert("RGB"))
        tgt = self.to_tensor(Image.open(self.targets[idx]).convert("RGB"))
        inp, tgt = self._reflect_pad(inp, tgt)
        _, h, w = inp.shape
        cs = self.crop
        y = random.randint(0, h - cs)
        x = random.randint(0, w - cs)
        inp = inp[:, y:y + cs, x:x + cs]
        tgt = tgt[:, y:y + cs, x:x + cs]
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[2])
            tgt = torch.flip(tgt, dims=[2])
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[1])
            tgt = torch.flip(tgt, dims=[1])
        return inp, tgt

    def _reflect_pad(self, inp: torch.Tensor, tgt: torch.Tensor):
        _, h, w = inp.shape
        cs = self.crop
        pad_h = max(0, cs - h)
        pad_w = max(0, cs - w)
        if pad_h == 0 and pad_w == 0:
            return inp, tgt
        pads = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        inp = F.pad(inp.unsqueeze(0), pads, mode="reflect").squeeze(0)
        tgt = F.pad(tgt.unsqueeze(0), pads, mode="reflect").squeeze(0)
        return inp, tgt


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def compute_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return LR_PEAK * (step + 1) / WARMUP_ITERS
    progress = (step - WARMUP_ITERS) / max(1, TOTAL_ITERS - WARMUP_ITERS)
    progress = min(1.0, max(0.0, progress))
    return LR_FLOOR + 0.5 * (LR_PEAK - LR_FLOOR) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = EMA_DECAY) -> None:
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.mul_(decay).add_(mp.detach(), alpha=1 - decay)
    for eb, mb in zip(ema.buffers(), model.buffers()):
        eb.copy_(mb)


@torch.no_grad()
def validate(model: nn.Module) -> float:
    ds = DerainDataset(str(VAL_INPUT), str(VAL_TARGET))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    psnrs = []
    for inp, gt, _ in loader:
        inp = inp.to(DEVICE, non_blocking=True)
        gt = gt.to(DEVICE, non_blocking=True)
        pred = torch.clamp(model(inp), 0, 1)
        psnrs.append(compute_psnr(pred, gt))
    return float(np.mean(psnrs))


def save_full_ckpt(path, student, ema, optimizer, scaler, step, best_psnr):
    torch.save({
        "model_state_dict": student.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "current_iter": step,
        "best_psnr": best_psnr,
    }, path)


def phase_finetune(pruned_ckpt: str, resume: str | None = None) -> None:
    log = setup_logging(WORK_DIR / "train_structpruned30.log")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    log.info("=" * 70)
    log.info("  PHASE: FINETUNE  (%dK iters, KD from Restormer)", TOTAL_ITERS // 1000)
    log.info("=" * 70)

    # Reconstruct pruned architecture deterministically, then load weights
    log.info("Reconstructing pruned NAFNet architecture...")
    student = _build_pruned_student(device="cpu")
    pruned_sd = torch.load(pruned_ckpt, map_location="cpu", weights_only=True)
    student.load_state_dict(pruned_sd, strict=True)
    student = student.to(DEVICE)
    log.info("  Loaded weights from %s", pruned_ckpt)
    log.info("  Params: %.4f M", count_parameters(student))

    # Verify forward
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256, device=DEVICE)
        out = student(dummy)
    assert out.shape == (1, 3, 256, 256), f"Bad shape: {out.shape}"
    log.info("  Forward pass OK")
    del dummy, out

    # EMA
    ema = copy.deepcopy(student).to(DEVICE).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # Teacher (loads Restormer's basicsr — student is already constructed)
    log.info("Loading Restormer teacher...")
    teacher = _load_teacher(DEVICE)
    log.info("  Teacher loaded, frozen")

    # Data
    ds = DistillTrainDataset(TRAIN_INPUT, TRAIN_TARGET, crop_size=CROP)
    log.info("Train pairs: %d", len(ds))
    loader = DataLoader(
        ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    it = cycle(loader)

    # Optimizer
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY, betas=BETAS,
    )
    scaler = torch.amp.GradScaler("cuda")

    start_iter = 0
    best_psnr = -1.0

    if resume:
        log.info("Resuming from %s", resume)
        ck = torch.load(resume, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ck["model_state_dict"])
        ema.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_iter = int(ck["current_iter"]) + 1
        best_psnr = float(ck.get("best_psnr", -1.0))
        log.info("  Resumed at iter=%d, best_psnr=%.3f", start_iter, best_psnr)

    log.info(
        "Config: total_iters=%d warmup=%d batch=%d workers=%d lr=%.2e "
        "wd=%.0e betas=%s ema=%.3f w_pixel=%.1f w_distill=%.1f",
        TOTAL_ITERS, WARMUP_ITERS, BATCH, NUM_WORKERS, LR_PEAK,
        WEIGHT_DECAY, BETAS, EMA_DECAY, W_PIXEL, W_DISTILL,
    )

    # Training loop
    student.train()
    pbar = tqdm(range(start_iter, TOTAL_ITERS), initial=start_iter, total=TOTAL_ITERS,
                desc="finetune-pruned")
    t0 = time.time()

    for step in pbar:
        lr = compute_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        inp, tgt = next(it)
        inp = inp.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            pred_raw = student(inp)
            pred = torch.clamp(pred_raw, 0, 1)
            with torch.no_grad():
                teach_raw = teacher(inp)
                teach = torch.clamp(teach_raw, 0, 1)
            l_pixel = F.l1_loss(pred, tgt)
            l_distill = F.l1_loss(pred, teach)
            loss = W_PIXEL * l_pixel + W_DISTILL * l_distill

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, student, decay=EMA_DECAY)

        if (step + 1) % LOG_EVERY == 0 or step == start_iter:
            msg = (
                f"METRIC iter={step + 1} total={loss.item():.5f} "
                f"l_pix={l_pixel.item():.5f} l_distill={l_distill.item():.5f} "
                f"lr={lr:.3e}"
            )
            tqdm.write(msg)
            log.info(msg)

        if (step + 1) % VAL_EVERY == 0:
            psnr_h = validate(ema)
            msg = f"VAL iter={step + 1} psnr={psnr_h:.4f}"
            tqdm.write(msg)
            log.info(msg)
            if psnr_h > best_psnr:
                best_psnr = psnr_h
                torch.save(ema.state_dict(), CKPT_DIR / "nafnet_kd_structpruned30_best.pth")
                log.info("  New best EMA (PSNR=%.4f)", psnr_h)
            student.train()

        if (step + 1) % CKPT_EVERY == 0:
            ckpt_path = CKPT_DIR / f"nafnet_kd_structpruned30_iter_{step + 1}.pth"
            save_full_ckpt(ckpt_path, student, ema, optimizer, scaler, step, best_psnr)
            log.info("  Checkpoint saved -> %s", ckpt_path)

    elapsed = time.time() - t0
    log.info("Training done in %.1f hours. best_psnr=%.4f", elapsed / 3600, best_psnr)

    # Save final EMA if no best was written
    best_path = CKPT_DIR / "nafnet_kd_structpruned30_best.pth"
    if not best_path.exists():
        torch.save(ema.state_dict(), best_path)
        log.info("No best checkpoint existed, saved final EMA.")

    del teacher, student, ema
    torch.cuda.empty_cache()


# =====================================================================
#  STEP 3: EVAL
# =====================================================================

def phase_eval(checkpoint: str) -> None:
    log = setup_logging(WORK_DIR / "eval_pruned30.log")
    set_seed()

    log.info("=" * 70)
    log.info("  PHASE: EVAL")
    log.info("=" * 70)

    # Reconstruct pruned architecture, then load fine-tuned weights
    log.info("Loading checkpoint: %s", checkpoint)
    model = _build_pruned_student(device="cpu")
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.to(DEVICE).eval()

    params_m = count_parameters(model)
    gmacs = count_gmacs(model, device=DEVICE)
    lat = measure_latency(model, device=DEVICE)
    ckpt_size = get_model_size_mb(checkpoint)

    log.info("  Params: %.4f M, GMACs: %.3f, Latency: %.2f ms, Size: %.1f MB",
             params_m, gmacs, lat["mean_ms"], ckpt_size)

    _print_channel_widths(model, log)

    # Full evaluation on all 5 test sets
    log.info("\nEvaluating on all test sets...")
    lpips_metric = LPIPSMetric(DEVICE)
    results = {}
    for ds_name, ds_path in TESTSETS.items():
        ds = DerainDataset(str(ds_path / "input"), str(ds_path / "target"))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        psnrs, ssims, lpipss = [], [], []
        for inp, gt, _ in tqdm(loader, desc=f"  {ds_name}"):
            inp, gt = inp.to(DEVICE), gt.to(DEVICE)
            with torch.no_grad():
                pred = torch.clamp(model(inp), 0, 1)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim(pred, gt))
            lpipss.append(lpips_metric.compute(pred, gt))
        results[ds_name] = {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "lpips": float(np.mean(lpipss)),
        }
        log.info("  %s: PSNR=%.2f  SSIM=%.4f  LPIPS=%.4f",
                 ds_name, results[ds_name]["psnr"],
                 results[ds_name]["ssim"], results[ds_name]["lpips"])

    # FP16
    log.info("\nFP16 conversion...")
    model_fp16 = copy.deepcopy(model).half().to(DEVICE).eval()
    fp16_ckpt = CKPT_DIR / "nafnet_kd_structpruned30_best_fp16.pth"
    torch.save(model_fp16.state_dict(), fp16_ckpt)
    fp16_size = get_model_size_mb(str(fp16_ckpt))
    fp16_lat = measure_latency(model_fp16, device=DEVICE)

    @torch.no_grad()
    def fp16_eval(ds_name):
        ds = DerainDataset(
            str(TESTSETS[ds_name] / "input"), str(TESTSETS[ds_name] / "target"))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        psnrs = []
        for inp, gt, _ in loader:
            inp = inp.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = torch.clamp(model_fp16(inp.half()), 0, 1)
            psnrs.append(compute_psnr(pred.float().cpu(), gt))
        return float(np.mean(psnrs))

    fp16_psnr_h = fp16_eval("Rain100H")
    fp16_psnr_l = fp16_eval("Rain100L")
    log.info("  FP16: Rain100H=%.2f, Rain100L=%.2f, Size=%.1f MB, Latency=%.2f ms",
             fp16_psnr_h, fp16_psnr_l, fp16_size, fp16_lat["mean_ms"])

    # ONNX export
    log.info("\nONNX export...")
    onnx_path = WORK_DIR / "nafnet_kd_structpruned30_fp16.onnx"
    onnx_lat_gpu = float("nan")
    onnx_size = float("nan")
    try:
        model_export = copy.deepcopy(model).half().to(DEVICE).eval()
        dummy = torch.randn(1, 3, 256, 256, device=DEVICE, dtype=torch.float16)
        torch.onnx.export(
            model_export, dummy, str(onnx_path),
            opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {2: "height", 3: "width"},
                          "output": {2: "height", 3: "width"}},
            dynamo=False,
        )
        onnx_size = os.path.getsize(onnx_path) / 1e6
        log.info("  ONNX exported: %s (%.1f MB)", onnx_path.name, onnx_size)

        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        log.info("  ONNX verified OK")

        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        in_name = sess.get_inputs()[0].name
        inp_np = np.random.randn(1, 3, 256, 256).astype(np.float16)
        for _ in range(10):
            sess.run(None, {in_name: inp_np})
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess.run(None, {in_name: inp_np})
            times.append((time.perf_counter() - t0) * 1000)
        onnx_lat_gpu = float(np.mean(times))
        log.info("  ORT GPU latency: %.2f ms", onnx_lat_gpu)
    except Exception as e:
        log.error("  ONNX failed: %s", e)
        traceback.print_exc()

    # Write CSV
    csv_path = WORK_DIR / "results_pruned30.csv"
    fields = ["stage", "dataset", "psnr", "ssim", "lpips",
              "params_M", "gmacs", "model_size_mb", "gpu_latency_ms"]
    rows = []
    for ds_name, r in results.items():
        rows.append({
            "stage": "pruned30_ft_fp32",
            "dataset": ds_name,
            "psnr": r["psnr"],
            "ssim": r["ssim"],
            "lpips": r["lpips"],
            "params_M": params_m,
            "gmacs": gmacs,
            "model_size_mb": ckpt_size,
            "gpu_latency_ms": lat["mean_ms"],
        })
    rows.append({
        "stage": "pruned30_ft_fp16",
        "dataset": "Rain100H",
        "psnr": fp16_psnr_h, "ssim": "", "lpips": "",
        "params_M": params_m, "gmacs": gmacs,
        "model_size_mb": fp16_size, "gpu_latency_ms": fp16_lat["mean_ms"],
    })
    rows.append({
        "stage": "pruned30_ft_fp16",
        "dataset": "Rain100L",
        "psnr": fp16_psnr_l, "ssim": "", "lpips": "",
        "params_M": params_m, "gmacs": gmacs,
        "model_size_mb": fp16_size, "gpu_latency_ms": fp16_lat["mean_ms"],
    })
    rows.append({
        "stage": "pruned30_ft_fp16_onnx",
        "dataset": "summary",
        "psnr": fp16_psnr_h, "ssim": "", "lpips": "",
        "params_M": params_m, "gmacs": gmacs,
        "model_size_mb": onnx_size, "gpu_latency_ms": onnx_lat_gpu,
    })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("\nResults saved: %s", csv_path)

    # Summary table
    log.info("\n" + "=" * 70)
    log.info("  EVAL SUMMARY")
    log.info("=" * 70)
    log.info("  %-20s %8s %8s %8s %10s %10s %10s",
             "Stage", "PSNR-H", "PSNR-L", "GMACs", "Params(M)", "Size(MB)", "Lat(ms)")
    log.info("  " + "-" * 82)
    log.info("  %-20s %8.2f %8.2f %8.1f %10.4f %10.1f %10.2f",
             "Restormer FP32", 31.48, 39.15, 154.9, 26.13, 104.7, 46.35)
    log.info("  %-20s %8.2f %8.2f %8.1f %10.4f %10.1f %10.2f",
             "NAFNet-KD FP32", 30.43, 37.32, 16.1, 29.16, 116.9, 11.26)
    log.info("  %-20s %8.2f %8.2f %8.3f %10.4f %10.1f %10.2f",
             "Pruned30 FP32",
             results["Rain100H"]["psnr"], results["Rain100L"]["psnr"],
             gmacs, params_m, ckpt_size, lat["mean_ms"])
    log.info("  %-20s %8.2f %8.2f %8.3f %10.4f %10.1f %10.2f",
             "Pruned30 FP16",
             fp16_psnr_h, fp16_psnr_l,
             gmacs, params_m, fp16_size, fp16_lat["mean_ms"])
    if not np.isnan(onnx_lat_gpu):
        log.info("  %-20s %8.2f %8s %8.3f %10.4f %10.1f %10.2f",
                 "Pruned30 ONNX", fp16_psnr_h, "-", gmacs, params_m,
                 onnx_size, onnx_lat_gpu)
    log.info("=" * 70)

    del model, model_fp16
    torch.cuda.empty_cache()


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Physical structural pruning of NAFNet-KD via torch-pruning")
    parser.add_argument("--phase", required=True,
                        choices=["prune", "finetune", "eval"],
                        help="Pipeline phase to run")
    parser.add_argument("--pruning_ratio", type=float, default=0.3,
                        help="Fraction of channels to remove (default: 0.3)")
    parser.add_argument("--round_to", type=int, default=8,
                        help="Round channel counts to multiples of this (default: 8)")
    parser.add_argument("--pruned_ckpt", type=str, default=None,
                        help="Path to pruned raw checkpoint (for finetune phase)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (for eval phase)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to full training checkpoint to resume from")
    args = parser.parse_args()

    if args.phase == "prune":
        phase_prune(args.pruning_ratio, args.round_to)
    elif args.phase == "finetune":
        ckpt = args.pruned_ckpt
        if ckpt is None:
            ratio_tag = f"{int(args.pruning_ratio * 100)}"
            ckpt = str(CKPT_DIR / f"nafnet_kd_structpruned{ratio_tag}_raw.pth")
        phase_finetune(ckpt, resume=args.resume)
    elif args.phase == "eval":
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = str(CKPT_DIR / "nafnet_kd_structpruned30_best.pth")
        phase_eval(ckpt)


if __name__ == "__main__":
    main()
