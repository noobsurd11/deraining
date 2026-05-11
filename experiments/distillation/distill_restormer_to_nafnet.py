"""
Knowledge Distillation: Restormer (teacher) -> NAFNet-w32 (student).

Trains a NAFNet-w32 student on Rain13K with a two-term L1 loss:
  L = 1.0 * ||pred - gt||_1 + 0.5 * ||pred - teacher(inp)||_1

Teacher is frozen in eval mode; student is initialized from the trained
NAFNet-w32 checkpoint (not random). EMA (decay 0.999) of the student
parameters is used for validation and the deployed best-weights file.

Runs 50k iterations with AdamW + linear warmup -> cosine decay, mixed
precision (fp16 autocast + GradScaler), validates on Rain100H every
2.5k iters, saves full state every 10k iters and a best-EMA snapshot.

Post-training the script:
  - Copies best EMA weights to pretrained/nafnet_w32_kd.pth,
  - Runs evaluation/evaluate_model.py on all 5 test sets and renames
    the CSV to nafnet_w32_kd_results.csv (preserves the baseline CSV),
  - Prints a summary vs baseline + teacher.
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import DerainDataset
from evaluation.metrics import compute_psnr
from evaluation.model_wrappers import load_model


# ---------- config ----------

SEED = 42
DEVICE = "cuda"

TOTAL_ITERS = 50_000
WARMUP_ITERS = 2_000
LR_PEAK = 2e-4
LR_FLOOR = 1e-6
WEIGHT_DECAY = 1e-3
BETAS = (0.9, 0.9)

BATCH = 16
NUM_WORKERS = 8
CROP = 256

EMA_DECAY = 0.999
W_PIXEL = 1.0
W_DISTILL = 0.5

VAL_EVERY = 2_500
CKPT_EVERY = 10_000
LOG_EVERY = 500

BASELINE_RAIN100H = 30.23
TEACHER_RAIN100H = 31.48

TRAIN_INPUT = PROJECT_ROOT / "datasets/Rain13K/train/input"
TRAIN_TARGET = PROJECT_ROOT / "datasets/Rain13K/train/target"
VAL_INPUT = PROJECT_ROOT / "datasets/Rain13K/test/Rain100H/input"
VAL_TARGET = PROJECT_ROOT / "datasets/Rain13K/test/Rain100H/target"

TEACHER_CKPT = PROJECT_ROOT / "pretrained/restormer_deraining.pth"
STUDENT_CKPT = PROJECT_ROOT / "pretrained/nafnet_w32_deraining.pth"

OUT_DIR = PROJECT_ROOT / "experiments/distillation"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_FILE = OUT_DIR / "train.log"
DEPLOY_CKPT = PROJECT_ROOT / "pretrained/nafnet_w32_kd.pth"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results/baselines/tables"
BASELINE_CSV = RESULTS_TABLE_DIR / "nafnet_w32_results.csv"
KD_CSV = RESULTS_TABLE_DIR / "nafnet_w32_kd_results.csv"


# ---------- setup helpers ----------


def _purge_basicsr() -> None:
    """Clear cached `basicsr.*` so the next load_model picks up the right repo.

    NAFNet and Restormer ship separate bundled `basicsr` trees; Python would
    reuse whichever was imported first. Purge between loads.
    """
    for key in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
        del sys.modules[key]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer", "models/Diff-Mamba"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("distill")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ---------- data ----------


class DistillTrainDataset(Dataset):
    """Rain13K paired loader for distillation training.

    For each sample:
      1. If H or W is below CROP, reflect-pad to at least CROP on that dim.
      2. Shared random CROP x CROP crop on input + target.
      3. Shared random H-flip and V-flip (each p=0.5).
    Tensors in [0,1] float32.
    """

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
        assert len(self.inputs) == len(self.targets), (
            f"pair mismatch: {len(self.inputs)} inputs vs {len(self.targets)} targets"
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def _reflect_pad_to_min(self, t_inp: torch.Tensor, t_tgt: torch.Tensor):
        _, h, w = t_inp.shape
        cs = self.crop
        pad_h = max(0, cs - h)
        pad_w = max(0, cs - w)
        if pad_h == 0 and pad_w == 0:
            return t_inp, t_tgt
        # F.pad on (N,C,H,W) with reflect needs 4D; wrap/unwrap.
        pads = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        t_inp = F.pad(t_inp.unsqueeze(0), pads, mode="reflect").squeeze(0)
        t_tgt = F.pad(t_tgt.unsqueeze(0), pads, mode="reflect").squeeze(0)
        return t_inp, t_tgt

    def __getitem__(self, idx: int):
        inp = self.to_tensor(Image.open(self.inputs[idx]).convert("RGB"))
        tgt = self.to_tensor(Image.open(self.targets[idx]).convert("RGB"))
        inp, tgt = self._reflect_pad_to_min(inp, tgt)
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


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# ---------- EMA ----------


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float = EMA_DECAY) -> None:
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.mul_(decay).add_(mp.detach(), alpha=1 - decay)
    for eb, mb in zip(ema.buffers(), model.buffers()):
        eb.copy_(mb)


# ---------- schedule ----------


def compute_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return LR_PEAK * (step + 1) / WARMUP_ITERS
    progress = (step - WARMUP_ITERS) / max(1, TOTAL_ITERS - WARMUP_ITERS)
    progress = min(1.0, max(0.0, progress))
    return LR_FLOOR + 0.5 * (LR_PEAK - LR_FLOOR) * (1.0 + math.cos(math.pi * progress))


# ---------- validation ----------


@torch.no_grad()
def validate(ema_model: torch.nn.Module, logger: logging.Logger) -> float:
    ds = DerainDataset(str(VAL_INPUT), str(VAL_TARGET))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    ema_model.eval()
    psnrs = []
    for inp, gt, _ in loader:
        inp = inp.to(DEVICE, non_blocking=True)
        gt = gt.to(DEVICE, non_blocking=True)
        pred = torch.clamp(ema_model(inp), 0, 1)
        psnrs.append(compute_psnr(pred, gt))
    return float(np.mean(psnrs))


# ---------- checkpointing ----------


def save_full_ckpt(
    path: Path,
    student: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    current_iter: int,
    best_psnr: float,
) -> None:
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,  # schedule is a pure function of iter
            "scaler_state_dict": scaler.state_dict(),
            "current_iter": current_iter,
            "best_psnr": best_psnr,
        },
        path,
    )


def save_best_ema(path: Path, ema: torch.nn.Module) -> None:
    # Bare state_dict so evaluation/model_wrappers.load_model accepts it directly.
    torch.save(ema.state_dict(), path)


# ---------- post-training ----------


def run_final_eval(logger: logging.Logger) -> dict:
    """Run evaluate_model.py on all 5 testsets, save CSV under a KD name.

    evaluate_model.py writes to a fixed `{model}_results.csv` path; we back
    the baseline CSV up, let the subprocess overwrite, rename to the KD
    filename, then restore the baseline.
    """
    logger.info("Running evaluate_model.py on all 5 test sets ...")
    backup = None
    if BASELINE_CSV.exists():
        backup = BASELINE_CSV.with_suffix(".csv.kdbak")
        shutil.copy2(BASELINE_CSV, backup)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation/evaluate_model.py"),
        "--model", "nafnet_w32",
        "--checkpoint", str(DEPLOY_CKPT),
        "--testset", "all",
        "--output", str(PROJECT_ROOT / "results/baselines/"),
    ]
    logger.info("  cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    produced = BASELINE_CSV  # evaluate_model.py wrote here
    if produced.exists():
        shutil.move(str(produced), str(KD_CSV))
    if backup is not None:
        shutil.move(str(backup), str(BASELINE_CSV))
    logger.info("KD eval CSV saved to %s", KD_CSV)

    rows = _read_csv(KD_CSV)
    return rows


def _read_csv(path: Path) -> dict:
    import csv
    by_dataset = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_dataset[row["dataset"]] = row
    return by_dataset


def print_final_summary(logger: logging.Logger) -> None:
    kd = _read_csv(KD_CSV)
    baseline = _read_csv(BASELINE_CSV) if BASELINE_CSV.exists() else {}
    restormer_csv = RESULTS_TABLE_DIR / "restormer_results.csv"
    teacher = _read_csv(restormer_csv) if restormer_csv.exists() else {}

    kd_h = float(kd["Rain100H"]["psnr"]) if "Rain100H" in kd else float("nan")
    base_h = float(baseline["Rain100H"]["psnr"]) if "Rain100H" in baseline else BASELINE_RAIN100H
    tch_h = float(teacher["Rain100H"]["psnr"]) if "Rain100H" in teacher else TEACHER_RAIN100H

    gain = kd_h - base_h
    gap = tch_h - base_h
    gap_pct = 100.0 * gain / gap if gap > 0 else float("nan")

    # size / speed ratios from params_M and mean_ms (use KD vs teacher rows).
    size_ratio = float("nan")
    speed_ratio = float("nan")
    if "Rain100H" in teacher and "Rain100H" in kd:
        size_ratio = float(teacher["Rain100H"]["params_M"]) / float(kd["Rain100H"]["params_M"])
        speed_ratio = float(teacher["Rain100H"]["mean_ms"]) / float(kd["Rain100H"]["mean_ms"])

    lines = [
        "",
        "=" * 70,
        "  DISTILLATION FINAL SUMMARY",
        "=" * 70,
        f"  NAFNet baseline : {base_h:.2f} dB -> NAFNet-KD : {kd_h:.2f} dB "
        f"(gain: {gain:+.2f} dB)",
        f"  Restormer target: {tch_h:.2f} dB -> gap closed: {gap_pct:.1f}%",
        f"  Student is {size_ratio:.2f}x smaller and {speed_ratio:.2f}x faster than teacher",
        "=" * 70,
        "",
    ]
    for line in lines:
        logger.info(line)


# ---------- training ----------


def train(resume: str | None) -> None:
    set_seed(SEED)
    logger = setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading teacher (Restormer) ...")
    _purge_basicsr()
    teacher = load_model("restormer", str(TEACHER_CKPT), DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    logger.info("Loading student (NAFNet-w32) from baseline checkpoint ...")
    _purge_basicsr()
    student = load_model("nafnet_w32", str(STUDENT_CKPT), DEVICE)
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)

    logger.info("Copying student -> EMA ...")
    ema = copy.deepcopy(student).to(DEVICE)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # Data.
    logger.info("Building train dataset from %s", TRAIN_INPUT)
    ds = DistillTrainDataset(TRAIN_INPUT, TRAIN_TARGET, crop_size=CROP)
    logger.info("  train pairs: %d", len(ds))
    loader = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    it = cycle(loader)

    # Opt / AMP.
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY, betas=BETAS,
    )
    scaler = torch.amp.GradScaler("cuda")

    start_iter = 0
    best_psnr = -1.0
    if resume:
        logger.info("Resuming from %s", resume)
        ck = torch.load(resume, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ck["model_state_dict"])
        ema.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_iter = int(ck["current_iter"]) + 1
        best_psnr = float(ck.get("best_psnr", -1.0))
        logger.info("  resumed at iter=%d, best_psnr=%.3f", start_iter, best_psnr)

    logger.info(
        "Config: total_iters=%d warmup=%d batch=%d workers=%d lr_peak=%.2e lr_floor=%.2e "
        "wd=%.0e betas=%s ema=%.3f w_pixel=%.1f w_distill=%.1f",
        TOTAL_ITERS, WARMUP_ITERS, BATCH, NUM_WORKERS, LR_PEAK, LR_FLOOR,
        WEIGHT_DECAY, BETAS, EMA_DECAY, W_PIXEL, W_DISTILL,
    )

    # Loop.
    student.train()
    pbar = tqdm(range(start_iter, TOTAL_ITERS), initial=start_iter, total=TOTAL_ITERS,
                desc="distill")
    first_val_checked = False
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
            logger.info(msg)

        if (step + 1) % VAL_EVERY == 0:
            psnr_h = validate(ema, logger)
            msg = f"VAL iter={step + 1} psnr={psnr_h:.4f}"
            tqdm.write(msg)
            logger.info(msg)
            logger.info("Validation @ iter %d: PSNR = %.2f dB", step + 1, psnr_h)
            if not first_val_checked:
                first_val_checked = True
                if psnr_h < 25.0:
                    logger.warning(
                        "First validation PSNR %.2f < 25 dB -- something is likely wrong "
                        "(check teacher/student loading, AMP, data ranges).", psnr_h,
                    )
            if psnr_h > best_psnr:
                best_psnr = psnr_h
                save_best_ema(CKPT_DIR / "nafnet_kd_best.pth", ema)
                logger.info("  new best EMA checkpoint saved (PSNR=%.4f)", psnr_h)
            student.train()

        if (step + 1) % CKPT_EVERY == 0:
            full_path = CKPT_DIR / f"nafnet_kd_iter_{step + 1}.pth"
            save_full_ckpt(full_path, student, ema, optimizer, scaler, step, best_psnr)
            logger.info("  full checkpoint saved -> %s", full_path)

    elapsed = time.time() - t0
    logger.info("Training done in %.1f min. best_psnr=%.4f", elapsed / 60.0, best_psnr)

    # Deploy + final eval + summary.
    best_path = CKPT_DIR / "nafnet_kd_best.pth"
    if not best_path.exists():
        logger.warning("No best checkpoint written -- saving final EMA as best.")
        save_best_ema(best_path, ema)
    logger.info("Copying %s -> %s", best_path, DEPLOY_CKPT)
    DEPLOY_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, DEPLOY_CKPT)

    # We no longer need teacher on GPU.
    del teacher
    torch.cuda.empty_cache()

    run_final_eval(logger)
    print_final_summary(logger)


# ---------- cli ----------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a full checkpoint (iter snapshot) to resume from.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training, just run final eval + summary "
                         "(assumes pretrained/nafnet_w32_kd.pth exists).")
    args = ap.parse_args()

    if args.eval_only:
        logger = setup_logging()
        if not DEPLOY_CKPT.exists():
            raise SystemExit(f"--eval-only requires {DEPLOY_CKPT} to exist.")
        run_final_eval(logger)
        print_final_summary(logger)
        return

    train(args.resume)


if __name__ == "__main__":
    main()
