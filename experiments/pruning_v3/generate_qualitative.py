"""
Generate qualitative comparison figures for Rain100H.

For each selected image, produces a strip:
  Input (rainy) | Restormer | NAFNet-KD | NAFNet-KD-Pruned | Ground Truth

Each prediction panel shows PSNR overlay. A zoomed inset patch (red box)
highlights fine detail differences.

Outputs: results/baselines/visual_samples/v3/comparison_{01-05}.{png,pdf}
         results/baselines/visual_samples/v3/failure_01.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_psnr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAIN_DIR = PROJECT_ROOT / "datasets" / "Rain13K" / "test" / "Rain100H"
OUT_DIR = PROJECT_ROOT / "results" / "baselines" / "visual_samples" / "v3"

RESTORMER_CKPT = PROJECT_ROOT / "pretrained" / "restormer_deraining.pth"
NAFNET_KD_CKPT = PROJECT_ROOT / "pretrained" / "nafnet_w32_kd.pth"
PRUNED_CKPT = Path(__file__).resolve().parent / "checkpoints" / "nafnet_kd_structpruned30_best.pth"

PADDER = 16
to_tensor = transforms.ToTensor()


def _purge_basicsr():
    for k in [m for m in sys.modules if m == "basicsr" or m.startswith("basicsr.")]:
        del sys.modules[k]
    for pat in ("models/Restormer", "models/NAFNet", "models/DRSformer"):
        sys.path[:] = [p for p in sys.path if pat not in p]


def load_restormer():
    _purge_basicsr()
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "Restormer"))
    from basicsr.models.archs.restormer_arch import Restormer
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
    )
    sd = torch.load(str(RESTORMER_CKPT), map_location="cpu", weights_only=False)
    sd = sd.get("params", sd.get("state_dict", sd))
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


def load_nafnet_kd():
    _purge_basicsr()
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "NAFNet"))
    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(
        img_channel=3, width=32, middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )
    sd = torch.load(str(NAFNET_KD_CKPT), map_location="cpu", weights_only=False)
    sd = sd.get("params", sd.get("state_dict", sd))
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


def load_pruned():
    _purge_basicsr()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from structured_pruning_physical import _build_pruned_student
    model = _build_pruned_student(device="cpu")
    sd = torch.load(str(PRUNED_CKPT), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


def pad_input(x):
    _, _, h, w = x.shape
    mod_h = (PADDER - h % PADDER) % PADDER
    mod_w = (PADDER - w % PADDER) % PADDER
    if mod_h or mod_w:
        x = torch.nn.functional.pad(x, (0, mod_w, 0, mod_h), mode="reflect")
    return x, h, w


@torch.no_grad()
def predict(model, inp_tensor):
    x, h, w = pad_input(inp_tensor.to(DEVICE))
    out = model(x)[:, :, :h, :w]
    return torch.clamp(out, 0, 1).cpu()


def to_numpy(t):
    return t.squeeze(0).permute(1, 2, 0).numpy()


def find_best_crop(inp_np, gt_np, pred_np, crop_size=80):
    """Find the crop region with highest rain intensity (input-GT difference)."""
    diff = np.mean(np.abs(inp_np - gt_np), axis=2)
    h, w = diff.shape
    best_score, best_y, best_x = -1, 0, 0
    step = crop_size // 2
    for y in range(0, h - crop_size, step):
        for x in range(0, w - crop_size, step):
            score = diff[y:y+crop_size, x:x+crop_size].mean()
            if score > best_score:
                best_score = score
                best_y, best_x = y, x
    return best_y, best_x, crop_size


def draw_strip(inp_np, preds, gt_np, labels, psnrs, crop_box, title, out_path_stem):
    """Draw a comparison strip with zoomed inset patches."""
    n_panels = len(preds) + 2  # input + preds + GT
    all_imgs = [inp_np] + preds + [gt_np]
    all_labels = ["Input (Rainy)"] + labels + ["Ground Truth"]
    all_psnrs = [None] + psnrs + [None]

    cy, cx, cs = crop_box
    inset_scale = 2.5

    fig_w = 4.0 * n_panels
    fig_h = 4.8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), dpi=150)

    for i, (img, label, psnr_val) in enumerate(zip(all_imgs, all_labels, all_psnrs)):
        ax = axes[i]
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

        # Red rectangle on main image
        rect = patches.Rectangle(
            (cx, cy), cs, cs, linewidth=1.5,
            edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)

        # Zoomed inset in bottom-right
        crop = img[cy:cy+cs, cx:cx+cs]
        ih, iw = img.shape[:2]
        inset_px = int(cs * inset_scale)
        ax_ins = ax.inset_axes(
            [1.0 - inset_px/iw - 0.02, 0.02, inset_px/iw, inset_px/ih],
            transform=ax.transAxes,
        )
        ax_ins.imshow(np.clip(crop, 0, 1))
        ax_ins.axis("off")
        for spine in ax_ins.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)

        # PSNR overlay
        if psnr_val is not None:
            ax.text(
                0.03, 0.97, f"PSNR: {psnr_val:.2f}",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("png", "pdf"):
        out = f"{out_path_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path_stem}.{{png,pdf}}")


def compute_per_image_psnrs(gt_tensor, preds_tensors):
    return [compute_psnr(p, gt_tensor) for p in preds_tensors]


def rank_images_by_difficulty(restormer, nafnet_kd, pruned):
    """Rank Rain100H images by rain severity and find failure cases."""
    print("Scanning all 100 Rain100H images for difficulty ranking...")
    records = []
    for idx in range(1, 101):
        inp_path = RAIN_DIR / "input" / f"{idx}.png"
        gt_path = RAIN_DIR / "target" / f"{idx}.png"
        if not inp_path.exists() or not gt_path.exists():
            continue
        inp_t = to_tensor(Image.open(inp_path).convert("RGB")).unsqueeze(0)
        gt_t = to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0)

        p_rest = predict(restormer, inp_t)
        p_kd = predict(nafnet_kd, inp_t)
        p_prune = predict(pruned, inp_t)

        psnr_rest = compute_psnr(p_rest, gt_t)
        psnr_kd = compute_psnr(p_kd, gt_t)
        psnr_prune = compute_psnr(p_prune, gt_t)

        rain_severity = np.mean(np.abs(to_numpy(inp_t) - to_numpy(gt_t)))
        prune_gap = psnr_kd - psnr_prune

        records.append({
            "idx": idx, "rain_severity": rain_severity,
            "psnr_rest": psnr_rest, "psnr_kd": psnr_kd, "psnr_prune": psnr_prune,
            "prune_gap": prune_gap,
            "inp_t": inp_t, "gt_t": gt_t,
            "p_rest": p_rest, "p_kd": p_kd, "p_prune": p_prune,
        })
        if idx % 20 == 0:
            print(f"    {idx}/100 done")

    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Restormer...")
    restormer = load_restormer()
    print("Loading NAFNet-KD...")
    nafnet_kd = load_nafnet_kd()
    print("Loading NAFNet-KD-StructPrune30...")
    pruned = load_pruned()

    records = rank_images_by_difficulty(restormer, nafnet_kd, pruned)

    # Sort by rain severity (heaviest first) for challenging examples
    by_severity = sorted(records, key=lambda r: r["rain_severity"], reverse=True)

    # Pick top 5 most challenging images
    chosen = by_severity[:5]
    print(f"\nSelected challenging images: {[r['idx'] for r in chosen]}")
    for i, r in enumerate(chosen):
        print(f"  Image {r['idx']}: severity={r['rain_severity']:.4f}  "
              f"PSNR rest={r['psnr_rest']:.2f} kd={r['psnr_kd']:.2f} prune={r['psnr_prune']:.2f}")

    # Generate comparison strips
    for i, r in enumerate(chosen):
        inp_np = to_numpy(r["inp_t"])
        gt_np = to_numpy(r["gt_t"])
        preds_np = [to_numpy(r["p_rest"]), to_numpy(r["p_kd"]), to_numpy(r["p_prune"])]
        psnrs = [r["psnr_rest"], r["psnr_kd"], r["psnr_prune"]]
        labels = ["Restormer", "NAFNet-KD", "KD+Prune30"]

        crop_box = find_best_crop(inp_np, gt_np, preds_np[0])
        title = f"Rain100H #{r['idx']} — Qualitative Comparison"
        stem = str(OUT_DIR / f"comparison_{i+1:02d}")
        draw_strip(inp_np, preds_np, gt_np, labels, psnrs, crop_box, title, stem)

    # Find failure case: largest PSNR gap (pruned worst vs unpruned)
    by_gap = sorted(records, key=lambda r: r["prune_gap"], reverse=True)
    fail = by_gap[0]
    print(f"\nFailure case: image {fail['idx']} (prune_gap={fail['prune_gap']:.2f} dB)")
    print(f"  PSNR rest={fail['psnr_rest']:.2f} kd={fail['psnr_kd']:.2f} prune={fail['psnr_prune']:.2f}")

    inp_np = to_numpy(fail["inp_t"])
    gt_np = to_numpy(fail["gt_t"])
    preds_np = [to_numpy(fail["p_rest"]), to_numpy(fail["p_kd"]), to_numpy(fail["p_prune"])]
    psnrs = [fail["psnr_rest"], fail["psnr_kd"], fail["psnr_prune"]]
    labels = ["Restormer", "NAFNet-KD", "KD+Prune30"]
    crop_box = find_best_crop(inp_np, gt_np, preds_np[1])
    title = f"Rain100H #{fail['idx']} — Failure Case (Pruned {fail['prune_gap']:+.2f} dB vs Unpruned)"
    stem = str(OUT_DIR / "failure_01")
    draw_strip(inp_np, preds_np, gt_np, labels, psnrs, crop_box, title, stem)

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'Image':>6s}  {'Severity':>8s}  {'Restormer':>10s}  {'NAFNet-KD':>10s}  {'Pruned':>10s}  {'Gap':>6s}")
    print("-" * 70)
    for r in chosen + [fail]:
        print(f"  {r['idx']:>6d}  {r['rain_severity']:8.4f}  {r['psnr_rest']:10.2f}  {r['psnr_kd']:10.2f}  {r['psnr_prune']:10.2f}  {r['prune_gap']:+6.2f}")
    print("=" * 70)

    del restormer, nafnet_kd, pruned
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
