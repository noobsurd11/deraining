"""
Image quality metrics for deraining evaluation.
All metrics computed on Y channel (YCbCr) with 4-pixel border crop.
All inputs: PyTorch tensors (B, C, H, W) in [0, 1] range.
"""
import torch
import torch.nn.functional as F
import lpips as lpips_module


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to Y channel [0,1]. Standard BT.601."""
    # Y = 16/255 + 65.481/255*R + 128.553/255*G + 24.966/255*B
    coeffs = torch.tensor([65.481, 128.553, 24.966],
                          device=img.device, dtype=img.dtype).view(1, 3, 1, 1) / 255.0
    return (img * coeffs).sum(dim=1, keepdim=True) + 16.0 / 255.0


def crop_border(img: torch.Tensor, border: int = 4) -> torch.Tensor:
    """Crop border pixels."""
    return img[..., border:-border, border:-border]


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR on Y channel with 4px border crop. Returns dB."""
    pred_y = crop_border(rgb_to_y(pred))
    gt_y = crop_border(rgb_to_y(gt))
    mse = F.mse_loss(pred_y, gt_y)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    """SSIM on Y channel with 4px border crop. Pure PyTorch implementation."""
    pred_y = crop_border(rgb_to_y(pred))
    gt_y = crop_border(rgb_to_y(gt))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)

    mu1 = F.conv2d(pred_y, window, padding=window_size // 2)
    mu2 = F.conv2d(gt_y, window, padding=window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred_y ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(gt_y ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred_y * gt_y, window, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


class LPIPSMetric:
    """Wrapper for LPIPS. Initialize once, call many times."""
    def __init__(self, device: str = 'cuda'):
        self.fn = lpips_module.LPIPS(net='vgg').to(device).eval()

    @torch.no_grad()
    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """LPIPS between pred and gt. Input [0,1], internally scaled to [-1,1]."""
        return self.fn(pred * 2 - 1, gt * 2 - 1).item()
