"""
Model efficiency metrics: params, GMACs, latency, memory, size.
"""
import os
import time
import torch
from fvcore.nn import FlopCountAnalysis


def count_parameters(model: torch.nn.Module) -> float:
    """Total trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_gmacs(model: torch.nn.Module, input_size: tuple = (1, 3, 256, 256),
                device: str = 'cuda') -> float:
    """GMACs using fvcore. Returns GMACs (= FLOPs / 2e9)."""
    dummy = torch.randn(*input_size, device=device)
    model = model.to(device).eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9  # GMACs (fvcore reports MACs)


def measure_latency(model: torch.nn.Module, input_size: tuple = (1, 3, 256, 256),
                    device: str = 'cuda', warmup: int = 10, runs: int = 100) -> dict:
    """Returns dict with mean_ms and std_ms."""
    dummy = torch.randn(*input_size, device=device)
    model = model.to(device).eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return {'mean_ms': t.mean().item(), 'std_ms': t.std().item()}


def measure_peak_memory(model: torch.nn.Module, input_size: tuple = (1, 3, 256, 256),
                        device: str = 'cuda') -> float:
    """Peak GPU memory in GB during single forward pass."""
    dummy = torch.randn(*input_size, device=device)
    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy)
    return torch.cuda.max_memory_allocated(device) / 1e9


def get_model_size_mb(checkpoint_path: str) -> float:
    """File size in MB."""
    return os.path.getsize(checkpoint_path) / 1e6
