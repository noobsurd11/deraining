"""
Unified model loading interface for all 4 deraining models.
Each wrapper imports from the cloned repo, loads weights, returns model.eval().
"""
import importlib.util
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'


def load_model(model_name: str, checkpoint_path: str, device: str = 'cuda') -> torch.nn.Module:
    """
    Load a deraining model by name.

    Args:
        model_name: One of 'restormer', 'drsformer', 'nafnet_w32', 'nafnet_w64', 'diffmamba'
        checkpoint_path: Path to .pth weights file
        device: 'cuda' or 'cpu'

    Returns:
        model in eval mode on device
    """
    if model_name == 'restormer':
        return _load_restormer(checkpoint_path, device)
    elif model_name == 'drsformer':
        return _load_drsformer(checkpoint_path, device)
    elif model_name.startswith('nafnet'):
        width = 64 if 'w64' in model_name else 32
        return _load_nafnet(checkpoint_path, device, width=width)
    elif model_name == 'diffmamba':
        return _load_diffmamba(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _apply_state_dict(model, ckpt_path, device):
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt_data.get('params', ckpt_data.get('state_dict', ckpt_data)) \
        if isinstance(ckpt_data, dict) else ckpt_data
    # Some checkpoints prefix params_ema
    if isinstance(state_dict, dict) and 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    model.load_state_dict(state_dict, strict=True)
    return model


def _load_restormer(ckpt: str, device: str) -> torch.nn.Module:
    sys.path.insert(0, str(MODELS_DIR / 'Restormer'))
    from basicsr.models.archs.restormer_arch import Restormer
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias',
        dual_pixel_task=False
    )
    model = _apply_state_dict(model, ckpt, device)
    return model.to(device).eval()


def _load_drsformer(ckpt: str, device: str) -> torch.nn.Module:
    sys.path.insert(0, str(MODELS_DIR / 'DRSformer'))
    from basicsr.models.archs.DRSformer_arch import DRSformer
    model = DRSformer()
    model = _apply_state_dict(model, ckpt, device)
    return model.to(device).eval()


def _load_nafnet(ckpt: str, device: str, width: int = 32) -> torch.nn.Module:
    sys.path.insert(0, str(MODELS_DIR / 'NAFNet'))
    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(
        img_channel=3, width=width,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )
    model = _apply_state_dict(model, ckpt, device)
    return model.to(device).eval()


def _load_diffmamba(ckpt: str, device: str) -> torch.nn.Module:
    """
    Load the Diff-Mamba DFT backbone.

    Notes
    -----
    The upstream repo is structured around diffusion-based coarse-to-fine
    sampling (inp_channels=6, conditioned on a coarse estimate + noisy target).
    A single-forward `model(x)` call from the evaluation script will NOT
    produce a derained image — you need the diffusion sampler from the repo's
    Deraining/test.py. This wrapper only instantiates the arch and loads
    weights so efficiency metrics (params/GMACs/latency) can be measured.
    """
    dm_root = MODELS_DIR / 'Diff-Mamba'
    # Make 'basicsr.models.archs.vmamba' / '...model' importable for Diff-Mamba.py's imports.
    sys.path.insert(0, str(dm_root))
    # Register a synthetic 'basicsr.models.archs' package mapped to Diff-Mamba/models/archs/
    archs_dir = dm_root / 'models' / 'archs'
    for pkg in ('basicsr', 'basicsr.models', 'basicsr.models.archs'):
        if pkg not in sys.modules:
            import types
            m = types.ModuleType(pkg)
            if pkg == 'basicsr.models.archs':
                m.__path__ = [str(archs_dir)]
            else:
                m.__path__ = []
            sys.modules[pkg] = m
    # Preload siblings required by Diff-Mamba.py
    for mod_name, file_name in [
        ('basicsr.models.archs.vmamba', 'vmamba.py'),
        ('basicsr.models.archs.model', 'model.py'),
    ]:
        if mod_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(mod_name, archs_dir / file_name)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
    # Load the DFT class from the hyphenated file
    spec = importlib.util.spec_from_file_location(
        'basicsr.models.archs.DFT_arch', archs_dir / 'Diff-Mamba.py'
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['basicsr.models.archs.DFT_arch'] = mod
    spec.loader.exec_module(mod)
    DFT = mod.DFT

    model = DFT(
        inp_channels=6, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
        ffn_factor=4.0, bias=False, LayerNorm_type='WithBias',
        dual_pixel_task=False,
    )
    model = _apply_state_dict(model, ckpt, device)
    return model.to(device).eval()
