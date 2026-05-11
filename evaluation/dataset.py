"""
Unified PyTorch Dataset for deraining test sets.
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DerainDataset(Dataset):
    """
    Loads paired input/target images for deraining evaluation.

    Args:
        input_dir: Path to rainy images
        target_dir: Path to clean ground truth images
    """
    def __init__(self, input_dir: str, target_dir: str):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.transform = transforms.ToTensor()  # Converts to [0,1] float32

        # Get sorted file lists
        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.input_files = sorted([
            f for f in self.input_dir.iterdir()
            if f.suffix.lower() in valid_ext
        ])
        self.target_files = sorted([
            f for f in self.target_dir.iterdir()
            if f.suffix.lower() in valid_ext
        ])
        assert len(self.input_files) == len(self.target_files), \
            f"Mismatch: {len(self.input_files)} inputs vs {len(self.target_files)} targets"

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int) -> tuple:
        inp = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        tgt = self.transform(Image.open(self.target_files[idx]).convert('RGB'))
        return inp, tgt, self.input_files[idx].name
