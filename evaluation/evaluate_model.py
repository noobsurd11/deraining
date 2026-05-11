"""
Unified evaluation script for any deraining model.

Usage:
    python evaluation/evaluate_model.py --model restormer \
        --checkpoint pretrained/restormer_deraining.pth \
        --testset all --output results/baselines/
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_psnr, compute_ssim, LPIPSMetric
from evaluation.efficiency import (count_parameters, count_gmacs,
                                    measure_latency, measure_peak_memory,
                                    get_model_size_mb)
from evaluation.dataset import DerainDataset
from evaluation.model_wrappers import load_model


TESTSETS = {
    'Rain100H': 'datasets/Rain13K/test/Rain100H',
    'Rain100L': 'datasets/Rain13K/test/Rain100L',
    'Test100': 'datasets/Rain13K/test/Test100',
    'Test1200': 'datasets/Rain13K/test/Test1200',
    'Test2800': 'datasets/Rain13K/test/Test2800',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_quality(model, testset_name, testset_path, output_dir, num_visual=5):
    """Evaluate quality metrics on a single test set."""
    dataset = DerainDataset(
        input_dir=os.path.join(PROJECT_ROOT, testset_path, 'input'),
        target_dir=os.path.join(PROJECT_ROOT, testset_path, 'target')
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    lpips_metric = LPIPSMetric(DEVICE)

    psnrs, ssims, lpipss = [], [], []
    visual_dir = os.path.join(output_dir, 'visual_samples')
    os.makedirs(visual_dir, exist_ok=True)

    print(f'  Evaluating on {testset_name} ({len(dataset)} images)...')
    for i, (inp, gt, fname) in enumerate(tqdm(loader, desc=f'  {testset_name}')):
        inp, gt = inp.to(DEVICE), gt.to(DEVICE)

        with torch.no_grad():
            pred = model(inp)
            pred = torch.clamp(pred, 0, 1)

        psnrs.append(compute_psnr(pred, gt))
        ssims.append(compute_ssim(pred, gt))
        lpipss.append(lpips_metric.compute(pred, gt))

        # Save visual samples
        if i < num_visual:
            comparison = torch.cat([inp, pred, gt], dim=3)  # side by side
            save_image(comparison, os.path.join(visual_dir,
                       f'{testset_name}_{fname[0]}'), nrow=1)

    results = {
        'dataset': testset_name,
        'psnr': sum(psnrs) / len(psnrs),
        'ssim': sum(ssims) / len(ssims),
        'lpips': sum(lpipss) / len(lpipss),
        'num_images': len(dataset),
    }
    return results


def evaluate_efficiency(model, checkpoint_path):
    """Compute all efficiency metrics."""
    return {
        'params_M': count_parameters(model),
        'gmacs': count_gmacs(model),
        **measure_latency(model),
        'peak_mem_GB': measure_peak_memory(model),
        'model_size_MB': get_model_size_mb(checkpoint_path) if os.path.exists(checkpoint_path) else -1,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate deraining model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['restormer', 'drsformer', 'nafnet_w32', 'nafnet_w64', 'diffmamba'],
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--testset', type=str, default='all',
                        help='Test set name or "all"')
    parser.add_argument('--output', type=str, default='results/baselines/',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, args.output)
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # Load model
    ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    print(f'\n{"="*60}')
    print(f'Model: {args.model}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'{"="*60}')
    model = load_model(args.model, ckpt_path, DEVICE)

    # Efficiency metrics (once)
    print('\nComputing efficiency metrics...')
    eff = evaluate_efficiency(model, ckpt_path)
    print(f'  Params: {eff["params_M"]:.2f}M | GMACs: {eff["gmacs"]:.1f} | '
          f'Latency: {eff["mean_ms"]:.1f}ms | Peak Mem: {eff["peak_mem_GB"]:.2f}GB')

    # Quality metrics (per test set)
    testsets = TESTSETS if args.testset == 'all' else {args.testset: TESTSETS[args.testset]}
    all_results = []

    for name, path in testsets.items():
        result = evaluate_quality(model, name, path, output_dir)
        result.update(eff)
        result['model'] = args.model
        all_results.append(result)
        print(f'  {name}: PSNR={result["psnr"]:.2f} | SSIM={result["ssim"]:.4f} | '
              f'LPIPS={result["lpips"]:.4f}')

    # Save to CSV
    csv_path = os.path.join(tables_dir, f'{args.model}_results.csv')
    fieldnames = ['model', 'dataset', 'psnr', 'ssim', 'lpips', 'num_images',
                  'params_M', 'gmacs', 'mean_ms', 'std_ms', 'peak_mem_GB', 'model_size_MB']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
