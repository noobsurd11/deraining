#!/bin/bash
cd /home/user/noob/deraining

echo "=== Evaluating Restormer ==="
python evaluation/evaluate_model.py --model restormer \
    --checkpoint pretrained/restormer_deraining.pth --testset all

echo "=== Evaluating DRSformer ==="
python evaluation/evaluate_model.py --model drsformer \
    --checkpoint pretrained/drsformer_deraining.pth --testset all

# echo "=== Evaluating NAFNet-w32 ==="
# python evaluation/evaluate_model.py --model nafnet_w32 \
#     --checkpoint pretrained/nafnet_w32_deraining.pth --testset all

# echo "=== Evaluating Diff-Mamba ==="
# python evaluation/evaluate_model.py --model diffmamba \
#     --checkpoint pretrained/diffmamba_deraining.pth --testset all

echo "=== Generating Plots ==="
python results/baselines/generate_plots.py

echo "=== Done ==="
