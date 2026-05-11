#!/bin/bash
# Download Rain13K train + test sets via Restormer's download_data.py,
# then arrange them under /home/user/noob/deraining/datasets/Rain13K/
# following the {train,test/<subset>}/{input,target} layout.
set -euo pipefail

PROJECT_ROOT=/home/user/noob/deraining
DATA_ROOT=$PROJECT_ROOT/datasets/Rain13K
RESTORMER_DERAIN=$PROJECT_ROOT/models/Restormer/Deraining

cd "$RESTORMER_DERAIN"

# Google Drive IDs come from Restormer's download_data.py
RAIN13K_TRAIN_ID=14BidJeG4nSNuFNFDf99K-7eErCq4i47t
RAIN13K_TEST_ID=1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8

mkdir -p Datasets

if [[ ! -d Datasets/train ]]; then
    echo "Downloading Rain13K train..."
    gdown "$RAIN13K_TRAIN_ID" -O Datasets/train.zip
    unzip -q Datasets/train.zip -d Datasets/
    rm Datasets/train.zip
fi

if [[ ! -d Datasets/test ]]; then
    echo "Downloading Rain13K test..."
    gdown "$RAIN13K_TEST_ID" -O Datasets/test.zip
    unzip -q Datasets/test.zip -d Datasets/
    rm Datasets/test.zip
fi

echo "Arranging into $DATA_ROOT layout..."
# Train: symlink input/target into DATA_ROOT/train.
# Restormer's archive extracts to Datasets/train/Rain13K/{input,target}.
mkdir -p "$DATA_ROOT/train"
rm -rf "$DATA_ROOT/train/input" "$DATA_ROOT/train/target"
ln -sfn "$RESTORMER_DERAIN/Datasets/train/Rain13K/input"  "$DATA_ROOT/train/input"
ln -sfn "$RESTORMER_DERAIN/Datasets/train/Rain13K/target" "$DATA_ROOT/train/target"

# Test: each subset has input/target
for subset in Rain100H Rain100L Test100 Test1200 Test2800; do
    src="$RESTORMER_DERAIN/Datasets/test/$subset"
    dst="$DATA_ROOT/test/$subset"
    mkdir -p "$dst"
    rm -rf "$dst/input" "$dst/target"
    ln -sfn "$src/input"  "$dst/input"
    ln -sfn "$src/target" "$dst/target"
done

echo "Done."
