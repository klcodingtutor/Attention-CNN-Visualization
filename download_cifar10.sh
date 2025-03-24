#!/bin/bash

# download_cifar10.sh

# Script to download and validate CIFAR-10 dataset

# Parameters
DATA_ROOT="./cifar10_data"
BATCH_SIZE=4
NUM_WORKERS=2

# Create data directory if it doesn't exist
mkdir -p "$DATA_ROOT"

# Run the download and validation script
python download_cifar10.py

echo "CIFAR-10 download and validation script executed!"