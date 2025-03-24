#!/bin/bash

# Script to train MultiViewAttentionCNN on CIFAR-10 dataset

# Parameters
DATA_FOLDER="./cifar10_data"  # Directory to store CIFAR-10 data
NUM_CLASSES=10                # CIFAR-10 has 10 classes
IMG_SIZE=32                   # CIFAR-10 images are 32x32
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=4   # will div by 4 for multi-stage
DROPOUT_RATE=0.5
NUM_WORKERS=4
DEVICE="gpu"    # either "gpu" or "cpu"
MODEL_SAVE_PATH="./models"

# Create directories if they don't exist (no train/test subdirs)
mkdir -p "$DATA_FOLDER"
mkdir -p "$MODEL_SAVE_PATH"

# Run the training script
python multi_view_train_script.py \
    --data_folder "$DATA_FOLDER" \
    --num_classes "$NUM_CLASSES" \
    --img_size "$IMG_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --epoch "$EPOCHS" \
    --dropout_rate "$DROPOUT_RATE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --model_save_path "$MODEL_SAVE_PATH"

echo "Training completed!"