#!/bin/bash

# Script to train MultiViewAttentionCNN on CIFAR-10 dataset

# Parameters
DATA_FOLDER="./cifar10_data"  # Directory to store CIFAR-10 data (downloaded automatically by torchvision)
NUM_CLASSES=10                # CIFAR-10 has 10 classes
IMG_SIZE=32                   # CIFAR-10 images are 32x32
BATCH_SIZE=32                 # Batch size for training
LEARNING_RATE=0.001           # Learning rate for Adam optimizer
EPOCHS=20                     # Total epochs (divided across stages)
DROPOUT_RATE=0.5              # Dropout probability
NUM_WORKERS=4                 # Number of workers for DataLoader
DEVICE="gpu"                  # Use GPU if available
MODEL_SAVE_PATH="./models"    # Directory to save the model

# Create directories if they don't exist
mkdir -p "$DATA_FOLDER"
mkdir -p "$MODEL_SAVE_PATH"

# Run the training script
python train_script.py \
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