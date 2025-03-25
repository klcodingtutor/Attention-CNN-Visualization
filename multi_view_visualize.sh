#!/bin/bash

# Script to visualize MultiViewAttentionCNN on CIFAR-10 test dataset

# Parameters
DATA_FOLDER="./cifar10_data"  # Directory containing CIFAR-10 data
NUM_CLASSES=10                # CIFAR-10 has 10 classes
IMG_SIZE=224                   # CIFAR-10 images are 32x32
DROPOUT_RATE=0.5              # Dropout rate used in training
NUM_WORKERS=4                 # Number of workers for DataLoader
DEVICE="gpu"                  # Either "gpu" or "cpu"
MODEL_SAVE_PATH="./models"    # Directory where the trained model is saved

# Ensure the data and model directories exist
mkdir -p "$DATA_FOLDER"
if [ ! -d "$MODEL_SAVE_PATH" ]; then
    echo "Model save path $MODEL_SAVE_PATH does not exist! Please train the model first."
    exit 1
fi

# Check if the trained model exists
MODEL_FILE="$MODEL_SAVE_PATH/multi_view_attention_cnn_cifar10.pth"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Trained model file $MODEL_FILE not found! Please train the model first."
    exit 1
fi

# Run the visualization script
python multi_view_visualize.py \
    --data_folder "$DATA_FOLDER" \
    --num_classes "$NUM_CLASSES" \
    --img_size "$IMG_SIZE" \
    --dropout_rate "$DROPOUT_RATE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --model_save_path "$MODEL_SAVE_PATH"

echo "Visualization completed!"