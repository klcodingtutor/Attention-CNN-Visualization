#!/bin/bash

# Script to train MultiViewAttentionCNN on face image dataset for multiple tasks

# Parameters
DATA_FOLDER="./data"          # Directory containing face images and CSV
IMG_SIZE=224                  # Face images are resized to 224x224
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=20                     # Will be divided by 4 for multi-stage training
DROPOUT_RATE=0.5
NUM_WORKERS=4
DEVICE="gpu"                  # Either "gpu" or "cpu"
MODEL_SAVE_PATH="./models"

# Create directories if they don't exist
mkdir -p "$DATA_FOLDER"
mkdir -p "$MODEL_SAVE_PATH"

echo "Starting Training script with the following parameters:"
echo "python multi_view_train_script.py \\"
echo "    --data_folder $DATA_FOLDER \\"
echo "    --img_size $IMG_SIZE \\"
echo "    --batch_size $BATCH_SIZE \\"
echo "    --learning_rate $LEARNING_RATE \\"
echo "    --epoch $EPOCHS \\"
echo "    --dropout_rate $DROPOUT_RATE \\"
echo "    --num_workers $NUM_WORKERS \\"
echo "    --device $DEVICE \\"
echo "    --model_save_path $MODEL_SAVE_PATH"

# Run the training script
python multi_view_train_script.py \
    --data_folder "$DATA_FOLDER" \
    --img_size "$IMG_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --epoch "$EPOCHS" \
    --dropout_rate "$DROPOUT_RATE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --model_save_path "$MODEL_SAVE_PATH"

echo "Training completed!"