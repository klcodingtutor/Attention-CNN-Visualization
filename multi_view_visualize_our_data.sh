#!/bin/bash

# Script to visualize MultiViewAttentionCNN on custom face image dataset test split

# Parameters
DATA_FOLDER="C:/Users/megah/Dropbox/Prompt/self_attention_face"  # Directory containing face images
CSV_FILE="C:/Users/megah/Dropbox/Prompt/face_images_path_with_meta_jpg_exist_only.csv"  # CSV with metadata
TASK="gender"                 # Task to visualize: 'gender', 'age_10', 'age_5', or 'disease'
NUM_CLASSES=""                # Number of classes (set dynamically based on TASK)
IMG_SIZE=32                  # Image size after resizing (from your transforms)
DROPOUT_RATE=0.5              # Dropout rate (assumed from your model)
NUM_WORKERS=4                 # Number of workers for DataLoader
DEVICE="gpu"                  # Either "gpu" or "cpu"
MODEL_SAVE_PATH="./models"    # Directory where trained models are saved

# Ensure the data and model directories exist
if [ ! -d "$DATA_FOLDER" ]; then
    echo "Data folder $DATA_FOLDER does not exist! Please check the path."
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "CSV file $CSV_FILE not found! Please check the path."
    exit 1
fi

mkdir -p "$MODEL_SAVE_PATH"
if [ ! -d "$MODEL_SAVE_PATH" ]; then
    echo "Failed to create model save path $MODEL_SAVE_PATH!"
    exit 1
fi

# Set NUM_CLASSES dynamically based on TASK (approximate values based on typical datasets)
case "$TASK" in
    "gender")
        NUM_CLASSES=2  # Typically 2 classes: male, female
        ;;
    "age_10")
        NUM_CLASSES=10  # Assuming age divided by 10, rounded (e.g., 0-90 -> ~10 classes)
        ;;
    "age_5")
        NUM_CLASSES=20  # Assuming age divided by 5, rounded (e.g., 0-95 -> ~20 classes)
        ;;
    "disease")
        NUM_CLASSES=5  # Placeholder; adjust based on your actual dataset
        ;;
    *)
        echo "Invalid TASK: $TASK. Must be 'gender', 'age_10', 'age_5', or 'disease'."
        exit 1
        ;;
esac

# Define the model file path based on the task
MODEL_FILE="$MODEL_SAVE_PATH/multi_view_attention_cnn_${TASK}.pth"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Trained model file $MODEL_FILE not found! Please train the model for task '$TASK' first."
    exit 1
fi

# Adjust path for Windows compatibility if running in a Windows environment (e.g., Git Bash)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    DATA_FOLDER=$(echo "$DATA_FOLDER" | sed 's|/|\\|g')
    CSV_FILE=$(echo "$CSV_FILE" | sed 's|/|\\|g')
    MODEL_SAVE_PATH=$(echo "$MODEL_SAVE_PATH" | sed 's|/|\\|g')
    MODEL_FILE=$(echo "$MODEL_FILE" | sed 's|/|\\|g')
fi

echo "Starting visualization script with the following parameters:"
echo "python multi_view_visualize.py \\"
echo "    --data_folder \"$DATA_FOLDER\" \\"
echo "    --csv_file \"$CSV_FILE\" \\"
echo "    --task \"$TASK\" \\"
echo "    --num_classes $NUM_CLASSES \\"
echo "    --img_size $IMG_SIZE \\"
echo "    --dropout_rate $DROPOUT_RATE \\"
echo "    --num_workers $NUM_WORKERS \\"
echo "    --device \"$DEVICE\" \\"
echo "    --model_save_path \"$MODEL_SAVE_PATH\""

# Run the visualization script
python multi_view_visualize.py \
    --data_folder "$DATA_FOLDER" \
    --csv_file "$CSV_FILE" \
    --task "$TASK" \
    --num_classes "$NUM_CLASSES" \
    --img_size "$IMG_SIZE" \
    --dropout_rate "$DROPOUT_RATE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --model_save_path "$MODEL_SAVE_PATH"

echo "Visualization completed!"