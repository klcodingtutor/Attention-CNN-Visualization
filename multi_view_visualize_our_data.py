import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from attention_cnn import MultiViewAttentionCNN  # Your model class
from runtime_args import args  # Runtime arguments from training

# Define CustomDataset (copied from training script for standalone use)
class CustomDataset(Dataset):
    def __init__(self, dataframe, task, image_folder, transform=None):
        self.dataframe = dataframe
        self.task = task
        self.transform = transform
        self.image_folder = image_folder
        
        valid_tasks = ['gender', 'age_10', 'age_5', 'disease']
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of {valid_tasks}, got {task}")

        # Create label mappings based on the task
        if self.task == 'gender':
            self.label_col = 'gender'
            unique_labels = self.dataframe[self.label_col].unique()
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        elif self.task == 'age_10':
            self.label_col = 'age_div_10_round'
            unique_labels = sorted(self.dataframe[self.label_col].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        elif self.task == 'age_5':
            self.label_col = 'age_div_5_round'
            unique_labels = sorted(self.dataframe[self.label_col].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        elif self.task == 'disease':
            self.label_col = 'disease'
            unique_labels = self.dataframe[self.label_col].unique()
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Print number of classes for verification
        print(f"Task: {self.task}, Number of classes: {len(self.label_to_idx)}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_folder, self.dataframe.iloc[idx]['dest_filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Get label using the internal mapping
        label = self.label_to_idx[self.dataframe.iloc[idx][self.label_col]]

        return image, label

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Load trained model
tasks = ['gender', 'age_10', 'disease']
num_classes_list = []  # Will be populated from dataloaders
model_path = os.path.join(args.model_save_path, 'multi_view_attention_cnn_face_tasks.pth')
image_size = 32  # From training script

# Define test transform consistent with training
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CSV and create dataloaders
import pandas as pd
df = pd.read_csv('./data/face_images_path_with_meta_jpg_exist_only.csv')
train_df = df[df['split'] == 'train']
image_folder = './data'

dataloaders = {}
for task in tasks:
    dataset = CustomDataset(train_df, task=task, image_folder=image_folder, transform=test_transform)
    num_classes_list.append(len(dataset.label_to_idx))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    dataloaders[f'train_{task}_loader'] = loader

# Instantiate and load model
disease_num_classes = num_classes_list[2]  # 'disease' is final output
model = MultiViewAttentionCNN(
    image_size=image_size,
    image_depth=3,
    num_classes_list=num_classes_list,  # From datasets
    drop_prob=args.dropout_rate,
    device=device,
    num_classes_final=disease_num_classes
)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Denormalization function
def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = image.clone()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    return img

# Process attention filters into heatmap
def process_to_heatmap(attended_filters, input_img):
    attended_combined = torch.max(attended_filters.squeeze(0), 0)[0].detach().cpu().numpy()
    attended_combined = cv2.resize(attended_combined, (input_img.size(2), input_img.size(3)))
    input_img_np = denormalize(input_img).squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_img_np = np.clip(input_img_np, 0, 1)
    heatmap = cv2.addWeighted(
        input_img_np[:, :, 1].astype(np.float32), 0.97,
        attended_combined.astype(np.float32), 0.07, 0
    )
    return heatmap

# Resize images for better visualization
def resize_image(img, size=(128, 128)):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

# Visualization for first image from each of 3 dataloaders
selected_keys = ['train_gender_loader', 'train_age_10_loader', 'train_disease_loader']
fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 3 rows (tasks), 4 cols (input + 3 heatmaps)

for row, key in enumerate(selected_keys):
    loader = dataloaders[key]
    task = key.split('_')[1]
    dataset = loader.dataset
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    
    # Get first image
    image, label = next(iter(loader))
    image = image.to(device)
    true_label = idx_to_label[label.item()]

    # Get attention filters
    with torch.no_grad():
        attended_a, _, _ = model.cnn_view_a(image)
        attended_b, _, _ = model.cnn_view_b(image)
        attended_c, _, _ = model.cnn_view_c(image)
        output = model(image, image, image)  # Same image for all views as per training
    predicted_idx = torch.argmax(output).item()
    predicted_label = idx_to_label[predicted_idx]  # Using disease mapping for final output

    # Generate heatmaps
    heatmap_a = process_to_heatmap(attended_a, image)
    heatmap_b = process_to_heatmap(attended_b, image)
    heatmap_c = process_to_heatmap(attended_c, image)
    input_img = denormalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_img = np.clip(input_img, 0, 1)

    # Resize for display
    input_img_resized = resize_image(input_img)
    heatmap_a_resized = resize_image(heatmap_a)
    heatmap_b_resized = resize_image(heatmap_b)
    heatmap_c_resized = resize_image(heatmap_c)

    # Plot
    axes[row, 0].imshow(input_img_resized)
    axes[row, 0].set_title(f"Input (Task: {task})")
    axes[row, 0].axis('off')
    axes[row, 1].imshow(heatmap_a_resized)
    axes[row, 1].set_title("View A Heatmap")
    axes[row, 1].axis('off')
    axes[row, 2].imshow(heatmap_b_resized)
    axes[row, 2].set_title("View B Heatmap")
    axes[row, 2].axis('off')
    axes[row, 3].imshow(heatmap_c_resized)
    axes[row, 3].set_title("View C Heatmap")
    axes[row, 3].axis('off')
    axes[row, 0].text(0.5, -0.1, f"True: {true_label}, Pred: {predicted_label}", 
                      transform=axes[row, 0].transAxes, ha='center')

plt.tight_layout()
plt.show()