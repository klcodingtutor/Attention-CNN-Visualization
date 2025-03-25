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
import pandas as pd

# Define CustomDataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, task, image_folder, transform=None):
        self.dataframe = dataframe
        self.task = task
        self.transform = transform
        self.image_folder = image_folder
        
        valid_tasks = ['gender', 'age_10', 'disease']
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
        img_filename = self.dataframe.iloc[idx]['dest_filename']
        img_path = os.path.join(self.image_folder, img_filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_to_idx[self.dataframe.iloc[idx][self.label_col]]
        return image, label

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Load trained model
tasks = ['gender', 'age_10', 'disease']
num_classes_list = []
model_path = os.path.join(args.model_save_path, 'multi_view_attention_cnn_face_tasks.pth')
image_size = 32

# Define test transform
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CSV and create dataloaders
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
    num_classes_list=num_classes_list,
    drop_prob=args.dropout_rate,
    device=device,
    num_classes_final=disease_num_classes
)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Fixed denormalization function
def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor image with shape (B, C, H, W) or (C, H, W).
    Returns the denormalized image with the same shape.
    """
    img = image.clone()
    # Ensure image has batch dimension
    if img.dim() == 3:  # (C, H, W)
        img = img.unsqueeze(0)  # Add batch dim -> (1, C, H, W)
    elif img.dim() != 4:
        raise ValueError(f"Expected image tensor with 3 or 4 dims, got {img.dim()} dims with shape {img.shape}")
    
    # Check number of channels
    channels = img.size(1)
    if channels != 3:
        raise ValueError(f"Expected 3 channels, got {channels}")
    
    # Denormalize each channel
    for c in range(channels):
        img[:, c, :, :] = img[:, c, :, :] * std[c] + mean[c]
    return img.squeeze(0) if image.dim() == 3 else img  # Remove batch dim if input was (C, H, W)

# Process attention filters into heatmap
def process_to_heatmap(attended_filters, input_img):
    attended_combined = torch.max(attended_filters.squeeze(0), 0)[0].detach().cpu().numpy()
    attended_combined = cv2.resize(attended_combined, (input_img.size(2), input_img.size(3)))
    input_img_np = denormalize(input_img).squeeze(0).permute(1, 2, 0).cpu().numpy()  # Remove batch dim here
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
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for row, key in enumerate(selected_keys):
    loader = dataloaders[key]
    task = key.split('_')[1]
    dataset = loader.dataset
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    
    # Get first image
    image, label = next(iter(loader))
    print(f"Image shape from {key}: {image.shape}")  # Debug print
    image = image.to(device)
    true_label = idx_to_label[label.item()]

    # Get attention filters
    with torch.no_grad():
        attended_a, _, _ = model.cnn_view_a(image)
        attended_b, _, _ = model.cnn_view_b(image)
        attended_c, _, _ = model.cnn_view_c(image)
        output = model(image, image, image)
    predicted_idx = torch.argmax(output).item()
    predicted_label = idx_to_label[predicted_idx]  # Using disease mapping for final output

    # Generate heatmaps
    heatmap_a = process_to_heatmap(attended_a, image)
    heatmap_b = process_to_heatmap(attended_b, image)
    heatmap_c = process_to_heatmap(attended_c, image)
    # save the heatmaps
    cv2.imwrite(os.path.join("output", f'heatmap_{task}_a.png'), heatmap_a)
    cv2.imwrite(os.path.join("output", f'heatmap_{task}_b.png'), heatmap_b)
    cv2.imwrite(os.path.join("output", f'heatmap_{task}_c.png'), heatmap_c)
    
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
# save the plot
plt.savefig(os.path.join("output", 'multi_view_attention_cnn_heatmaps.png'))