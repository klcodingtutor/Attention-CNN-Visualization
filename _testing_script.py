import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 1: Load the CSV and split into train and test sets
df = pd.read_csv('face_images_path_with_meta_jpg_exist_only.csv')
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# Step 2: Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Define a single parameterized Dataset class with internal mappings
class CustomDataset(Dataset):
    def __init__(self, dataframe, task, image_folder, transform=None):
        """
        Parameters:
            dataframe: pandas DataFrame with the data
            task: str, one of 'gender', 'age_10', 'age_5', 'disease'
            image_folder: str, path to the folder containing images
            transform: torchvision transforms to apply to images
        """
        self.dataframe = dataframe
        self.task = task
        self.transform = transform
        self.image_folder = image_folder
        
        # Validate task parameter
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

        # Verify that image folder exists
        if not os.path.exists(self.image_folder):
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Construct image path
        img_filename = self.dataframe.iloc[idx]['dest_filename']
        img_path = os.path.join(self.image_folder, img_filename)
        
        # Debug: Print the path being attempted
        # print(f"Attempting to load: {img_path}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Get label using the internal mapping
        label = self.label_to_idx[self.dataframe.iloc[idx][self.label_col]]

        return image, label

# Step 4: Create Datasets and DataLoaders using the parameterized class
tasks = ['gender', 'age_10', 'age_5', 'disease']
image_folder = './data'

# Ensure the image folder exists
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Specified image folder does not exist: {image_folder}")

dataloaders = {}

for task in tasks:
    # Create train and test datasets
    train_dataset = CustomDataset(train_df, task=task, image_folder=image_folder, transform=train_transform)
    test_dataset = CustomDataset(test_df, task=task, image_folder=image_folder, transform=test_transform)
    
    # Create train and test DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Store in dictionary
    dataloaders[f'train_{task}_loader'] = train_loader
    dataloaders[f'test_{task}_loader'] = test_loader

# Step 5: Verify the DataLoaders by iterating and printing sizes
for task in tasks:
    train_key = f'train_{task}_loader'
    test_key = f'test_{task}_loader'
    print(f"{train_key} size: {len(dataloaders[train_key].dataset)}")
    print(f"{test_key} size: {len(dataloaders[test_key].dataset)}")
    train_batch = next(iter(dataloaders[train_key]))
    test_batch = next(iter(dataloaders[test_key]))
    print(f"Sample train image shape: {train_batch[0].shape}")
    print(f"Sample train label value: {train_batch[1]}")
    print(f"Sample test image shape: {test_batch[0].shape}")
    print(f"Sample test label value: {test_batch[1]}")
    print()

# Step 6: Plot the first image from three selected dataloaders
# Define denormalize function
def denormalize(image, mean, std):
    image = image.clone()
    for c in range(3):
        image[c] = image[c] * std[c] + mean[c]
    return image

# Select three dataloaders
selected_keys = ['train_gender_loader', 'train_age_10_loader', 'train_age_5_loader']
selected_loaders = [dataloaders[key] for key in selected_keys]
tasks = [key.split('_')[1] for key in selected_keys]  # ['gender', 'age_10', 'age_5']

# Normalization parameters (same as used in transforms)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (loader, task) in enumerate(zip(selected_loaders, tasks)):
    # Get dataset
    dataset = loader.dataset
    # Create idx_to_label mapping to retrieve original labels
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    # Get first batch
    batch = next(iter(loader))
    images, labels = batch
    # Get first image and label
    image = images[0]
    label_idx = labels[0].item()  # Convert tensor to scalar
    # Get original label
    original_label = idx_to_label[label_idx]
    # Denormalize image for proper visualization
    image_denorm = denormalize(image, mean, std)
    # Convert to numpy: (C, H, W) -> (H, W, C)
    image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
    # Clip to [0,1] for display
    image_np = np.clip(image_np, 0, 1)
    # Plot
    axes[i].imshow(image_np)
    axes[i].set_title(f"Task: {task}, Label: {original_label}")
    axes[i].axis('off')  # Hide axes for cleaner visualization

# Adjust layout and show plot
plt.tight_layout()
plt.show()
