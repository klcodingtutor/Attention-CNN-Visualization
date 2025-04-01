# Suitable filename: generate_dataloaders.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torch.optim import Adam
from torchsummary import summary
from runtime_args import args  # Import runtime arguments
from attention_cnn import AttentionCNN, MultiViewAttentionCNN  # Import model classes


def generate_dataloaders():
    # Step 1: Load the CSV and split into train and test sets
    df = pd.read_csv('./data/face_images_path_with_meta_jpg_exist_only.csv')
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    # Step 2: Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
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
    tasks = ['gender', 'age_10', 'disease']  # Primary tasks, keeping 'age_5' as an option
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
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Store in dictionary
        dataloaders[f'train_{task}_loader'] = train_loader
        dataloaders[f'test_{task}_loader'] = test_loader

    return dataloaders, df, train_df, test_df

def test_plot(dataloader):
    """
    Function to test the dataloader by plotting a few images and their labels.
    """
    # Get a batch of data
    images, labels = next(iter(dataloader))

    # Convert to numpy for plotting
    images = images.numpy().transpose((0, 2, 3, 1))  # Change from (N, C, H, W) to (N, H, W, C)
    labels = labels.numpy()

    # Plot the first 5 images and their labels
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()
