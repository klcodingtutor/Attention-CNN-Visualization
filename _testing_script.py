import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
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

# Step 4: Create Datasets and DataLoaders using the parameterized class
tasks = ['gender', 'age_10', 'age_5', 'disease']
image_folder = 'C:/Users/megah/Dropbox/Prompt/self_attention_face/'


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

# Optional: Verify the DataLoaders by iterating and print out size for image and value
for task in tasks:
    train_key = f'train_{task}_loader'
    test_key = f'test_{task}_loader'
    print(f"{train_key} size: {len(dataloaders[train_key].dataset)}")
    print(f"{test_key} size: {len(dataloaders[test_key].dataset)}")
    print(f"Sample train image shape: {next(iter(dataloaders[train_key]))[0].shape}")
    print(f"Sample train label value: {next(iter(dataloaders[train_key]))[1]}")
    print(f"Sample test image shape: {next(iter(dataloaders[test_key]))[0].shape}")
    print(f"Sample test label value: {next(iter(dataloaders[test_key]))[1]}")
    print()
