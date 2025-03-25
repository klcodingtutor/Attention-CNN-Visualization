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

# Step 1: Load the CSV and split into train and test sets
df = pd.read_csv('./data/face_images_path_with_meta_jpg_exist_only.csv')
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# Step 2: Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
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

# Select three dataloaders for visualization (gender, age_10, disease)
selected_keys = ['train_gender_loader', 'train_age_10_loader', 'train_disease_loader']
selected_loaders = [dataloaders[key] for key in selected_keys]
tasks_display = [key.split('_')[1] for key in selected_keys]  # ['gender', 'age_10', 'disease']

# Normalization parameters (same as used in transforms)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (loader, task) in enumerate(zip(selected_loaders, tasks_display)):
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

# Main Training Script Starts Here
'''Script to train the MultiViewAttentionCNN model in stages on the face image dataset.'''

print("------------------------------------------------------------")

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Determine the number of classes for each task
num_classes_list = []
for task in tasks:  # Use gender, age_10, disease for views
    train_loader = dataloaders[f'train_{task}_loader']
    num_classes = len(train_loader.dataset.label_to_idx)
    num_classes_list.append(num_classes)
# For the final fusion output, use the 'disease' task
disease_num_classes = len(dataloaders['train_disease_loader'].dataset.label_to_idx)

print(f"Number of classes per view: {num_classes_list}")
print(f"Number of classes for final output (disease): {disease_num_classes}")

# Instantiate the MultiViewAttentionCNN model
model = MultiViewAttentionCNN(
    image_size=32,  # Adjusted for face images
    image_depth=3,
    num_classes_list=num_classes_list,  # Classes for gender, age_10, disease
    drop_prob=args.dropout_rate,
    device=device,
    num_classes_final=disease_num_classes  # Final output for disease
)
model = model.to(device)

# Define the loss function (multiple outputs require multiple loss functions)
criterion_views = torch.nn.CrossEntropyLoss()  # For individual views
criterion_final = torch.nn.CrossEntropyLoss()  # For fused output

# Print model summary (for one view as example)
summary(model.cnn_view_a, (3, 32, 32))

# Print model summary (for entire model)
summary(model, [(3, 32, 32), (3, 32, 32), (3, 32, 32)])

# Modified training functions to handle multiple tasks
def train_single_view(submodel, dataloader, optimizer, criterion, device, num_epochs, task_idx, is_train=True):
    mode = "Training" if is_train else "Testing"
    submodel.train() if is_train else submodel.eval()
    
    epoch_loss = []
    epoch_accuracy = []
    for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, labels = images.to(device), labels.to(device)
        
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            # Get individual view output
            _, _, net_output = submodel(images)
            total_loss = criterion(net_output, labels)
            if is_train:
                total_loss.backward()
                optimizer.step()
        
        batch_acc = submodel.calculate_accuracy(net_output, labels)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)
    
    avg_loss = sum(epoch_loss) / len(dataloader)
    avg_acc = sum(epoch_accuracy) / len(dataloader)
    return avg_loss, avg_acc

def train_multi_view(model, dataloader, optimizer, criterion, device, num_epochs, is_train=True):
    mode = "Training" if is_train else "Testing"
    model.train() if is_train else model.eval()
    
    epoch_loss = []
    epoch_accuracy = []
    for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, labels = images.to(device), labels.to(device)
        view_a, view_b, view_c = images, images, images  # Same image for all views
        
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            net_output = model(view_a, view_b, view_c, return_individual_outputs=False)
            total_loss = criterion(net_output, labels)
            if is_train:
                total_loss.backward()
                optimizer.step()
        
        batch_acc = model.calculate_accuracy(net_output, labels)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)
    
    avg_loss = sum(epoch_loss) / len(dataloader)
    avg_acc = sum(epoch_accuracy) / len(dataloader)
    return avg_loss, avg_acc

# Training stages
num_epochs_per_stage = args.epoch // 4
best_accuracy = 0

# Stage 1: Train View A on Gender (Freeze B and C)
print("Stage 1: Training View A on Gender (Freeze B and C)")
optimizer_a = Adam(model.cnn_view_a.parameters(), lr=args.learning_rate)
for param in model.cnn_view_b.parameters():
    param.requires_grad = False
for param in model.cnn_view_c.parameters():
    param.requires_grad = False
for param in model.fusion_layers.parameters():
    param.requires_grad = False

for epoch_idx in range(num_epochs_per_stage):
    train_loader = dataloaders['train_gender_loader']
    test_loader = dataloaders['test_gender_loader']
    train_loss, train_acc = train_single_view(model.cnn_view_a, train_loader, optimizer_a, criterion_views, device, 1, 0)
    test_loss, test_acc = train_single_view(model.cnn_view_a, test_loader, None, criterion_views, device, 1, 0, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    print('------------------------------------------------------------------------------')

# Stage 2: Train View B on Age_10 (Freeze A and C)
print("Stage 2: Training View B on Age_10 (Freeze A and C)")
optimizer_b = Adam(model.cnn_view_b.parameters(), lr=args.learning_rate)
for param in model.cnn_view_a.parameters():
    param.requires_grad = False
for param in model.cnn_view_b.parameters():
    param.requires_grad = True
for param in model.cnn_view_c.parameters():
    param.requires_grad = False
for param in model.fusion_layers.parameters():
    param.requires_grad = False

for epoch_idx in range(num_epochs_per_stage):
    train_loader = dataloaders['train_age_10_loader']
    test_loader = dataloaders['test_age_10_loader']
    train_loss, train_acc = train_single_view(model.cnn_view_b, train_loader, optimizer_b, criterion_views, device, 1, 1)
    test_loss, test_acc = train_single_view(model.cnn_view_b, test_loader, None, criterion_views, device, 1, 1, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    print('------------------------------------------------------------------------------')

# Stage 3: Train View C on Disease (Freeze A and B)
print("Stage 3: Training View C on Disease (Freeze A and B)")
optimizer_c = Adam(model.cnn_view_c.parameters(), lr=args.learning_rate)
for param in model.cnn_view_a.parameters():
    param.requires_grad = False
for param in model.cnn_view_b.parameters():
    param.requires_grad = False
for param in model.cnn_view_c.parameters():
    param.requires_grad = True
for param in model.fusion_layers.parameters():
    param.requires_grad = False

for epoch_idx in range(num_epochs_per_stage):
    train_loader = dataloaders['train_disease_loader']
    test_loader = dataloaders['test_disease_loader']
    train_loss, train_acc = train_single_view(model.cnn_view_c, train_loader, optimizer_c, criterion_views, device, 1, 2)
    test_loss, test_acc = train_single_view(model.cnn_view_c, test_loader, None, criterion_views, device, 1, 2, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    print('------------------------------------------------------------------------------')

# Stage 4: Fine-tune Fusion Layers on Disease
print("Stage 4: Fine-tuning Fusion Layers on Disease")
optimizer_fusion = Adam(model.fusion_layers.parameters(), lr=args.learning_rate)
for param in model.cnn_view_a.parameters():
    param.requires_grad = False
for param in model.cnn_view_b.parameters():
    param.requires_grad = False
for param in model.cnn_view_c.parameters():
    param.requires_grad = False
for param in model.fusion_layers.parameters():
    param.requires_grad = True

for epoch_idx in range(num_epochs_per_stage):
    train_loader = dataloaders['train_disease_loader']
    test_loader = dataloaders['test_disease_loader']
    train_loss, train_acc = train_multi_view(model, train_loader, optimizer_fusion, criterion_final, device, 1)
    test_loss, test_acc = train_multi_view(model, test_loader, None, criterion_final, device, 1, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    if test_acc > best_acc:
        torch.save(model.state_dict(), args.model_save_path.rstrip('/') + '/multi_view_attention_cnn_face_tasks.pth')
        best_accuracy = test_acc
        print("Model is saved!")
    print('------------------------------------------------------------------------------')

print("Training complete!")