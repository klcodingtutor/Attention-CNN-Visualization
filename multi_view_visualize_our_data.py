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
import cv2

# Set device based on training script logic
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

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

# After creating dataloaders
num_classes_list = []
for task in tasks:
    train_loader = dataloaders[f'train_{task}_loader']
    num_classes = len(train_loader.dataset.label_to_idx)
    num_classes_list.append(num_classes)

print(f"Number of classes for each task: {num_classes_list}")
disease_num_classes = len(dataloaders['train_disease_loader'].dataset.label_to_idx)
print(f"Number of disease classes: {disease_num_classes}")

# Instantiate the model with consistent parameters
model = MultiViewAttentionCNN(
    image_size=args.img_size,
    image_depth=3,
    num_classes_list=num_classes_list,
    drop_prob=args.dropout_rate,
    device=device,
    num_classes_final=disease_num_classes  # Match training
)

# Load the state dictionary
model.load_state_dict(torch.load(args.model_save_path.rstrip('/') + '/multi_view_attention_cnn_face_tasks.pth'))
model.to(device)
model.eval()

# Define denormalization function with training-specific mean and std
def denormalize(image, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    img = image.clone()
    for i in range(3):
        img[:, i, :, :] = img[:, i, :, :] * std[i] + mean[i]
    return img


# Function to process attention filters into a heatmap
def process_to_heatmap(attended_filters, input_img):
    # Combine attention filters by taking max across channels
    attended_combined = torch.max(attended_filters.squeeze(0), 0)[0].detach().cpu().numpy()
    # Resize to match input image dimensions (32x32)
    attended_combined = cv2.resize(attended_combined, (input_img.size(3), input_img.size(2)))
    # Denormalize input image for overlay
    input_img_np = denormalize(input_img).squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_img_np = np.clip(input_img_np, 0, 1)
    # Overlay attention on green channel (simplified approach)
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

# specifically extract idx_to_label for disease
idx_to_label_disease = {v: k for k, v in dataloaders['train_disease_loader'].dataset.label_to_idx.items()}
for row, key in enumerate(selected_keys):
    loader = dataloaders[key]
    task = key.split('_')[1]
    dataset = loader.dataset
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    


    

    # Get the batch
    images, labels = next(iter(loader))

    # Select the first image and label
    image = images[0].unsqueeze(0)  # Add batch dimension for model
    label = labels[0]               # Scalar tensor

    # Get the true label
    true_label = idx_to_label[label.item()]

    image = image.to(device)


    # Get attention filters
    with torch.no_grad():
        attended_a, _, _ = model.cnn_view_a(image)
        attended_b, _, _ = model.cnn_view_b(image)
        attended_c, _, _ = model.cnn_view_c(image)
        output = model(image, image, image)
    predicted_idx = torch.argmax(output).item()
    print(f"Predicted index: {predicted_idx}")
    print(f"idx_to_label: {idx_to_label}")
    predicted_label = idx_to_label_disease[predicted_idx]
    print(f"Predicted label: {predicted_label}")
    print(f"True label: {true_label}")

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
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(input_img_resized)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    axes[1].imshow(heatmap_a_resized)
    axes[1].set_title("View A Attention Heatmap")
    axes[1].axis('off')
    axes[2].imshow(heatmap_b_resized)
    axes[2].set_title("View B Attention Heatmap")
    axes[2].axis('off')
    axes[3].imshow(heatmap_c_resized)
    axes[3].set_title("View C Attention Heatmap")
    axes[3].axis('off')
    # fig.suptitle(f"Predicted: {predicted_class} (Label: {class_names[label.item()]})")
    # fig.suptitle(f"Predicted: {predicted_label} (True: {true_label})")
    plt.savefig(os.path.join("output", f"test_image_1.png"))
    plt.close()
