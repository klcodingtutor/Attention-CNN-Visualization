import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
from attention_cnn import MultiViewAttentionCNN  # Import model class
from runtime_args import args  # Import runtime arguments

# Step 1: Load the CSV and get the test set
df = pd.read_csv('./data/face_images_path_with_meta_jpg_exist_only.csv')
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']


# Step 2: Define image transformations (same as in training script for consistency)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Reuse the CustomDataset class from your training script
class CustomDataset(Dataset):
    def __init__(self, dataframe, task, image_folder, transform=None):
        self.dataframe = dataframe
        self.task = task
        self.transform = transform
        self.image_folder = image_folder
        
        valid_tasks = ['gender', 'age_10', 'age_5', 'disease']
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of {valid_tasks}, got {task}")

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
    # train_dataset = CustomDataset(train_df, task=task, image_folder=image_folder, transform=train_transform)
    test_dataset = CustomDataset(test_df, task=task, image_folder=image_folder, transform=test_transform)
    
    # Create train and test DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Store in dictionary
    # dataloaders[f'train_{task}_loader'] = train_loader
    dataloaders[f'test_{task}_loader'] = test_loader

# Step 5: Verify the DataLoaders by iterating and printing sizes
for task in tasks:
    # train_key = f'train_{task}_loader'
    test_key = f'test_{task}_loader'
    # print(f"{train_key} size: {len(dataloaders[train_key].dataset)}")
    print(f"{test_key} size: {len(dataloaders[test_key].dataset)}")
    # train_batch = next(iter(dataloaders[train_key]))
    test_batch = next(iter(dataloaders[test_key]))
    # print(f"Sample train image shape: {train_batch[0].shape}")
    # print(f"Sample train label value: {train_batch[1]}")
    print(f"Sample test image shape: {test_batch[0].shape}")
    print(f"Sample test label value: {test_batch[1]}")
    print()
    
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Determine the number of classes for each task
num_classes_list = []
for task in tasks:  # Use gender, age_10, disease for views
    test_loader = dataloaders[f'test_{task}_loader']
    num_classes = len(test_loader.dataset.label_to_idx)
    num_classes_list.append(num_classes)
# For the final fusion output, use the 'disease' task
disease_num_classes = len(dataloaders['test_disease_loader'].dataset.label_to_idx)

print(f"Number of classes per view: {num_classes_list}")
print(f"Number of classes for final output (disease): {disease_num_classes}")

# Instantiate the MultiViewAttentionCNN model
model = MultiViewAttentionCNN(
    image_size=64,
    image_depth=3,
    num_classes_list=num_classes_list,
    drop_prob=args.dropout_rate,
    device=device,
    num_classes_final=disease_num_classes
)
model.load_state_dict(torch.load(args.model_save_path.rstrip('/') + '/multi_view_attention_cnn_face_tasks.pth'))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Step 6: Testing function for Gender (View A)
def test_gender_view(model, dataloader, device, output_dir='gender_test_outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_predictions = []
    all_true_labels = []
    all_attention_maps = []
    all_filenames = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            filenames = dataloader.dataset.dataframe.iloc[i * dataloader.batch_size:(i + 1) * dataloader.batch_size]['dest_filename'].values
            images, labels = images.to(device), labels.to(device)
            
            # Get outputs from View A (gender) with attention maps
            view_a_output, view_a_attention, view_a_pred_output = model.cnn_view_a(images)
            
            # Predictions
            _, preds = torch.max(view_a_pred_output, 1)

            # Collect data
            all_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_attention_maps.extend(view_a_attention.cpu().numpy())
            all_filenames.extend(filenames)

    # Convert idx back to original labels
    idx_to_label = {v: k for k, v in dataloader.dataset.label_to_idx.items()}
    pred_int = [int(pred) for pred in all_predictions]
    predicted_labels = [idx_to_label[pred] for pred in all_predictions]
    true_labels = [idx_to_label[label] for label in all_true_labels]

    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels)) * 100
    print(f"Gender Test Accuracy: {accuracy:.2f}%")

    # Save results to a DataFrame
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': true_labels,
        'predicted_label': predicted_labels
    })
    results_df.to_csv(os.path.join(output_dir, 'inferenced_gender_predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(output_dir, 'inferenced_gender_predictions.csv')}")

    # Save attention maps as numpy arrays
    attention_maps_array = np.array(all_attention_maps)
    np.save(os.path.join(output_dir, 'inferenced_gender_attention_maps.npy'), attention_maps_array)
    print(f"Attention maps saved to {os.path.join(output_dir, 'inferenced_gender_attention_maps.npy')}")

    return accuracy, results_df, attention_maps_array

# Step 7: Run the test for Gender
print("Testing Gender View (View A)")
output_dir = 'gender_test_outputs'
accuracy, results_df, attention_maps = test_gender_view(model, test_loader, device, output_dir)

print(f"Testing complete! Results saved in {output_dir}")