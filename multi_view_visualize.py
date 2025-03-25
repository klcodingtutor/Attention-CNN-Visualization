import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from attention_cnn import MultiViewAttentionCNN  # Adjust if your model class name differs
from runtime_args import args  # Assuming this handles runtime arguments as in training

# Set device based on training script logic
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Load the trained model
model = MultiViewAttentionCNN(
    image_size=args.img_size,        # 32 from training script
    image_depth=3,                   # RGB channels
    num_classes=args.num_classes,    # 10 from training script
    drop_prob=args.dropout_rate,     # 0.5 from training script
    device=device
)
model_path = os.path.join(args.model_save_path, 'multi_view_attention_cnn_cifar10.pth')
model.load_state_dict(torch.load(model_path))
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

# Function to resize images for better visualization
def resize_image(img, size=(128, 128)):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

# Define test transform consistent with training
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(
    root=args.data_folder,  # './cifar10_data' from training script
    train=False,
    download=True,
    transform=test_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,          # Process one image at a time for detailed visualization
    shuffle=False,
    num_workers=args.num_workers  # 4 from training script
)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create output folder
output_folder = './output'
os.makedirs(output_folder, exist_ok=True)

# Visualization loop
for i, (image, label) in enumerate(tqdm(test_loader, desc="Visualizing")):
    image = image.to(device)

    # Get attention filters from each view
    with torch.no_grad():
        attended_a, _, _ = model.cnn_view_a(image)
        attended_b, _, _ = model.cnn_view_b(image)
        attended_c, _, _ = model.cnn_view_c(image)

    # Get final model output (passing same image to all views as per training assumption)
    with torch.no_grad():
        output = model(image, image, image)

    # Compute predicted class
    predicted_class = class_names[torch.argmax(output).item()]

    # Generate heatmaps for each view
    heatmap_a = process_to_heatmap(attended_a, image)
    heatmap_b = process_to_heatmap(attended_b, image)
    heatmap_c = process_to_heatmap(attended_c, image)

    # Prepare input image for display
    input_img = denormalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_img = np.clip(input_img, 0, 1)

    # Resize for better visibility
    input_img_resized = resize_image(input_img)
    heatmap_a_resized = resize_image(heatmap_a)
    heatmap_b_resized = resize_image(heatmap_b)
    heatmap_c_resized = resize_image(heatmap_c)

    # Create plot
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
    fig.suptitle(f"Predicted: {predicted_class} (Label: {class_names[label.item()]})")
    plt.savefig(os.path.join(output_folder, f"test_image_{i}.png"))
    plt.close()