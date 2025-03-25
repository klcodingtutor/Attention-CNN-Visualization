# download_cifar10.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def download_and_validate_cifar10(data_root="./cifar10_data", batch_size=4, num_workers=2):
    """Download CIFAR-10 dataset and validate its structure and contents."""
    
    # Ensure the root directory exists
    os.makedirs(data_root, exist_ok=True)
    print(f"Using data root directory: {data_root}")

    # Define basic transform (no augmentation, just tensor conversion for validation)
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    # Download CIFAR-10 training set
    print("Downloading CIFAR-10 training set...")
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    
    # Download CIFAR-10 test set
    print("Downloading CIFAR-10 test set...")
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # Create DataLoaders for validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Validate dataset structure
    print("\nValidating dataset structure...")
    print(f"Training set size: {len(train_dataset)} samples")
    print(f"Test set size: {len(test_dataset)} samples")
    print(f"Data directory contents: {os.listdir(data_root)}")

    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Validate a few samples from the training set
    print("\nValidating training set samples...")
    train_iter = iter(train_loader)
    images, labels = next(train_iter)
    print(f"Batch shape: {images.shape} (expected: [batch_size, 3, 32, 32])")
    print(f"Label shape: {labels.shape} (expected: [batch_size])")
    print("Sample labels and classes:")
    for i in range(min(batch_size, len(labels))):
        print(f"Sample {i}: Label={labels[i].item()}, Class={class_names[labels[i].item()]}")
    print(f"Image min/max values: {images.min().item():.4f}/{images.max().item():.4f} (expected: 0.0/1.0)")

    # Validate a few samples from the test set
    print("\nValidating test set samples...")
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    print(f"Batch shape: {images.shape} (expected: [batch_size, 3, 32, 32])")
    print(f"Label shape: {labels.shape} (expected: [batch_size])")
    print("Sample labels and classes:")
    for i in range(min(batch_size, len(labels))):
        print(f"Sample {i}: Label={labels[i].item()}, Class={class_names[labels[i].item()]}")
    print(f"Image min/max values: {images.min().item():.4f}/{images.max().item():.4f} (expected: 0.0/1.0)")

    print("\nCIFAR-10 dataset download and validation completed successfully!")

if __name__ == "__main__":
    # Parameters (adjust as needed)
    data_root = "./cifar10_data"
    batch_size = 4
    num_workers = 2

    try:
        download_and_validate_cifar10(data_root, batch_size, num_workers)
    except Exception as e:
        print(f"Error occurred: {str(e)}")