'''Script to train the MultiViewAttentionCNN model in stages on CIFAR-10.'''

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from torchvision import datasets, transforms

from runtime_args import args  # Import runtime arguments
from attention_cnn import AttentionCNN, MultiViewAttentionCNN  # Import model classes

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Define transforms for CIFAR-10
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root=args.data_folder, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root=args.data_folder, train=False, download=True, transform=test_transform)

# Create DataLoaders
train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Instantiate the MultiViewAttentionCNN model
model = MultiViewAttentionCNN(image_size=args.img_size, image_depth=3, num_classes=args.num_classes, 
                              drop_prob=args.dropout_rate, device=device)
model = model.to(device)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Print model summary (for one view)
summary(model.cnn_view_a, (3, args.img_size, args.img_size))

def train_single_view(submodel, dataloader, optimizer, criterion, device, num_epochs, is_train=True):
    mode = "Training" if is_train else "Testing"
    submodel.train() if is_train else submodel.eval()
    
    epoch_loss = []
    epoch_accuracy = []
    for i, (images, labels) in tqdm(enumerate(dataloader)):
        images, labels = images.to(device), labels.to(device)
        
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            _, _, net_output = submodel(images)
            total_loss = criterion(net_output, labels)
            if is_train:
                total_loss.backward()
                optimizer.step()
        
        batch_acc = submodel.calculate_accuracy(net_output, labels)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)
    
    avg_loss = sum(epoch_loss) / (i + 1)
    avg_acc = sum(epoch_accuracy) / (i + 1)
    return avg_loss, avg_acc

def train_multi_view(model, dataloader, optimizer, criterion, device, num_epochs, is_train=True):
    mode = "Training" if is_train else "Testing"
    model.train() if is_train else model.eval()
    
    epoch_loss = []
    epoch_accuracy = []
    for i, (images, labels) in tqdm(enumerate(dataloader)):
        images, labels = images.to(device), labels.to(device)
        view_a, view_b, view_c = images, images, images  # Use same images for all views
        
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            net_output = model(view_a, view_b, view_c)
            total_loss = criterion(net_output, labels)
            if is_train:
                total_loss.backward()
                optimizer.step()
        
        batch_acc = model.calculate_accuracy(net_output, labels)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)
    
    avg_loss = sum(epoch_loss) / (i + 1)
    avg_acc = sum(epoch_accuracy) / (i + 1)
    return avg_loss, avg_acc

# Training stages
num_epochs_per_stage = args.epoch // 4
best_accuracy = 0

# Stage 1: Train View A
print("Stage 1: Training View A")
optimizer_a = Adam(model.cnn_view_a.parameters(), lr=args.learning_rate)
for param in model.cnn_view_b.parameters():
    param.requires_grad = False
for param in model.cnn_view_c.parameters():
    param.requires_grad = False
for param in model.fusion_layers.parameters():
    param.requires_grad = False

for epoch_idx in range(num_epochs_per_stage):
    train_loss, train_acc = train_single_view(model.cnn_view_a, train_generator, optimizer_a, criterion, device, 1)
    test_loss, test_acc = train_single_view(model.cnn_view_a, test_generator, None, criterion, device, 1, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    print('------------------------------------------------------------------------------')

# Stage 2: Train View B
print("Stage 2: Training View B")
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
    train_loss, train_acc = train_single_view(model.cnn_view_b, train_generator, optimizer_b, criterion, device, 1)
    test_loss, test_acc = train_single_view(model.cnn_view_b, test_generator, None, criterion, device, 1, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    print('------------------------------------------------------------------------------')

# Stage 3: Train View C (A and B frozen)
print("Stage 3: Training View C (A and B frozen)")
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
    train_loss, train_acc = train_single_view(model.cnn_view_c, train_generator, optimizer_c, criterion, device, 1)
    test_loss, test_acc = train_single_view(model.cnn_view_c, test_generator, None, criterion, device, 1, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    print('------------------------------------------------------------------------------')

# Stage 4: Fine-tune Fusion Layers
print("Stage 4: Fine-tuning Fusion Layers")
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
    train_loss, train_acc = train_multi_view(model, train_generator, optimizer_fusion, criterion, device, 1)
    test_loss, test_acc = train_multi_view(model, test_generator, None, criterion, device, 1, is_train=False)
    print(f"Epoch {epoch_idx+1}/{num_epochs_per_stage}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")
    
    if test_acc > best_accuracy:
        torch.save(model.state_dict(), args.model_save_path.rstrip('/') + '/multi_view_attention_cnn_cifar10.pth')
        best_accuracy = test_acc
        print("Model is saved!")
    print('------------------------------------------------------------------------------')

print("Training complete!")