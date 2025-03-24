# multi_view_train_script.py

'''Script to train the MultiViewAttentionCNN model in stages.'''

from tqdm import tqdm  # Import tqdm for progress bar visualization
import torch  # Import PyTorch for tensor operations and neural networks
from torch.utils.data import DataLoader  # Import DataLoader for batching and shuffling data
from torch.optim import Adam  # Import Adam optimizer for gradient descent
from torchsummary import summary  # Import summary to display model architecture
from torchvision import transforms  # Import transforms for image preprocessing

from runtime_args import args  # Import runtime arguments (e.g., batch_size, learning_rate)
from attention_cnn import AttentionCNN, MultiViewAttentionCNN  # Import the custom model classes
from load_dataset import LoadDataset  # Import custom dataset loader class

# Set the device to GPU if available and specified in args, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Initialize the training dataset (used for all views for now)
train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
                            transform=transforms.ToTensor())
# Initialize the testing dataset (used for all views for now)
test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
                           transform=transforms.ToTensor())

# Create DataLoaders for training and testing (shared across views A, B, C for now)
train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Instantiate the MultiViewAttentionCNN model
model = MultiViewAttentionCNN(image_size=args.img_size, image_depth=args.img_depth, num_classes=args.num_classes, 
                              drop_prob=args.dropout_rate, device=device)
model = model.to(device)

# Define the loss function as CrossEntropyLoss for classification tasks
criterion = torch.nn.CrossEntropyLoss()

# Print a summary of the model architecture (using view A as an example input)
summary(model.cnn_view_a, (args.img_depth, args.img_size, args.img_size))

def train_single_view(submodel, dataloader, optimizer, criterion, device, num_epochs, is_train=True):
    '''Train or evaluate a single view submodule.'''
    mode = "Training" if is_train else "Testing"
    submodel.train() if is_train else submodel.eval()
    
    epoch_loss = []
    epoch_accuracy = []
    for i, sample in tqdm(enumerate(dataloader)):
        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)
        
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            _, _, net_output = submodel(batch_x)
            total_loss = criterion(net_output, batch_y)
            if is_train:
                total_loss.backward()
                optimizer.step()
        
        batch_acc = submodel.calculate_accuracy(net_output, batch_y)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)
    
    avg_loss = sum(epoch_loss) / (i + 1)
    avg_acc = sum(epoch_accuracy) / (i + 1)
    return avg_loss, avg_acc

def train_multi_view(model, dataloader, optimizer, criterion, device, num_epochs, is_train=True):
    '''Train or evaluate the full multi-view model.'''
    mode = "Training" if is_train else "Testing"
    model.train() if is_train else model.eval()
    
    epoch_loss = []
    epoch_accuracy = []
    for i, sample in tqdm(enumerate(dataloader)):
        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)
        # Duplicate the same batch for views B and C (for now)
        view_a, view_b, view_c = batch_x, batch_x, batch_x
        
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            net_output = model(view_a, view_b, view_c)
            total_loss = criterion(net_output, batch_y)
            if is_train:
                total_loss.backward()
                optimizer.step()
        
        batch_acc = model.calculate_accuracy(net_output, batch_y)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)
    
    avg_loss = sum(epoch_loss) / (i + 1)
    avg_acc = sum(epoch_accuracy) / (i + 1)
    return avg_loss, avg_acc

# Training stages
num_epochs_per_stage = args.epoch // 4  # Divide total epochs across 4 stages
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
    
    # Save the model if testing accuracy improves
    if test_acc > best_accuracy:
        torch.save(model.state_dict(), args.model_save_path.rstrip('/') + '/multi_view_attention_cnn.pth')
        best_accuracy = test_acc
        print("Model is saved!")
    print('------------------------------------------------------------------------------')

print("Training complete!")