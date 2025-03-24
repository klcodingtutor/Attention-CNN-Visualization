'''Script to train the Attention-CNN model.'''

from tqdm import tqdm  # Import tqdm for progress bar visualization
import torch  # Import PyTorch for tensor operations and neural networks
from torch.utils.data import DataLoader  # Import DataLoader for batching and shuffling data
from torch.optim import Adam  # Import Adam optimizer for gradient descent
from torchsummary import summary  # Import summary to display model architecture
from torchvision import transforms  # Import transforms for image preprocessing

from runtime_args import args  # Import runtime arguments (e.g., batch_size, learning_rate)
from attention_cnn import AttentionCNN  # Import the custom AttentionCNN model class
from load_dataset import LoadDataset  # Import custom dataset loader class

# Set the device to GPU if available and specified in args, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

# Initialize the training dataset with specified folder path, image size, depth, and transformation
train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
                            transform=transforms.ToTensor())
# Initialize the testing dataset (train=False) with similar parameters
test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
                           transform=transforms.ToTensor())

# Create a DataLoader for training data with batching, shuffling, and multi-worker support
train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
# Create a DataLoader for testing data with similar settings
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Instantiate the AttentionCNN model with specified parameters (image size, depth, classes, dropout rate, device)
model = AttentionCNN(image_size=args.img_size, image_depth=args.img_depth, num_classes=args.num_classes, drop_prob=args.dropout_rate, device=device)
# Set up the Adam optimizer with the model's parameters and learning rate from args
optimizer = Adam(model.parameters(), lr=args.learning_rate)
# Define the loss function as CrossEntropyLoss for classification tasks
criterion = torch.nn.CrossEntropyLoss()

# Move the model to the specified device (CPU or GPU)
model = model.to(device)
# Print a summary of the model architecture based on input shape (depth, height, width)
summary(model, (args.img_depth, args.img_size, args.img_size))

# Initialize variable to track the best testing accuracy for model saving
best_accuracy = 0
# Loop over the number of epochs specified in args
for epoch_idx in range(args.epoch):

    # Set the model to training mode (enables dropout, batch norm, etc.)
    model.train()

    # Lists to store loss and accuracy for each batch in the epoch
    epoch_loss = []
    epoch_accuracy = []
    i = 0  # Counter for batches
    # Iterate over batches in the training DataLoader with a progress bar
    for i, sample in tqdm(enumerate(train_generator)):

        # Extract images and labels from the sample and move them to the device
        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        # Clear gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: get model predictions (ignoring intermediate outputs with _)
        _, _, net_output = model(batch_x)

        # Compute the loss between predictions and true labels
        total_loss = criterion(input=net_output, target=batch_y)

        # Backward pass: compute gradients
        total_loss.backward()
        # Update model parameters using the optimizer
        optimizer.step()
        # Calculate accuracy for the current batch
        batch_acc = model.calculate_accuracy(predicted=net_output, target=batch_y)
        # Store the loss and accuracy for this batch
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)

    # Compute average accuracy and loss for the epoch
    curr_accuracy = sum(epoch_accuracy) / (i + 1)
    curr_loss = sum(epoch_loss) / (i + 1)

    # Print epoch number and training metrics
    print(f"Epoch {epoch_idx}")
    print(f"Training Loss : {curr_loss}, Training accuracy : {curr_accuracy}")

    # Set the model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()

    # Reset lists for testing phase
    epoch_loss = []
    epoch_accuracy = []
    i = 0  # Reset batch counter
    # Iterate over batches in the testing DataLoader with a progress bar
    for i, sample in tqdm(enumerate(test_generator)):

        # Extract images and labels from the sample and move them to the device
        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        # Forward pass: get model predictions (no gradients needed in eval mode)
        _, _, net_output = model(batch_x)

        # Compute the loss between predictions and true labels
        total_loss = criterion(input=net_output, target=batch_y)

        # Calculate accuracy for the current batch
        batch_acc = model.calculate_accuracy(predicted=net_output, target=batch_y)
        # Store the loss and accuracy for this batch
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)

    # Compute average accuracy and loss for the testing phase
    curr_accuracy = sum(epoch_accuracy) / (i + 1)
    curr_loss = sum(epoch_loss) / (i + 1)

    # Print testing metrics
    print(f"Testing Loss : {curr_loss}, Testing accuracy : {curr_accuracy}")

    # Save the model if the current testing accuracy exceeds the previous best
    if curr_accuracy > best_accuracy:
        # Save the model state dictionary to the specified path
        torch.save(model.state_dict(), args.model_save_path.rstrip('/') + '/attention_cnn.pth')
        # Update the best accuracy
        best_accuracy = curr_accuracy
        print("Model is saved!")

    # Print a separator line between epochs
    print('------------------------------------------------------------------------------')