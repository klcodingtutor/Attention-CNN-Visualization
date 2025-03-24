'''
Visualize the trained model's feature maps.
'''

import os  # Import os for file and directory operations
from tqdm import tqdm  # Import tqdm for progress bar visualization
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
import torch  # Import PyTorch for tensor operations
from torch.utils.data import DataLoader  # Import DataLoader for batching data
from torchvision import transforms  # Import transforms for image preprocessing
import matplotlib.pyplot as plt  # Import matplotlib for plotting

from load_dataset import LoadInputImages  # Import custom class for loading input images
from attention_cnn import AttentionCNN  # Import the AttentionCNN model class
from runtime_args import args  # Import runtime arguments (e.g., img_size, data_folder)

# Set the device to GPU if available and specified in args, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
# Instantiate the AttentionCNN model with specified parameters
model = AttentionCNN(image_size=args.img_size, image_depth=args.img_depth, num_classes=args.num_classes, drop_prob=args.dropout_rate, device=device)

# Move the model to the specified device (CPU or GPU)
model = model.to(device)

# Check if the trained model file exists, raise an assertion error if it doesn’t
assert os.path.exists(args.model_save_path.rstrip('/')+'/attention_cnn.pth'), 'A trained model does not exist!'

try:
    # Load the trained model's state dictionary from the saved file
    model.load_state_dict(torch.load(args.model_save_path.rstrip('/')+'/attention_cnn.pth'))
    print("Model loaded!")
except Exception as e:
    # Print any error that occurs during model loading
    print(e)

# Set the model to evaluation mode (disables dropout and batch norm updates)
model.eval()

# Load input images from the specified folder with preprocessing (e.g., to tensor)
input_data = LoadInputImages(input_folder=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, transform=transforms.ToTensor())
# Create a DataLoader for the input images with batch size 1, no shuffling, and 1 worker
data_generator = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=1)

# Define class names for interpreting model predictions
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Define the output folder for saving visualizations
output_folder = './output'

# Create the output folder if it doesn’t exist (Note: corrected 'makedir' to 'makedirs')
if not os.path.exists(output_folder): os.makedirs(output_folder)

# Create a figure for plotting with a specified size
fig = plt.figure(figsize=(20, 5))

# Iterate over each image in the DataLoader with a progress bar
for i, image in tqdm(enumerate(data_generator)):

    # Clear the current plot to prepare for the new image
    plt.clf()

    # Move the input image tensor to the specified device
    image = image.to(device)

    # Forward pass: get attention filters, CNN filters, and model output
    attention_filters, cnn_filters, output = model(image)

    # Apply softmax to the output to get class probabilities
    softmaxed_output = torch.nn.Softmax(dim=1)(output)
    # Get the predicted class index and map it to the corresponding class name
    predicted_class = class_names[torch.argmax(softmaxed_output).cpu().numpy()]

    # Combine attention filters by taking the maximum across the channel dimension and resizing to original image size
    attention_combined_filter = cv2.resize(torch.max(attention_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))
    # Combine CNN filters similarly by taking the maximum across channels and resizing
    cnn_combined_filter = cv2.resize(torch.max(cnn_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))

    # Convert the input image tensor to NumPy format (HWC) and resize it
    input_img = cv2.resize(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), (args.img_size, args.img_size))

    # Create heatmaps by overlaying the combined filters on the green channel of the input image
    heatmap_att = cv2.addWeighted(np.asarray(input_img[:,:, 1], dtype=np.float32), 0.97,  # Green channel of input
                                  np.asarray(attention_combined_filter, dtype=np.float32), 0.07, 0)  # Attention filter
    heatmap_cnn = cv2.addWeighted(np.asarray(input_img[:,:, 1], dtype=np.float32), 0.97,  # Green channel of input
                                  np.asarray(cnn_combined_filter, dtype=np.float32), 0.07, 0)  # CNN filter

    # Add subplot for the input image
    fig.add_subplot(151)
    plt.imshow(input_img)  # Display the original input image
    plt.title("Input Image")  # Set title
    plt.xticks(())  # Hide x-axis ticks
    plt.yticks(())  # Hide y-axis ticks

    # Add subplot for the attention feature map
    fig.add_subplot(152)
    plt.imshow(attention_combined_filter)  # Display the combined attention filter
    plt.title("Attention Feature Map")  # Set title
    plt.xticks(())  # Hide x-axis ticks
    plt.yticks(())  # Hide y-axis ticks

    # Add subplot for the CNN feature map
    fig.add_subplot(153)
    plt.imshow(cnn_combined_filter)  # Display the combined CNN filter
    plt.title("CNN Feature Map")  # Set title
    plt.xticks(())  # Hide x-axis ticks
    plt.yticks(())  # Hide y-axis ticks

    # Add subplot for the attention heatmap
    fig.add_subplot(154)
    plt.imshow(heatmap_att)  # Display the attention heatmap
    plt.title("Attention Heat Map")  # Set title
    plt.xticks(())  # Hide x-axis ticks
    plt.yticks(())  # Hide y-axis ticks

    # Add subplot for the CNN heatmap
    fig.add_subplot(155)
    plt.imshow(heatmap_cnn)  # Display the CNN heatmap
    plt.title("CNN Heat Map")  # Set title
    plt.xticks(())  # Hide x-axis ticks
    plt.yticks(())  # Hide y-axis ticks

    # Add a super title with the predicted class
    fig.suptitle(f"Network's prediction : {predicted_class.capitalize()}", fontsize=20)

    # Save the figure as a PNG file in the output folder
    plt.savefig(f'{output_folder}/{i}.png')