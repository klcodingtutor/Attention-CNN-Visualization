'''
CNN with an attention mechanism.
'''

import torch  # Import PyTorch for tensor operations
import torch.nn as nn  # Import PyTorch neural network module

class AttentionCNN(nn.Module):
    '''A CNN architecture with an attention mechanism.
    '''

    def __init__(self, image_size, image_depth, num_classes, drop_prob, device):
        '''Initialize model parameters and build the architecture.
        '''

        super(AttentionCNN, self).__init__()  # Call the parent class (nn.Module) constructor

        # Store input parameters as instance variables
        self.image_size = image_size  # Size (height/width) of the input image
        self.image_depth = image_depth  # Number of channels in the input image (e.g., 3 for RGB)
        self.num_classes = num_classes  # Number of output classes for classification
        self.drop_prob = drop_prob  # Dropout probability for regularization
        self.device = device  # Device to run the model on (CPU or GPU)

        self.build_model()  # Build the model architecture


    def init_weights(self, m):
        '''Initialize weights for convolutional and linear layers using Xavier initialization.
        '''

        # Check if the layer is a Linear or Conv2d layer
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)  # Initialize weights with Xavier uniform distribution
            m.bias.data.fill_(0.01)  # Initialize biases with a small constant value (0.01)

    def build_model(self):
        '''Define and build the architecture of the CNN model with attention mechanism.
        '''

        # Define the convolutional layers as a sequential block
        self.conv_layers = nn.Sequential(
            # First conv layer: input channels = image_depth, output channels = 128
            nn.Conv2d(in_channels=self.image_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # ReLU activation (inplace=True saves memory)
            nn.BatchNorm2d(num_features=128),  # Batch normalization for 128 channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling to reduce spatial dimensions by half
            # Second conv layer: 128 input channels, 128 output channels
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            # Third conv layer: 128 input channels, 128 output channels
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            # Fourth conv layer: 128 input channels, 70 output channels
            nn.Conv2d(in_channels=128, out_channels=70, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)  # ReLU activation
        )

        # Calculate the size of the flattened feature vector after convolutions and pooling
        # Image size is reduced by 2^3 (three MaxPool layers), multiplied by 70 output channels
        self.feature_vector_size = (self.image_size // (2**3))**2 * 70

        # Define the attention mechanism as a sequential block
        self.attention = nn.Sequential(
            nn.Linear(self.feature_vector_size, self.feature_vector_size),  # Linear layer for attention
            nn.ReLU(inplace=True)  # ReLU activation
        )
        # Initialize attention weights as a learnable parameter with small random values
        self.weight = nn.Parameter(torch.randn(1, self.feature_vector_size) * 0.05)

        # Define the fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_vector_size, 256),  # First FC layer: feature_vector_size to 256
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Dropout(p=self.drop_prob),  # Dropout for regularization
            nn.Linear(256, self.num_classes)  # Output layer: 256 to num_classes
        )

    def forward(self, x):
        '''Perform forward propagation through the network.
        '''

        # Pass input through convolutional layers
        x = self.conv_layers(x)
        # Flatten the output of conv layers into a 1D vector for each sample in the batch
        x1 = torch.flatten(x, 1)

        # Apply attention mechanism: compute attention scores and weigh the features
        attention_out = nn.Sigmoid()(self.weight * self.attention(x1))  # Sigmoid to normalize attention weights
        x1 = attention_out * x1  # Element-wise multiplication to apply attention

        # Reshape the attended features back to a 4D tensor (batch_size, channels, height, width)
        reshaped_filters = x1.view(-1, 70, self.image_size // (2**3), self.image_size // (2**3))

        # Pass the attended features through fully connected layers to get the final output
        output = self.fc_layers(x1)
        # Return intermediate reshaped filters, conv output, and final output
        return reshaped_filters, x, output

    def calculate_accuracy(self, predicted, target):
        '''Calculate the accuracy of the model's predictions.
        '''

        num_data = target.size()[0]  # Get the number of samples in the batch
        predicted = torch.argmax(predicted, dim=1)  # Get predicted class indices (max logit per sample)
        correct_pred = torch.sum(predicted == target)  # Count correct predictions

        accuracy = correct_pred * (100 / num_data)  # Compute accuracy as a percentage

        return accuracy.item()  # Return accuracy as a Python scalar


# Assuming AttentionCNN is already defined as in your previous code

class MultiViewAttentionCNN(nn.Module):
    '''A multi-view CNN architecture with attention mechanism for three parallel image inputs, each for a different classification task.'''

    def __init__(self, image_size, image_depth, num_classes_list, drop_prob, device):
        '''Initialize the multi-view model with three AttentionCNN submodules, each for a different task.'''

        super(MultiViewAttentionCNN, self).__init__()

        # Ensure num_classes_list has three elements
        assert len(num_classes_list) == 3, "num_classes_list should have three elements"

        # Store parameters
        self.image_size = image_size
        self.image_depth = image_depth
        self.num_classes_list = num_classes_list
        self.drop_prob = drop_prob
        self.device = device

        # Instantiate three AttentionCNN submodules, each with its own num_classes
        self.cnn_view_a = AttentionCNN(image_size, image_depth, num_classes_list[0], drop_prob, device)
        self.cnn_view_b = AttentionCNN(image_size, image_depth, num_classes_list[1], drop_prob, device)
        self.cnn_view_c = AttentionCNN(image_size, image_depth, num_classes_list[2], drop_prob, device)

    def forward(self, view_a, view_b, view_c, return_individual_outputs=False, return_attention_features=False):
        # Process each view through its respective AttentionCNN
        features_a_reshaped_filters, features_a_x, features_a_output = self.cnn_view_a(view_a)  # Extract feature vector
        features_b_reshaped_filters, features_b_x, features_b_output = self.cnn_view_b(view_b)
        features_c_reshaped_filters, features_c_x, features_c_output = self.cnn_view_c(view_c)

        # print(f"Size of features_a_reshaped_filters : {features_a_reshaped_filters.size()}")
        # print(f"Size of features_a_x                : {features_a_x.size()}")
        # print(f"Size of features_a_output           : {features_a_output.size()}")
        # print(f"Size of features_b_reshaped_filters : {features_b_reshaped_filters.size()}")
        # print(f"Size of features_b_x                : {features_b_x.size()}")
        # print(f"Size of features_b_output           : {features_b_output.size()}")
        # print(f"Size of features_c_reshaped_filters : {features_c_reshaped_filters.size()}")
        # print(f"Size of features_c_x                : {features_c_x.size()}")
        # print(f"Size of features_c_output           : {features_c_output.size()}")


        if return_individual_outputs:
            if return_attention_features:
                return features_a_output, features_b_output, features_c_output, features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters
            else:
                return features_a_output, features_b_output, features_c_output
        else:
            # Concatenate attended features for fusion
            combined_features = torch.cat((features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters), dim=1)
            combined_features = combined_features.view(combined_features.size(0), -1)
            fused_output = self.fusion_layers(combined_features)
            if return_attention_features:
                return fused_output, features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters
            else:
                return fused_output
                

    def calculate_accuracy(self, predicted, target):
        '''Calculate the accuracy of the model's predictions.'''
        num_data = target.size()[0]
        predicted = torch.argmax(predicted, dim=1)
        correct_pred = torch.sum(predicted == target)
        accuracy = correct_pred * (100 / num_data)
        return accuracy.item()

