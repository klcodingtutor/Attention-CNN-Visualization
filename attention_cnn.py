'''
CNN with an attention mechanism.
'''

import torch
import torch.nn as nn

class AttentionCNN(nn.Module):
    '''A CNN architecture with an attention mechanism at the final layer.'''

    def __init__(self, image_size, image_depth, num_classes, drop_prob, device):
        super(AttentionCNN, self).__init__()

        self.image_size = image_size
        self.image_depth = image_depth
        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self.device = device

        self.build_model()

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def build_model(self):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.image_depth, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 70, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Calculate feature dimensions after convolutions
        self.feature_channels = 70
        self.feature_size = self.image_size // (2**3)
        self.feature_vector_size = self.feature_channels * (self.feature_size)**2
        self.scale = nn.Parameter(torch.zeros(1))

        # Attention components (applied at the last layer)
        self.norm = nn.LayerNorm(self.feature_channels)
        self.mha = nn.MultiheadAttention(embed_dim=self.feature_channels, num_heads=5)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_vector_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_prob),
            nn.Linear(256, self.num_classes)
        )

    def use_attention(self, x):
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)  # BSxHWxC
        x_att = self.norm(x_att)
        att_out, att_map = self.mha(x_att, x_att, x_att)
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    def forward(self, x):
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Apply attention at the last layer
        attended_features, attention_map = self.use_attention(x)
        x = self.scale * attended_features + x  # Residual connection
        
        # Flatten for fully connected layers
        x_flat = x.reshape(x.size(0), -1)
        
        # Final classification
        output = self.fc_layers(x_flat)
        
        return attended_features, x, output

    def calculate_accuracy(self, predicted, target):
        num_data = target.size()[0]
        predicted = torch.argmax(predicted, dim=1)
        correct_pred = torch.sum(predicted == target)
        accuracy = correct_pred * (100 / num_data)
        return accuracy.item()

    
class MultiViewAttentionCNN(nn.Module):
    '''A multi-view CNN architecture with attention mechanism for three parallel image inputs.'''

    def __init__(self, image_size, image_depth, num_classes_list, num_classes_final, drop_prob, device):
        super(MultiViewAttentionCNN, self).__init__()

        assert len(num_classes_list) == 3, "num_classes_list should have three elements"

        self.image_size = image_size
        self.image_depth = image_depth
        self.num_classes_list = num_classes_list
        self.drop_prob = drop_prob
        self.device = device
        self.num_classes_final = num_classes_final

        # Three AttentionCNN submodules
        self.cnn_view_a = AttentionCNN(image_size, image_depth, num_classes_list[0], drop_prob, device)
        self.cnn_view_b = AttentionCNN(image_size, image_depth, num_classes_list[1], drop_prob, device)
        self.cnn_view_c = AttentionCNN(image_size, image_depth, num_classes_list[2], drop_prob, device)

        single_feature_size = self.cnn_view_a.feature_vector_size
        self.combined_feature_size = 3 * single_feature_size

        self.fusion_layers = nn.Sequential(
            nn.Linear(self.combined_feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_prob),
            nn.Linear(512, self.num_classes_final)
        )

        self.fusion_layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, view_a, view_b, view_c, return_individual_outputs=False, return_attention_features=False):
        features_a_reshaped_filters, features_a_x, features_a_output = self.cnn_view_a(view_a)
        features_b_reshaped_filters, features_b_x, features_b_output = self.cnn_view_b(view_b)
        features_c_reshaped_filters, features_c_x, features_c_output = self.cnn_view_c(view_c)

        if return_individual_outputs:
            if return_attention_features:
                return features_a_output, features_b_output, features_c_output, features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters
            else:
                return features_a_output, features_b_output, features_c_output
        else:
            # print(f"Shape of combined_features: {features_a_reshaped_filters.shape}, {features_b_reshaped_filters.shape}, {features_c_reshaped_filters.shape}")
            combined_features = torch.cat((features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters), dim=1)
            # print(f"Shape of combined_features after cat: {features_a_reshaped_filters.shape}, {features_b_reshaped_filters.shape}, {features_c_reshaped_filters.shape}")
            combined_features = combined_features.reshape(combined_features.size(0), -1)
            # print(f"Shape of combined_features after reshape: {features_a_reshaped_filters.shape}, {features_b_reshaped_filters.shape}, {features_c_reshaped_filters.shape}")
            fused_output = self.fusion_layers(combined_features)
            if return_attention_features:
                return fused_output, features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters
            else:
                return fused_output

    def calculate_accuracy(self, predicted, target):
        num_data = target.size()[0]
        predicted = torch.argmax(predicted, dim=1)
        correct_pred = torch.sum(predicted == target)
        accuracy = correct_pred * (100 / num_data)
        return accuracy.item()