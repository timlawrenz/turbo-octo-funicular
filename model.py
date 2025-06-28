"""
Neural network model for 3D object localization from multi-view images.

This module defines the SceneReconstructionModel that processes multiple
camera views and poses to predict 3D object locations.
"""
import torch
from torch import nn
from torchvision import models

class SceneReconstructionModel(nn.Module):
    """
    An image encoder model based on a pretrained ResNet.
    It removes the final classification layers of the ResNet and adds a
    new head to produce a feature vector of a specified dimension.
    """
    def __init__(self, num_frames=16, feature_dim=512):
        """
        Initializes the ImageEncoder model.

        Args:
            feature_dim (int): The desired dimension of the output feature vector.
        """
        super().__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim

        # Load a pretrained ResNet18 model
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Create the backbone by removing the original average pooling and
        # fully connected layers of the ResNet.
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Add a final convolutional layer and adaptive pooling to get the desired feature dimension
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(512, feature_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2. Pose Encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # 3. Output Head
        self.output_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output is a 3D location (x, y, z)
        )

    def forward(self, images, poses):
        """
        Defines the forward pass of the model.

        Args:
            images (torch.Tensor): A tensor of images with shape [B, N, C, H, W]
            poses (torch.Tensor): A tensor of poses with shape [B, N, 6]

        Returns:
            torch.Tensor: The predicted 3D location(s) with shape [B, 3]
        """
        batch_size = images.shape[0]

        # --- 1. Encoding ---
        # Reshape images to process them all at once: [B*N, C, H, W]
        images_flat = images.view(batch_size * self.num_frames, *images.shape[2:])

        # Get image features
        image_features_raw = self.image_encoder(images_flat)
        image_features = self.feature_adapter(image_features_raw).squeeze() # Shape: [B*N, D]
        # Reshape back: [B, N, D]
        image_features = image_features.view(batch_size, self.num_frames, self.feature_dim)

        # Get pose features
        pose_features = self.pose_encoder(poses) # Shape: [B, N, D]

        # --- 2. Fusion & Aggregation ---
        # Fuse by element-wise addition
        combined_features = image_features + pose_features

        # Aggregate across the frames dimension (dim=1) using mean pooling
        scene_vector = torch.mean(combined_features, dim=1) # Shape: [B, D]

        # --- 3. Output Head ---
        predicted_location = self.output_head(scene_vector) # Shape: [B, 3]

        return predicted_location
