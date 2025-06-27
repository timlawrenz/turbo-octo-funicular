import torch
import torch.nn as nn
import torchvision.models as models

class SceneReconstructionModel(nn.Module):
    """
    A model that uses a ResNet backbone and a Transformer Encoder to predict
    an object's 3D location from a sequence of images and poses.
    """
    def __init__(self, num_frames=16, feature_dim=512, nhead=8, num_encoder_layers=4):
        """
        Initializes the SceneReconstructionModel.

        Args:
            num_frames (int): The number of frames in each scene sequence.
            feature_dim (int): The dimension for the image and pose features.
            nhead (int): The number of attention heads in the Transformer.
            num_encoder_layers (int): The number of layers in the Transformer Encoder.
        """
        super(SceneReconstructionModel, self).__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim

        # --- 1. Feature Extractors ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(512, feature_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # --- 2. Transformer Aggregation Components ---
        # Special learnable token that will be prepended to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # Learnable positional encodings for the sequence (CLS token + num_frames)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames + 1, feature_dim))
        
        # The Transformer Encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True  # This is important! It expects input as [B, Seq, Dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- 3. Output Head ---
        # The input to the head is the output of the [CLS] token from the transformer
        self.output_head = nn.Sequential(
            nn.LayerNorm(feature_dim), # Normalization is good practice here
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
        
        # --- Feature Extraction ---
        images_flat = images.view(batch_size * self.num_frames, *images.shape[2:])
        image_features_raw = self.image_encoder(images_flat)
        image_features = self.feature_adapter(image_features_raw).squeeze()
        image_features = image_features.view(batch_size, self.num_frames, self.feature_dim)
        
        pose_features = self.pose_encoder(poses)
        combined_features = image_features + pose_features

        # --- Transformer Aggregation ---
        # Prepend the [CLS] token to the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((cls_tokens, combined_features), dim=1)
        
        # Add positional encoding
        transformer_input += self.positional_encoding
        
        # Pass through the Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Extract the state of the [CLS] token (it's the first one in the sequence)
        # This is our aggregated scene vector
        scene_vector = transformer_output[:, 0, :]
        
        # --- Output Head ---
        predicted_location = self.output_head(scene_vector)
        
        return predicted_location
