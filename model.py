import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    An image encoder model based on a pretrained ResNet.
    It removes the final classification layers of the ResNet and adds a
    new head to produce a feature vector of a specified dimension.
    """
    def __init__(self, feature_dim=256):
        """
        Initializes the ImageEncoder model.

        Args:
            feature_dim (int): The desired dimension of the output feature vector.
        """
        super(ImageEncoder, self).__init__()

        # Load a pretrained ResNet18 model
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Create the backbone by removing the original average pooling and
        # fully connected layers of the ResNet.
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Create a new head to process the features from the backbone.
        # The ResNet18 backbone outputs features with 512 channels.
        self.head = nn.Sequential(
            # Adaptive average pooling reduces the spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            # A 1x1 convolution changes the number of channels to feature_dim
            nn.Conv2d(in_channels=512, out_channels=feature_dim, kernel_size=1),
            # Flatten the output to get a feature vector of shape [B, feature_dim]
            nn.Flatten()
        )

    def forward(self, x):
        """
        Forward pass for the image encoder.

        Args:
            x (torch.Tensor): Input image tensor with shape [B, C, H, W].

        Returns:
            torch.Tensor: Encoded feature vector with shape [B, feature_dim].
        """
        features = self.backbone(x)
        encoded = self.head(features)
        return encoded

# --- Example Usage ---
if __name__ == '__main__':
    # This block will only run when the script is executed directly.
    # It serves as a quick test to verify the model's architecture and output shape.

    # Create an instance of the model with a desired feature dimension
    feature_dimension = 256
    model = ImageEncoder(feature_dim=feature_dimension)
    
    # Put the model in evaluation mode
    model.eval()

    print("--- Model Architecture ---")
    print(model)

    # Create a dummy input tensor representing a batch of images
    # Batch size = 4, Channels = 3, Height = 224, Width = 224
    dummy_input = torch.randn(4, 3, 224, 224)

    # Pass the input through the model
    with torch.no_grad(): # Disable gradient calculation for inference
        output_features = model(dummy_input)

    # Print the shapes to verify
    print("\n--- Shape Verification ---")
    print(f"Dummy input shape:  {dummy_input.shape}")
    print(f"Output features shape: {output_features.shape}")
    
    # Check if the output shape is correct
    assert output_features.shape == (4, feature_dimension)
    print("\nOutput shape is correct.")
