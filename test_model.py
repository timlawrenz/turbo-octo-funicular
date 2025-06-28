import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from model import SceneReconstructionModel


class TestSceneReconstructionModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.num_frames = 16
        self.feature_dim = 512
        self.batch_size = 2
        self.image_channels = 3
        self.image_height = 224
        self.image_width = 224
        self.pose_dim = 6
    
    def test_init_default_parameters(self):
        """Test model initialization with default parameters."""
        model = SceneReconstructionModel()
        
        self.assertEqual(model.num_frames, 16)
        self.assertEqual(model.feature_dim, 512)
        
        # Check that all components are initialized
        self.assertIsInstance(model.image_encoder, nn.Sequential)
        self.assertIsInstance(model.feature_adapter, nn.Sequential)
        self.assertIsInstance(model.pose_encoder, nn.Sequential)
        self.assertIsInstance(model.output_head, nn.Sequential)
    
    def test_init_custom_parameters(self):
        """Test model initialization with custom parameters."""
        custom_frames = 8
        custom_feature_dim = 256
        
        model = SceneReconstructionModel(
            num_frames=custom_frames, 
            feature_dim=custom_feature_dim
        )
        
        self.assertEqual(model.num_frames, custom_frames)
        self.assertEqual(model.feature_dim, custom_feature_dim)
    
    def test_forward_pass_output_shape(self):
        """Test that forward pass produces correct output shape."""
        model = SceneReconstructionModel()
        
        # Create dummy input tensors
        images = torch.randn(self.batch_size, self.num_frames, 
                           self.image_channels, self.image_height, self.image_width)
        poses = torch.randn(self.batch_size, self.num_frames, self.pose_dim)
        
        # Forward pass
        output = model(images, poses)
        
        # Check output shape
        expected_shape = (self.batch_size, 3)  # 3D location prediction
        self.assertEqual(output.shape, expected_shape)
    
    def test_forward_pass_output_type(self):
        """Test that forward pass produces torch.Tensor output."""
        model = SceneReconstructionModel()
        
        images = torch.randn(self.batch_size, self.num_frames, 
                           self.image_channels, self.image_height, self.image_width)
        poses = torch.randn(self.batch_size, self.num_frames, self.pose_dim)
        
        output = model(images, poses)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, torch.float32)
    
    def test_forward_pass_single_batch(self):
        """Test forward pass with batch size 1."""
        model = SceneReconstructionModel()
        
        images = torch.randn(1, self.num_frames, 
                           self.image_channels, self.image_height, self.image_width)
        poses = torch.randn(1, self.num_frames, self.pose_dim)
        
        output = model(images, poses)
        
        self.assertEqual(output.shape, (1, 3))
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = SceneReconstructionModel()
        
        for batch_size in [1, 4, 8]:
            with self.subTest(batch_size=batch_size):
                images = torch.randn(batch_size, self.num_frames, 
                                   self.image_channels, self.image_height, self.image_width)
                poses = torch.randn(batch_size, self.num_frames, self.pose_dim)
                
                output = model(images, poses)
                
                self.assertEqual(output.shape, (batch_size, 3))
    
    def test_forward_pass_custom_num_frames(self):
        """Test forward pass with custom number of frames."""
        custom_frames = 8
        model = SceneReconstructionModel(num_frames=custom_frames)
        
        images = torch.randn(self.batch_size, custom_frames, 
                           self.image_channels, self.image_height, self.image_width)
        poses = torch.randn(self.batch_size, custom_frames, self.pose_dim)
        
        output = model(images, poses)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
    
    def test_image_encoder_component(self):
        """Test that image encoder produces expected output shape."""
        model = SceneReconstructionModel()
        
        # Create flattened images as they would be processed internally
        batch_size_flat = self.batch_size * self.num_frames
        images_flat = torch.randn(batch_size_flat, self.image_channels, 
                                self.image_height, self.image_width)
        
        # Test image encoder
        image_features_raw = model.image_encoder(images_flat)
        
        # ResNet18 final conv layer has 512 channels
        expected_channels = 512
        self.assertEqual(image_features_raw.shape[0], batch_size_flat)
        self.assertEqual(image_features_raw.shape[1], expected_channels)
    
    def test_feature_adapter_component(self):
        """Test that feature adapter produces expected output shape."""
        model = SceneReconstructionModel()
        
        # Create dummy image features from ResNet
        batch_size_flat = self.batch_size * self.num_frames
        resnet_features = torch.randn(batch_size_flat, 512, 7, 7)  # Typical ResNet output
        
        # Test feature adapter
        adapted_features = model.feature_adapter(resnet_features)
        
        expected_shape = (batch_size_flat, self.feature_dim, 1, 1)
        self.assertEqual(adapted_features.shape, expected_shape)
    
    def test_pose_encoder_component(self):
        """Test that pose encoder produces expected output shape."""
        model = SceneReconstructionModel()
        
        poses = torch.randn(self.batch_size, self.num_frames, self.pose_dim)
        
        # Test pose encoder
        pose_features = model.pose_encoder(poses)
        
        expected_shape = (self.batch_size, self.num_frames, self.feature_dim)
        self.assertEqual(pose_features.shape, expected_shape)
    
    def test_output_head_component(self):
        """Test that output head produces expected output shape."""
        model = SceneReconstructionModel()
        
        # Create dummy scene vector
        scene_vector = torch.randn(self.batch_size, self.feature_dim)
        
        # Test output head
        predicted_location = model.output_head(scene_vector)
        
        expected_shape = (self.batch_size, 3)
        self.assertEqual(predicted_location.shape, expected_shape)
    
    def test_model_parameters_require_grad(self):
        """Test that model parameters require gradients by default."""
        model = SceneReconstructionModel()
        
        for param in model.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_model_train_eval_modes(self):
        """Test that model can switch between train and eval modes."""
        model = SceneReconstructionModel()
        
        # Test train mode
        model.train()
        self.assertTrue(model.training)
        
        # Test eval mode
        model.eval()
        self.assertFalse(model.training)
    
    def test_model_device_movement(self):
        """Test that model can be moved to different devices."""
        model = SceneReconstructionModel()
        
        # Test CPU (default)
        self.assertTrue(all(param.device.type == 'cpu' for param in model.parameters()))
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            self.assertTrue(all(param.device.type == 'cuda' for param in model.parameters()))
    
    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic given same inputs."""
        model = SceneReconstructionModel()
        model.eval()  # Disable dropout if any
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        images = torch.randn(self.batch_size, self.num_frames, 
                           self.image_channels, self.image_height, self.image_width)
        poses = torch.randn(self.batch_size, self.num_frames, self.pose_dim)
        
        # First forward pass
        output1 = model(images, poses)
        
        # Second forward pass with same inputs
        output2 = model(images, poses)
        
        # Outputs should be identical
        torch.testing.assert_close(output1, output2)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SceneReconstructionModel()
        
        images = torch.randn(self.batch_size, self.num_frames, 
                           self.image_channels, self.image_height, self.image_width)
        poses = torch.randn(self.batch_size, self.num_frames, self.pose_dim)
        
        # Forward pass
        output = model(images, poses)
        
        # Dummy loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_pose_dimension_validation(self):
        """Test that model handles correct pose dimensions."""
        model = SceneReconstructionModel()
        
        images = torch.randn(self.batch_size, self.num_frames, 
                           self.image_channels, self.image_height, self.image_width)
        
        # Test with correct pose dimension (6)
        poses_correct = torch.randn(self.batch_size, self.num_frames, 6)
        output = model(images, poses_correct)
        self.assertEqual(output.shape, (self.batch_size, 3))
        
        # Test with incorrect pose dimension should raise an error
        poses_incorrect = torch.randn(self.batch_size, self.num_frames, 5)
        with self.assertRaises(RuntimeError):
            model(images, poses_incorrect)


if __name__ == '__main__':
    unittest.main()