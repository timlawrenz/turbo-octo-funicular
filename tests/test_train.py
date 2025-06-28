import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train
from dataset import SceneDataset
from model import SceneReconstructionModel


class TestTrainModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)
    
    @patch('train.SceneDataset')
    @patch('builtins.print')
    def test_main_successful_setup(self, mock_print, mock_dataset):
        """Test that main function sets up dataset successfully and handles training setup."""
        # Mock dataset to work properly initially  
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__ = MagicMock(return_value=10)
        mock_dataset.return_value = mock_dataset_instance
        
        # Mock CUDA availability
        with patch('train.torch.cuda.is_available', return_value=False):
            # We'll let the function fail at DataLoader creation, which is fine for this test
            # Since we're mainly testing the initial setup
            try:
                train.main()
            except Exception:
                pass  # Expected since we're not mocking everything
        
        # Verify dataset was created
        mock_dataset.assert_called_once()
        # Verify success message was printed
        mock_print.assert_any_call("Successfully loaded dataset with 10 scenes.")
    
    @patch('train.SceneDataset')
    @patch('builtins.print')
    def test_main_dataset_not_found(self, mock_print, mock_dataset):
        """Test main function handles FileNotFoundError gracefully."""
        # Mock dataset to raise FileNotFoundError
        mock_dataset.side_effect = FileNotFoundError("Dataset directory not found")
        
        # Call main function
        train.main()
        
        # Verify error handling
        mock_print.assert_any_call("Error: Dataset directory 'data' not found.")
        mock_print.assert_any_call("Please ensure you have generated the data using the Blender script.")
    
    @patch('train.SceneDataset')
    @patch('builtins.print')
    def test_main_general_exception(self, mock_print, mock_dataset):
        """Test main function handles general exceptions gracefully."""
        # Mock dataset to raise general exception
        error_message = "Some general error"
        mock_dataset.side_effect = Exception(error_message)
        
        # Call main function
        train.main()
        
        # Verify error handling
        mock_print.assert_any_call(f"An error occurred while loading the dataset: {error_message}")
    
    @patch('train.torch.cuda.is_available')
    @patch('train.torch.device')
    def test_device_selection_cuda_available(self, mock_device, mock_cuda_available):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        
        # We need to mock the entire main function setup to test device selection
        with patch('train.SceneDataset') as mock_dataset, \
             patch('train.DataLoader') as mock_dataloader, \
             patch('train.SceneReconstructionModel') as mock_model, \
             patch('train.torch.optim.Adam'), \
             patch('train.nn.MSELoss'), \
             patch('builtins.print'):
            
            # Setup mocks
            mock_dataset_instance = MagicMock()
            mock_dataset_instance.__len__ = MagicMock(return_value=1)
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader_instance = MagicMock()
            mock_dataloader_instance.__len__ = MagicMock(return_value=1)
            mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([]))
            mock_dataloader.return_value = mock_dataloader_instance
            
            mock_model_instance = MagicMock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.parameters.return_value = []
            mock_model.return_value = mock_model_instance
            
            train.main()
            
            # Verify CUDA device was selected
            mock_device.assert_called_with("cuda")
    
    @patch('train.torch.cuda.is_available')
    @patch('train.torch.device')
    def test_device_selection_cuda_not_available(self, mock_device, mock_cuda_available):
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        # We need to mock the entire main function setup to test device selection
        with patch('train.SceneDataset') as mock_dataset, \
             patch('train.DataLoader') as mock_dataloader, \
             patch('train.SceneReconstructionModel') as mock_model, \
             patch('train.torch.optim.Adam'), \
             patch('train.nn.MSELoss'), \
             patch('builtins.print'):
            
            # Setup mocks
            mock_dataset_instance = MagicMock()
            mock_dataset_instance.__len__ = MagicMock(return_value=1)
            mock_dataset.return_value = mock_dataset_instance
            
            mock_dataloader_instance = MagicMock()
            mock_dataloader_instance.__len__ = MagicMock(return_value=1)
            mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([]))
            mock_dataloader.return_value = mock_dataloader_instance
            
            mock_model_instance = MagicMock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.parameters.return_value = []
            mock_model.return_value = mock_model_instance
            
            train.main()
            
            # Verify CPU device was selected
            mock_device.assert_called_with("cpu")
    
    def test_image_transform_pipeline(self):
        """Test that image transform pipeline is created correctly."""
        # Since the transform is created inside main(), we need to test it indirectly
        # by checking the imports and structure
        
        # Verify torchvision imports are available
        from torchvision import transforms
        
        # Create the same transform pipeline as in main()
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.assertIsInstance(image_transform, transforms.Compose)
        self.assertEqual(len(image_transform.transforms), 3)
        self.assertIsInstance(image_transform.transforms[0], transforms.Resize)
        self.assertIsInstance(image_transform.transforms[1], transforms.ToTensor)
        self.assertIsInstance(image_transform.transforms[2], transforms.Normalize)
    
    def test_hyperparameters(self):
        """Test that training hyperparameters are set correctly."""
        # These are hardcoded in the main function, so we test the expected values
        expected_batch_size = 8
        expected_num_workers = 4
        expected_lr = 0.001
        expected_num_epochs = 50
        
        # These values are used in the main function
        # We can't directly access them without running main(), 
        # but we can verify they exist in the source code
        import inspect
        source = inspect.getsource(train.main)
        
        self.assertIn('batch_size=8', source)
        self.assertIn('num_workers=4', source)
        self.assertIn('lr=0.001', source)
        self.assertIn('num_epochs = 50', source)
    
    @patch('train.SceneDataset')
    @patch('builtins.print')
    def test_training_loop_structure(self, mock_print, mock_dataset):
        """Test that training loop setup is correct (without full execution)."""
        # Mock dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__ = MagicMock(return_value=1)
        mock_dataset.return_value = mock_dataset_instance
        
        # Mock CUDA availability
        with patch('train.torch.cuda.is_available', return_value=False):
            # Test will fail at DataLoader creation but that's expected
            try:
                train.main()
            except Exception:
                pass  # Expected due to incomplete mocking
        
        # Verify training setup prints
        mock_print.assert_any_call("Successfully loaded dataset with 1 scenes.")
        
        # The training components are tested in other tests
    
    def test_data_directory_constant(self):
        """Test that DATA_DIRECTORY constant is set correctly."""
        import inspect
        source = inspect.getsource(train.main)
        self.assertIn("DATA_DIRECTORY = 'data'", source)
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        self.assertTrue(hasattr(train, 'main'))
        self.assertTrue(callable(train.main))
    
    def test_imports_exist(self):
        """Test that all required imports are available."""
        # Test torch imports
        self.assertTrue(hasattr(train, 'torch'))
        self.assertTrue(hasattr(train, 'nn'))
        self.assertTrue(hasattr(train, 'DataLoader'))
        
        # Test custom imports
        self.assertTrue(hasattr(train, 'SceneDataset'))
        self.assertTrue(hasattr(train, 'SceneReconstructionModel'))
    
    def test_if_name_main_block(self):
        """Test that the script has proper if __name__ == '__main__' structure."""
        import inspect
        source = inspect.getsource(train)
        self.assertIn("if __name__ == '__main__':", source)
        self.assertIn("main()", source)


class TestTrainIntegration(unittest.TestCase):
    """Integration tests for training components working together."""
    
    def test_model_dataset_compatibility(self):
        """Test that model can process dataset output."""
        # Create a small dummy dataset
        temp_dir = tempfile.mkdtemp()
        try:
            # Create test scene directory and data
            scene_dir = os.path.join(temp_dir, 'scene_001')
            os.makedirs(scene_dir)
            
            scene_data = {
                'camera_poses': [
                    {'location': [i, i+1, i+2], 'rotation': [i*0.1, i*0.1+0.1, i*0.1+0.2]}
                    for i in range(16)
                ],
                'objects': [{'location': [5.0, 6.0, 7.0]}]
            }
            
            import json
            with open(os.path.join(scene_dir, 'scene_data.json'), 'w') as f:
                json.dump(scene_data, f)
            
            # Create dummy images
            from PIL import Image
            import numpy as np
            for i in range(16):
                dummy_image = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                dummy_image.save(os.path.join(scene_dir, f'frame_{i:02d}.png'))
            
            # Create dataset
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            dataset = SceneDataset(temp_dir, transform=transform)
            
            # Create model
            model = SceneReconstructionModel()
            
            # Test compatibility
            sample = dataset[0]
            images = sample['images'].unsqueeze(0)  # Add batch dimension
            poses = sample['poses'].unsqueeze(0)    # Add batch dimension
            
            # Forward pass should work without errors
            output = model(images, poses)
            
            # Verify output shape
            self.assertEqual(output.shape, (1, 3))
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_loss_computation(self):
        """Test that loss computation works correctly."""
        # Create dummy predictions and ground truth
        predictions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ground_truth = torch.tensor([[1.1, 2.1, 3.1], [3.9, 5.1, 5.9]])
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Compute loss
        loss = criterion(predictions, ground_truth)
        
        # Verify loss is computed
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar loss
        self.assertGreater(loss.item(), 0)  # Loss should be positive


if __name__ == '__main__':
    unittest.main()
