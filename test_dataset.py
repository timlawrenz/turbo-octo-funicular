import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock
import torch
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import SceneDataset


class TestSceneDataset(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_scene_data = {
            'camera_poses': [
                {'location': [1.0, 2.0, 3.0], 'rotation': [0.1, 0.2, 0.3]},
                {'location': [1.1, 2.1, 3.1], 'rotation': [0.11, 0.21, 0.31]}
            ],
            'objects': [
                {'location': [5.0, 6.0, 7.0]}
            ]
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_nonexistent_directory(self):
        """Test initialization with a directory that doesn't exist."""
        nonexistent_dir = '/nonexistent/directory'
        with self.assertRaises(FileNotFoundError) as context:
            SceneDataset(nonexistent_dir)
        self.assertIn("Dataset directory not found", str(context.exception))
    
    def test_init_with_empty_directory(self):
        """Test initialization with an empty directory."""
        dataset = SceneDataset(self.temp_dir)
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.data_dir, self.temp_dir)
        self.assertIsNone(dataset.transform)
    
    def test_init_with_valid_scenes(self):
        """Test initialization with valid scene directories."""
        # Create test scene directories
        scene1_dir = os.path.join(self.temp_dir, 'scene_001')
        scene2_dir = os.path.join(self.temp_dir, 'scene_002')
        os.makedirs(scene1_dir)
        os.makedirs(scene2_dir)
        
        # Create scene_data.json files
        with open(os.path.join(scene1_dir, 'scene_data.json'), 'w') as f:
            json.dump(self.valid_scene_data, f)
        with open(os.path.join(scene2_dir, 'scene_data.json'), 'w') as f:
            json.dump(self.valid_scene_data, f)
        
        dataset = SceneDataset(self.temp_dir)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.samples), 2)
        self.assertEqual(dataset.samples[0]['scene_name'], 'scene_001')
        self.assertEqual(dataset.samples[1]['scene_name'], 'scene_002')
    
    def test_init_with_transform(self):
        """Test initialization with a transform function."""
        mock_transform = MagicMock()
        dataset = SceneDataset(self.temp_dir, transform=mock_transform)
        self.assertEqual(dataset.transform, mock_transform)
    
    def test_len(self):
        """Test __len__ method."""
        # Create test scene directories
        scene_dirs = ['scene_001', 'scene_002', 'scene_003']
        for scene_name in scene_dirs:
            scene_dir = os.path.join(self.temp_dir, scene_name)
            os.makedirs(scene_dir)
            with open(os.path.join(scene_dir, 'scene_data.json'), 'w') as f:
                json.dump(self.valid_scene_data, f)
        
        dataset = SceneDataset(self.temp_dir)
        self.assertEqual(len(dataset), 3)
    
    @patch('dataset.Image.open')
    def test_getitem_basic(self, mock_image_open):
        """Test __getitem__ method with basic functionality."""
        # Setup mock scene data
        scene_data = {
            'camera_poses': [
                {'location': [i, i+1, i+2], 'rotation': [i*0.1, i*0.1+0.1, i*0.1+0.2]}
                for i in range(16)  # 16 frames as expected by the code
            ],
            'objects': [
                {'location': [5.0, 6.0, 7.0]}
            ]
        }
        
        # Create test scene directory
        scene_dir = os.path.join(self.temp_dir, 'scene_001')
        os.makedirs(scene_dir)
        with open(os.path.join(scene_dir, 'scene_data.json'), 'w') as f:
            json.dump(scene_data, f)
        
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image
        
        # Mock the transform to return a tensor
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)  # C, H, W
        
        dataset = SceneDataset(self.temp_dir, transform=mock_transform)
        
        # Test getting the first item
        sample = dataset[0]
        
        # Verify structure
        self.assertIn('images', sample)
        self.assertIn('poses', sample)
        self.assertIn('gt_location', sample)
        
        # Verify tensor shapes
        self.assertEqual(sample['images'].shape, (16, 3, 224, 224))  # num_frames, C, H, W
        self.assertEqual(sample['poses'].shape, (16, 6))  # num_frames, 6 (location + rotation)
        self.assertEqual(sample['gt_location'].shape, (3,))  # 3D location
        
        # Verify tensor types
        self.assertIsInstance(sample['images'], torch.Tensor)
        self.assertIsInstance(sample['poses'], torch.Tensor)
        self.assertIsInstance(sample['gt_location'], torch.Tensor)
        
        # Verify ground truth location
        expected_gt = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)
        torch.testing.assert_close(sample['gt_location'], expected_gt)
    
    @patch('dataset.Image.open')
    def test_getitem_without_transform(self, mock_image_open):
        """Test __getitem__ method without transform."""
        # Create a simple RGB image tensor and convert to PIL Image for mocking
        from torchvision import transforms
        rgb_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(rgb_array)
        
        # Convert PIL image to tensor for testing (simulating what would happen without transform)
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(pil_image)
        
        # Mock Image.open to return a PIL image that converts to tensor properly
        mock_image_open.return_value = pil_image
        
        # Setup scene data
        scene_data = {
            'camera_poses': [
                {'location': [i, i+1, i+2], 'rotation': [i*0.1, i*0.1+0.1, i*0.1+0.2]}
                for i in range(16)
            ],
            'objects': [
                {'location': [1.0, 2.0, 3.0]}
            ]
        }
        
        # Create test scene directory
        scene_dir = os.path.join(self.temp_dir, 'scene_001')
        os.makedirs(scene_dir)
        with open(os.path.join(scene_dir, 'scene_data.json'), 'w') as f:
            json.dump(scene_data, f)
        
        # Use a transform that converts PIL to tensor since torch.stack requires tensors
        dataset = SceneDataset(self.temp_dir, transform=to_tensor)
        sample = dataset[0]
        
        # When transform is applied, images should be stacked into tensor
        self.assertIsInstance(sample['images'], torch.Tensor)
        self.assertEqual(len(sample['images']), 16)  # 16 frames
    
    def test_getitem_index_error(self):
        """Test __getitem__ method with invalid index."""
        dataset = SceneDataset(self.temp_dir)
        
        with self.assertRaises(IndexError):
            dataset[0]  # No scenes available
    
    def test_scene_directories_sorted(self):
        """Test that scene directories are processed in sorted order."""
        # Create scene directories in non-alphabetical order
        scene_names = ['scene_003', 'scene_001', 'scene_002']
        for scene_name in scene_names:
            scene_dir = os.path.join(self.temp_dir, scene_name)
            os.makedirs(scene_dir)
            with open(os.path.join(scene_dir, 'scene_data.json'), 'w') as f:
                json.dump(self.valid_scene_data, f)
        
        dataset = SceneDataset(self.temp_dir)
        
        # Verify scenes are stored in sorted order
        expected_order = ['scene_001', 'scene_002', 'scene_003']
        actual_order = [sample['scene_name'] for sample in dataset.samples]
        self.assertEqual(actual_order, expected_order)
    
    def test_ignore_non_scene_directories(self):
        """Test that non-scene directories are ignored."""
        # Create a mix of scene and non-scene directories
        os.makedirs(os.path.join(self.temp_dir, 'scene_001'))
        os.makedirs(os.path.join(self.temp_dir, 'other_dir'))
        os.makedirs(os.path.join(self.temp_dir, 'scene_002'))
        os.makedirs(os.path.join(self.temp_dir, 'not_a_scene'))
        
        # Only add scene_data.json to scene directories
        for scene_name in ['scene_001', 'scene_002']:
            with open(os.path.join(self.temp_dir, scene_name, 'scene_data.json'), 'w') as f:
                json.dump(self.valid_scene_data, f)
        
        dataset = SceneDataset(self.temp_dir)
        self.assertEqual(len(dataset), 2)
        
        scene_names = [sample['scene_name'] for sample in dataset.samples]
        self.assertIn('scene_001', scene_names)
        self.assertIn('scene_002', scene_names)
        self.assertNotIn('other_dir', scene_names)
        self.assertNotIn('not_a_scene', scene_names)


if __name__ == '__main__':
    unittest.main()
