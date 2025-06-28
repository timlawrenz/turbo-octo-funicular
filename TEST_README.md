# Unit Tests for Turbo-Octo-Funicular

This directory contains comprehensive unit tests for the core files of the turbo-octo-funicular project.

## Test Files

### `test_dataset.py`
Tests for the `SceneDataset` class in `dataset.py`:
- ✅ Dataset initialization with valid/invalid directories
- ✅ Dataset length calculation
- ✅ Item retrieval with and without transformations
- ✅ Scene directory filtering and sorting
- ✅ Error handling for missing data

### `test_model.py`
Tests for the `SceneReconstructionModel` class in `model.py`:
- ✅ Model initialization with default and custom parameters
- ✅ Forward pass functionality and output shapes
- ✅ Individual component testing (image encoder, pose encoder, output head)
- ✅ Device movement and training/evaluation modes
- ✅ Gradient flow and parameter requirements
- ✅ Deterministic behavior and error handling

### `test_train.py`
Tests for the training script `train.py`:
- ✅ Main function structure and error handling
- ✅ Device selection logic (CPU/CUDA)
- ✅ Dataset and model integration
- ✅ Hyperparameter configuration
- ✅ Loss computation functionality
- ✅ Import and dependency validation

## Running Tests

### Run All Tests
```bash
# Using the provided test runner
python run_tests.py

# Using unittest discovery
python -m unittest discover -s . -p 'test_*.py' -v
```

### Run Individual Test Modules
```bash
# Test dataset functionality
python -m unittest test_dataset.py -v

# Test model functionality  
python -m unittest test_model.py -v

# Test training functionality
python -m unittest test_train.py -v
```

### Run Specific Test Classes or Methods
```bash
# Test specific class
python -m unittest test_dataset.TestSceneDataset -v

# Test specific method
python -m unittest test_model.TestSceneReconstructionModel.test_forward_pass_output_shape -v
```

## Test Coverage

The tests cover:
- **Basic functionality**: All core methods and classes
- **Error handling**: Invalid inputs and edge cases
- **Integration**: Components working together
- **Shape validation**: Tensor dimensions and types
- **Device compatibility**: CPU/CUDA support
- **Configuration**: Parameters and hyperparameters

## Test Structure

Tests use Python's built-in `unittest` framework and follow these patterns:
- **Mocking**: External dependencies are mocked where appropriate
- **Temporary files**: Test data is created in temporary directories
- **Isolation**: Tests are independent and don't affect each other
- **Descriptive names**: Test names clearly describe what is being tested

## Dependencies

Tests require the same dependencies as the main project:
- `torch`
- `torchvision`
- `Pillow`

No additional testing dependencies are needed.

## Total Test Count

**41 tests** across all modules ensure comprehensive coverage of the core functionality.