# Project: 3D Object Localization from Synthetic Data

This project is an end-to-end pipeline for training a deep learning model to solve a 3D computer vision task using synthetically generated data.

## High-Level Goal

The primary objective is to **train a model that can infer the 3D location of an object by looking at it from multiple different viewpoints.** This is a classic 3D reconstruction problem, and this pipeline provides the tools to generate data, load it, and run a training loop.

## The Pipeline

The project is broken down into three main parts, each handled by a specific script.

### 1. Synthetic Data Generation (`blender/clean_scene.py`)

-   **Purpose**: Real-world 3D ground truth data is difficult and expensive to acquire. This script uses Blender to programmatically generate a limitless supply of "perfect" training data.
-   **Process**: For each scene, it creates a randomized environment with 1-3 objects (cubes, spheres, or pyramids) of varying colors, scales, and rotations. It then renders the scene from 16 different camera angles around the objects.
-   **Output**: The script produces a directory for each scene containing:
    -   16 rendered `.png` images.
    -   A `scene_data.json` file containing the **ground truth**: the exact 3D location, rotation, and scale of every object, plus the precise camera pose for each rendered image.

### 2. Data Loading & Preparation (`dataset.py`)

-   **Purpose**: A folder of images and JSON files is not a format that a machine learning framework like PyTorch can use directly. This script acts as the bridge.
-   **Process**: The `SceneDataset` class reads the generated data and packages it for training. A single "sample" from this dataset consists of an entire scene:
    -   A stacked tensor of all 16 images.
    -   A tensor of all 16 corresponding camera poses.
    -   The ground truth 3D location of the target object.
-   **Features**: It supports `torchvision.transforms` to apply on-the-fly image augmentations like resizing and normalization.

### 3. Model Training (`train.py`)

-   **Purpose**: This is the main script for training a model.
-   **Process**: It sets up a standard training loop that:
    1.  Initializes the `SceneDataset` and `DataLoader`.
    2.  Automatically selects a CUDA device if available.
    3.  Feeds the model batches of data, where each item contains the `images`, `poses`, and the `gt_location`.
    4.  (Future) Compares the model's predicted location to the ground truth and updates the model's weights to improve its accuracy.

## How to Run the Full Pipeline

1.  **Generate Data**
    First, use Blender to generate the dataset. This script will create an `output/` directory with all the scenes.
    ```bash
    blender --background --python blender/clean_scene.py
    ```

2.  **Install Dependencies**
    Install the required Python packages for training.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Training**
    Execute the training script. This will load the data generated in step 1 and begin the training loop.
    ```bash
    python train.py
    ```

## Potential Model Architecture

The data structure produced by this pipeline (multiple images + camera poses to predict a 3D property) is the exact input required for a class of models like **Neural Radiance Fields (NeRF)**. While the current goal is to predict a single object's location, this framework could easily be extended to train a NeRF to reconstruct an entire scene's geometry and appearance.
