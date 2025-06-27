import torch
from torchvision import transforms
import numpy as np
import random
import os

from dataset import SceneDataset
from model import SceneReconstructionModel

def evaluate_model():
    """
    Loads a trained model and evaluates it on a single sample from the dataset.
    """
    # --- 1. Setup ---
    DATA_DIRECTORY = 'data'
    MODEL_PATH = 'best_model.pth'
    
    # Pick a random scene to evaluate
    try:
        num_scenes = len([name for name in os.listdir(DATA_DIRECTORY) if os.path.isdir(os.path.join(DATA_DIRECTORY, name))])
        SAMPLE_INDEX = random.randint(0, num_scenes - 1)
    except FileNotFoundError:
        print(f"Error: Dataset directory '{DATA_DIRECTORY}' not found. Cannot determine sample index.")
        return
    except ValueError:
         print(f"Error: No scenes found in '{DATA_DIRECTORY}'.")
         return


    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load the Model ---
    try:
        model = SceneReconstructionModel().to(device)
        # Load the saved weights. map_location ensures it works even if trained on GPU and run on CPU.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()  # Set model to evaluation mode
        print(f"Model loaded successfully from {MODEL_PATH}.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please ensure you have trained the model and the 'best_model.pth' file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # --- 3. Load the Dataset ---
    # Define the same image transformation pipeline as used in training
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = SceneDataset(data_dir=DATA_DIRECTORY, transform=image_transform)
        print(f"Dataset loaded with {len(dataset)} scenes.")
    except FileNotFoundError:
        print(f"Error: Dataset directory '{DATA_DIRECTORY}' not found.")
        return
    
    # --- 4. Prepare a Single Sample ---
    if SAMPLE_INDEX >= len(dataset):
        print(f"Error: Sample index {SAMPLE_INDEX} is out of bounds for a dataset of size {len(dataset)}.")
        return

    print(f"\nEvaluating on sample from scene index: {SAMPLE_INDEX}")
    sample = dataset[SAMPLE_INDEX]

    # The model expects a batch, so we add a batch dimension (B=1)
    images = sample['images'].unsqueeze(0).to(device)
    poses = sample['poses'].unsqueeze(0).to(device)

    # --- 5. Make a Prediction ---
    with torch.no_grad():  # Disable gradient calculations for inference
        predicted_location = model(images, poses)

    # --- 6. Display the Results ---
    ground_truth = sample['gt_location'].numpy()
    # Squeeze to remove the batch dimension, move to CPU, detach from graph, and convert to numpy
    prediction = predicted_location.squeeze().cpu().detach().numpy()

    print("\n--- Model Evaluation ---")
    print(f"Ground Truth Location:  X={ground_truth[0]:.2f}, Y={ground_truth[1]:.2f}, Z={ground_truth[2]:.2f}")
    print(f"Predicted Location:     X={prediction[0]:.2f}, Y={prediction[1]:.2f}, Z={prediction[2]:.2f}")

    # Optional: Calculate and print the distance error for this specific sample
    error = np.linalg.norm(ground_truth - prediction)
    print(f"\nDistance Error: {error:.2f} units")


if __name__ == '__main__':
    evaluate_model()
