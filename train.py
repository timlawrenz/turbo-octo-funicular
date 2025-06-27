import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SceneDataset
from model import SceneReconstructionModel

def main():
    """
    Main function to set up and run a boilerplate training loop.
    """
    # --- 1. Dataset and DataLoader Setup ---

    # Define the image transformation pipeline
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset with the defined transformations
    # Assumes the generated data is in the 'data' directory.
    DATA_DIRECTORY = 'data'
    try:
        full_dataset = SceneDataset(data_dir=DATA_DIRECTORY, transform=image_transform)
        print(f"Successfully loaded dataset with {len(full_dataset)} scenes.")
    except FileNotFoundError:
        print(f"Error: Dataset directory '{DATA_DIRECTORY}' not found.")
        print("Please ensure you have generated the data using the Blender script.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # Create a DataLoader for batching, shuffling, and parallel loading
    data_loader = DataLoader(
        full_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    # --- 2. Training Setup ---

    # Set up the device (use CUDA if available, otherwise fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate the model and move it to the device
    model = SceneReconstructionModel().to(device)

    # Instantiate optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training parameters
    num_epochs = 50

    # --- 3. Training Loop ---
    
    print("\nStarting training loop...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        running_loss = 0.0
        
        # Enumerate over the DataLoader to get batches of data
        for i, batch in enumerate(data_loader):
            # Move all tensors in the batch to the selected device
            images = batch['images'].to(device)
            poses = batch['poses'].to(device)
            gt_location = batch['gt_location'].to(device)

            # --- Forward pass ---
            outputs = model(images, poses)
            
            # --- Loss calculation ---
            loss = criterion(outputs, gt_location)
            
            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track and report loss
            running_loss += loss.item()
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Batch {i + 1}/{len(data_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {epoch_loss:.4f}")

    print("\nTraining loop finished.")

if __name__ == '__main__':
    # This block ensures the code runs only when the script is executed directly
    main()
