import os
import json
import random
import math

def calculate_random_baseline(data_dir='data'):
    """
    Calculates the average Mean Squared Error (MSE) by making random guesses
    for object locations across the entire dataset.

    Args:
        data_dir (str): The root directory of the dataset.
    """
    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return

    print(f"Calculating random guess baseline for dataset in '{data_dir}'...")

    scene_folders = sorted([d for d in os.listdir(data_dir) if d.startswith('scene_')])
    total_squared_error = 0
    num_scenes = len(scene_folders)

    if num_scenes == 0:
        print("No scenes found in the directory.")
        return

    for scene_name in scene_folders:
        json_path = os.path.join(data_dir, scene_name, 'scene_data.json')
        
        with open(json_path, 'r') as f:
            scene_data = json.load(f)

        # Get the ground truth location of the first object
        gt_location = scene_data['objects'][0]['location']

        # --- Generate a Random Guess ---
        # Based on the Blender script, objects are placed in this range.
        # We assume a similar height range for Z for a fair guess.
        random_x = random.uniform(-10, 10)
        random_y = random.uniform(-10, 10)
        random_z = random.uniform(0, 3) # Objects sit on the plane, so height is ~ their radius
        predicted_location = [random_x, random_y, random_z]

        # --- Calculate Squared Error for this one sample ---
        squared_error = sum([(gt - pred)**2 for gt, pred in zip(gt_location, predicted_location)])
        total_squared_error += squared_error

    # --- Calculate and Report the Final Average Loss ---
    average_mse = total_squared_error / num_scenes
    average_rmse = math.sqrt(average_mse)

    print("\n--- Baseline Test Results ---")
    print(f"Total scenes processed: {num_scenes}")
    print(f"Average Mean Squared Error (MSE) for Random Guesses: {average_mse:.4f}")
    print(f"Average Root Mean Squared Error (RMSE) for Random Guesses: {average_rmse:.4f}")
    print(f"\nInterpretation: On average, a random guess is {average_rmse:.2f} units away from the true location.")


if __name__ == '__main__':
    # Make sure to point this to the directory containing your 'scene_xxxx' folders
    DATA_DIRECTORY = 'data'
    calculate_random_baseline(data_dir=DATA_DIRECTORY)
