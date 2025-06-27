# Blender Scene Generation Script (`clean_scene.py`)

This directory contains a Python script designed to be run within Blender for generating a synthetic image dataset. The script, `clean_scene.py`, automates the creation of thousands of unique scenes, rendering them from multiple camera angles and exporting ground truth data for each one.

## Core Functionality

The script performs the following actions in a loop to generate each unique scene:

1.  **Scene Cleanup**: Before each new scene is created, the script performs a thorough cleanup, deleting all objects, materials, and meshes from the previous iteration to ensure no data carries over and to manage memory usage effectively.

2.  **Stage Creation**: A large ground plane is created to serve as the stage for the objects.

3.  **Randomized Lighting**: A "Sun" light source is added to the scene. Its rotation and energy (brightness) are slightly randomized to provide varied lighting conditions across different scenes.

4.  **Random Object Placement**:
    *   The script places between 1 and 3 objects in the scene.
    *   Each object's shape is randomly chosen from a list: `CUBE`, `SPHERE`, or `PYRAMID`.
    *   Each object is assigned a random color via a new material.
    *   The scale and rotation of each object are randomized to increase variety.
    *   Objects are carefully placed at a random (X, Y) location near the center of the stage. The script calculates each object's lowest point after all transformations (scaling, rotation) to ensure it rests perfectly on the ground plane without intersecting it.

5.  **Camera and Rendering**:
    *   A camera is created and set to always point at the center of the scene `(0,0,0)`.
    *   The camera moves along a circular arc around the objects.
    *   For each scene, it renders 16 frames from different viewpoints along this arc.

6.  **Ground Truth Export**:
    *   For each scene, a `scene_data.json` file is created.
    *   This JSON file contains detailed ground truth information, including:
        *   The type, location, rotation, and scale of every object.
        *   The location, rotation, and corresponding image path for each of the 16 camera poses.

## Output Structure

The script generates all output into a top-level `output/` directory (relative to where Blender is executed). Each scene gets its own subfolder, named sequentially.

