import bpy
import random
import math
import json
import os

def generate_scenes():
    """
    Main function to generate a number of scenes.
    """
    # Set the total number of scenes to generate.
    # Increase this value to create a large dataset.
    num_scenes_to_generate = 10000

    for scene_idx in range(num_scenes_to_generate):
        print(f"--- Starting generation for scene {scene_idx + 1}/{num_scenes_to_generate} ---")

        # --- Scene Cleaning ---
        # Ensure we are in Object Mode
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Select all objects in the scene
        bpy.ops.object.select_all(action='SELECT')
        # Delete all selected objects (meshes, cameras, lights, etc.)
        if bpy.context.selected_objects:
            bpy.ops.object.delete()

        # Purge orphaned data blocks (materials, meshes, etc.) to keep the file clean
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        print("All scene objects and data purged.")


        # --- Stage Creation ---
        # Create a large ground plane that will serve as our stage
        print("Creating a large ground plane...")
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0)
        )
        ground_plane = bpy.context.active_object
        ground_plane.name = "GroundPlane"
        ground_plane.scale = (100, 100, 1)
        print("Ground plane created.")


        # --- Lighting Setup ---
        print("Creating and configuring a Sun light...")
        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 5))
        sun_light = bpy.context.active_object
        sun_light.name = "SunLight"
        light_data = sun_light.data
        base_x_rot_deg = 45
        random_x_rot_deg = random.uniform(-15, 15)
        sun_light.rotation_euler.x = math.radians(base_x_rot_deg + random_x_rot_deg)
        random_y_rot_deg = random.uniform(-15, 15)
        sun_light.rotation_euler.y = math.radians(random_y_rot_deg)
        sun_light.rotation_euler.z = random.uniform(0, math.pi * 2)
        light_data.energy = random.uniform(2.5, 5.0)
        print("Sun light created with randomized properties.")


        # --- Object Creation ---
        print("Creating and placing random objects...")
        scene_data = {'objects': [], 'camera_poses': []}
        object_types = ['CUBE', 'SPHERE', 'PYRAMID']
        num_objects = random.randint(1, 3)

        # Temporary list to hold object data before sorting
        objects_to_sort = []

        for i in range(num_objects):
            obj_type = random.choice(object_types)

            if obj_type == 'CUBE':
                bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
            elif obj_type == 'SPHERE':
                bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
            elif obj_type == 'PYRAMID':
                bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=1, depth=2, location=(0, 0, 0))

            obj = bpy.context.active_object
            obj.name = f"{obj_type.capitalize()}_{i+1}"
            
            s = random.uniform(0.8, 1.5)
            obj.scale = (s, s, s)
            obj.rotation_euler.x = random.uniform(0, 2 * math.pi)
            obj.rotation_euler.y = random.uniform(0, 2 * math.pi)
            obj.rotation_euler.z = random.uniform(0, 2 * math.pi)
            
            bpy.context.view_layer.update()
            matrix_world = obj.matrix_world
            world_vertices = [matrix_world @ v.co for v in obj.data.vertices]
            lowest_z = min(v.z for v in world_vertices)
            
            random_x = random.uniform(-10, 10)
            random_y = random.uniform(-10, 10)
            obj.location = (random_x, random_y, -lowest_z)

            # Record ground truth data into a temporary list
            object_info = {
                'object_type': obj_type,
                'location': list(obj.location),
                'rotation': list(obj.rotation_euler),
                'scale': list(obj.scale)
            }
            objects_to_sort.append(object_info)

            mat = bpy.data.materials.new(name=f"RandomMat_{scene_idx}_{i+1}")
            mat.use_nodes = True
            principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if principled_bsdf:
                principled_bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
            obj.data.materials.append(mat)
            
            print(f"Created {obj.name} with a random color, scale, and rotation.")
        
        # Sort the objects by their distance to the origin (0,0,0) for a canonical order.
        # Sorting by distance squared is equivalent and avoids a sqrt calculation.
        objects_to_sort.sort(key=lambda o: o['location'][0]**2 + o['location'][1]**2 + o['location'][2]**2)
        
        # Add the canonically ordered objects to the final scene data
        scene_data['objects'] = objects_to_sort

        print(f"Finished creating and sorting {num_objects} objects.")


        # --- Camera, Rendering, and Pose Saving ---
        print("Setting up camera and rendering...")
        output_dir = f"output/scene_{scene_idx:04d}"
        os.makedirs(output_dir, exist_ok=True)

        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.active_object
        camera.name = "SceneCamera"
        bpy.context.scene.camera = camera

        bpy.ops.object.empty_add(location=(0, 0, 0))
        track_target = bpy.context.active_object
        track_target.name = "TrackTarget"
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = track_target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        num_frames = 16
        radius = 25
        z_height = 12

        for i in range(num_frames):
            angle = (i / num_frames) * 2 * math.pi
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            camera.location = (x, y, z_height)
            
            bpy.context.view_layer.update()
            
            frame_filepath = os.path.join(output_dir, f"frame_{i:02d}.png")
            bpy.context.scene.render.filepath = frame_filepath
            
            pose_info = {
                'location': list(camera.location),
                'rotation': list(camera.rotation_euler),
                'image_path': frame_filepath
            }
            scene_data['camera_poses'].append(pose_info)
            
            bpy.ops.render.render(write_still=True)
            print(f"Rendered frame {i+1}/{num_frames} to {frame_filepath}")

        # --- Export Ground Truth Data ---
        json_filepath = os.path.join(output_dir, 'scene_data.json')
        with open(json_filepath, 'w') as f:
            json.dump(scene_data, f, indent=4)

        print(f"Scene data saved to {json_filepath}")
        print(f"--- Finished scene {scene_idx + 1}/{num_scenes_to_generate} ---")

# Run the main function when the script is executed
if __name__ == '__main__':
    generate_scenes()
