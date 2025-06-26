import bpy
import random
import math

# --- Scene Cleaning ---
# Ensure we are in Object Mode
if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
    bpy.ops.object.mode_set(mode='OBJECT')

# Deselect all objects
bpy.ops.object.select_all(action='DESELECT')

# Select all objects in the scene
for obj in bpy.data.objects:
    obj.select_set(True)

# Delete all selected objects
if bpy.context.selected_objects:
    bpy.ops.object.delete()

print("All scene objects deleted.")


# --- Stage Creation ---
# Create a large ground plane that will serve as our stage
print("Creating a large ground plane...")
bpy.ops.mesh.primitive_plane_add(
    size=1,
    enter_editmode=False,
    align='WORLD',
    location=(0, 0, 0)
)
# The new plane is now the active object
ground_plane = bpy.context.active_object
ground_plane.name = "GroundPlane"

# Scale the plane to make it large
ground_plane.scale = (100, 100, 1)

print("Ground plane created.")


# --- Lighting Setup ---
# Create a light source, like a "Sun" lamp
print("Creating and configuring a Sun light...")
bpy.ops.object.light_add(
    type='SUN',
    radius=1,
    align='WORLD',
    location=(0, 0, 5) # Position it above the ground plane
)
sun_light = bpy.context.active_object
sun_light.name = "SunLight"

# Access the light data
light_data = sun_light.data

# Randomize its rotation slightly. We'll set a base angle and add some randomness.
# The sun will generally point from a 45-degree angle on the X-axis.
base_x_rot_deg = 45
random_x_rot_deg = random.uniform(-15, 15)
sun_light.rotation_euler.x = math.radians(base_x_rot_deg + random_x_rot_deg)

random_y_rot_deg = random.uniform(-15, 15)
sun_light.rotation_euler.y = math.radians(random_y_rot_deg)

# Randomize Z rotation (spin) more freely
sun_light.rotation_euler.z = random.uniform(0, math.pi * 2) # Full 360 degrees

# Randomize its energy (brightness) slightly.
# A typical sun energy is > 1. Let's use a range from 2.5 to 5.0.
light_data.energy = random.uniform(2.5, 5.0)

print("Sun light created with randomized properties.")


# --- Object Creation ---
print("Creating and placing random objects...")

object_types = ['CUBE', 'SPHERE', 'PYRAMID']
num_objects = random.randint(1, 3)

for i in range(num_objects):
    # Choose a random object type
    obj_type = random.choice(object_types)

    # Create the object at the origin first
    if obj_type == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    elif obj_type == 'SPHERE':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
    elif obj_type == 'PYRAMID':
        # A cone with 4 vertices is a pyramid
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=1, depth=2, location=(0, 0, 0))

    obj = bpy.context.active_object
    obj.name = f"{obj_type.capitalize()}_{i+1}"

    # Calculate placement on the ground plane
    # The plane is 100x100, so it extends from -50 to +50. We'll place objects a bit inside the edges.
    random_x = random.uniform(-45, 45)
    random_y = random.uniform(-45, 45)
    # To sit on the plane, the object's Z location must be half its own height (since origin is center)
    z_offset = obj.dimensions.z / 2
    obj.location = (random_x, random_y, z_offset)
    
    # --- Create and assign a random material ---
    mat = bpy.data.materials.new(name=f"RandomMat_{i+1}")
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        # Assign a random RGB color
        principled_bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
    
    # Assign material to the object
    obj.data.materials.append(mat)
    
    print(f"Created {obj.name} with a random color.")

print(f"Finished creating {num_objects} objects.")
