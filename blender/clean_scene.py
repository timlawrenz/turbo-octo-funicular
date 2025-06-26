import bpy

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
