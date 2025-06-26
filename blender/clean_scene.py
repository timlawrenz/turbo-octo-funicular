import bpy

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
