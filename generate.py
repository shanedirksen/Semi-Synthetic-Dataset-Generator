import maya.cmds as cmds
import maya.mel as mel
import maya.app.general.createImageFormats as createImageFormats
from objectDetection import bounding_boxes
import os
import gc
import random
import uuid
from datetime import datetime
import yaml
import itertools
import glob
import shutil
import time

start_time = time.time()

# This is gross hard coding because I can't figure out how to make the textures apply when importing on just these
texture_files = {
    'pringles_bbq': 'C:/Users/shane/Documents/Maya/2022/scripts/objects/new_objects/pringles_bbq/Pringles-BBQ_revised.png',
    'honey_bunches_of_oats_honey_roasted': 'C:/Users/shane/Documents/Maya/2022/scripts/objects/new_objects/honey_bunches_of_oats_honey_roasted/uv.png',
    'pop_secret_light_butter': 'C:/Users/shane/Documents/Maya/2022/scripts/objects/new_objects/pop_secret_light_butter/uv.png',
    'hunts_sauce': 'C:/Users/shane/Documents/Maya/2022/scripts/objects/new_objects/hunts_sauce/hunt_sauce_texture.png'
}

object_names = {
    'pringles_bbq': 'pringles_bbq:Cap2group',
    'honey_bunches_of_oats_honey_roasted': 'honey_bunches_of_oats_honey_roasted:honey_bunches_of_oats_honey_roasted_obj_3d_modelgroup',
    'pop_secret_light_butter': 'pop_secret_light_butter:MeshBoundingShapegroup',
    'hunts_sauce': 'hunts_sauce:CirclegroupShape'
}

def apply_material_to_object(file, object_to_assign, texture_path):
    # Create a file node
    file_node = cmds.shadingNode('file', asTexture=True)
    # Set the file texture path
    cmds.setAttr(file_node + '.fileTextureName', texture_path, type='string')

    # If the file is 'hunts_sauce', use a Phong shader
    if 'hunts_sauce' in file:
        shader_node = cmds.shadingNode('phong', asShader=True)
    else:
        shader_node = cmds.shadingNode('lambert', asShader=True)

    # Create a shading group
    shading_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True)
    # Connect the shader to the shading group
    cmds.connectAttr(shader_node + '.outColor', shading_group + '.surfaceShader', force=True)
    # Connect the file node to the shader
    cmds.connectAttr(file_node + '.outColor', shader_node + '.color', force=True)

    # Assign the shading group to the object
    if cmds.objExists(object_to_assign):
        cmds.sets(object_to_assign, e=True, forceElement=shading_group)
    else:
        print(f'Object {object_to_assign} does not exist')

def duplicate_as_lambert(shader_name):
    # Create a new Lambert shader
    lambert_shader = cmds.shadingNode('lambert', asShader=True, name=shader_name + "_lambert")

    # Copy attributes from the original shader to the new Lambert shader
    attributes = ['color', 'transparency', 'ambientColor', 'incandescence', 'diffuse', 'translucence']
    for attr in attributes:
        value = cmds.getAttr(shader_name + "." + attr)
        if isinstance(value, tuple) or isinstance(value, list):
            cmds.setAttr(lambert_shader + "." + attr, value[0][0], value[0][1], value[0][2], type='double3')
        else:
            cmds.setAttr(lambert_shader + "." + attr, value)

    # Get shading group of the original shader
    shading_group = cmds.listConnections(shader_name, type='shadingEngine')
    if shading_group:
        # Assign the new Lambert shader to the shading group
        cmds.connectAttr(lambert_shader + '.outColor', shading_group[0] + '.surfaceShader', force=True)

    return lambert_shader

# Function to randomly add a negative object using addMEL.mel
def add_negative_object(imported_negatives):
    print("adding random negative INSIDE CODE")

    # Randomly select a negative object from the imported_negatives list
    chosen_object = random.choice(imported_negatives)

    # If the chosen object is the coffee_cup, adjust the shaders
    if chosen_object == "coffee_cup":
        shaders = ["blinn1SG1", "blinn2SG1"]
        reflectivity_value = 0.0577778
        specular_color = [0.0533333, 0.0533333, 0.0533333]

        for shader in shaders:
            # Adjust reflectivity
            reflectivity_attr = shader + ".reflectivity"
            if cmds.objExists(reflectivity_attr):
                cmds.setAttr(reflectivity_attr, reflectivity_value)

            # Adjust specular color
            specular_attr = shader + ".specularColor"
            if cmds.objExists(specular_attr):
                cmds.setAttr(specular_attr, specular_color[0], specular_color[1], specular_color[2], type="double3")

    # Add random rotations
    random_rotation = random.randint(0, 360)
    cmds.rotate(0, random_rotation, 0, chosen_object, os=True, r=True)

    # Select the chosen object and the plane
    cmds.select(clear=True)
    cmds.select(chosen_object)
    cmds.select('pPlane2.f[0:99]', add=True)

    # Run the fulladdMEL script
    mel.eval('source "C:/Users/shane/Documents/Maya/2022/scripts/fulladdMEL.mel"')
    mel.eval('fulladdMEL')

    # The addMEL script should have created a duplicate with the name appended by "1"
    placed_object = chosen_object + "1"

    # Return the name of the placed object for further operations
    print("Placed object:", placed_object)
    return placed_object




def save_txt(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

def find_texture_file(mtl_file_path):
    # Try to open the .mtl file
    try:
        with open(mtl_file_path, 'r') as mtl_file:
            for line in mtl_file:
                # Look for the line that specifies the texture file
                if line.startswith('map_Kd'):
                    # Get the texture file name from the line
                    texture_file_name = line.split(' ')[1].strip()
                    # Combine the .mtl file directory with the texture file name
                    texture_file_path = os.path.join(os.path.dirname(mtl_file_path), texture_file_name)
                    # Check if the texture file exists
                    if os.path.exists(texture_file_path):
                        return texture_file_path
    except FileNotFoundError:
        pass
    return None

#Runs MEL Script
def runMEL():
    print("Running MEL from Python")
    mel.eval('source "C:/Users/shane/Documents/Maya/2022/scripts/fulladdMEL.mel"')
    banana = mel.eval("fulladdMEL;")
    return banana

#Renders complete image
def renderJPEG(name):
    mel.eval('renderWindowRender redoPreviousRender renderView')
    editor = 'renderView'
    formatManager = createImageFormats.ImageFormats()
    formatManager.pushRenderGlobalsForDesc("JPEG")
    cmds.renderWindowEditor(editor, e=True, colorManage=True, writeImage='C:/Users/shane/Documents/Maya/2022/scripts/objectDetection/{0}.jpg'.format(name.replace('_obj', '')))
    formatManager.popRenderGlobals()

#Renders alpha images
def renderPNG(name):
    cmds.setAttr('imagePlane1.displayMode', 0) #off
    cmds.setAttr('defaultArnoldRenderOptions.ignoreLights', 1) #off
    cmds.setAttr('pPlane2.visibility', 0)  # off
    mel.eval('renderWindowRender redoPreviousRender renderView')
    editor = 'renderView'
    formatManager = createImageFormats.ImageFormats()
    formatManager.pushRenderGlobalsForDesc("PNG")
    cmds.renderWindowEditor(editor, e=True, colorManage=True, writeImage="C:/Users/shane/Documents/Maya/2022/scripts/objectDetection/{0}.png".format(name.replace('_obj', '')))
    formatManager.popRenderGlobals()
    cmds.setAttr('imagePlane1.displayMode', 2) #on
    cmds.setAttr('defaultArnoldRenderOptions.ignoreLights', 0) #on
    cmds.setAttr('pPlane2.visibility', 1)  # off

#Prompts number of pictures
result = cmds.promptDialog(
                title='Maximum Number of Combinations',
                message='Enter number:',
                button=['OK', 'Cancel'],
                defaultButton='OK',
                cancelButton='Cancel',
                dismissString='Cancel')
if result == 'OK':
        text = cmds.promptDialog(query=True, text=True)
else:
    quit()
iterations = int(text)

# Prompt for total number of outputs
result = cmds.promptDialog(
    title='Total Number of Outputs per Scene',
    message='Enter number or "all":',
    button=['OK', 'Cancel'],
    defaultButton='OK',
    cancelButton='Cancel',
    dismissString='Cancel')

if result == 'OK':
    text_outputs = cmds.promptDialog(query=True, text=True)
else:
    quit()

print("Beginning dataset generation...")

# Set the path to the folder containing the scenes
scenes_folder = 'C:/Users/shane/Documents/Maya/2022/scripts/scenes'

# Find all the .mb files in the folder
scene_files = glob.glob(scenes_folder + '/*.mb')

# Iterate through the scene files and open each one
for scene_file in scene_files:
    cmds.file(scene_file, o=True, force=True)
    print("Loading scene")
# Define the scenes folder

# # Find and load only scene10.mb from the folder
# scene_file = glob.glob(scenes_folder + '/scene10.mb')[0] if glob.glob(scenes_folder + '/scene10.mb') else None
# if scene_file:
#     cmds.file(scene_file, o=True, force=True)
#     print("Loading scene: " + scene_file)
    # Delete unused nodes
    mel.eval('hyperShadePanelMenuCommand("hyperShadePanel1", "deleteUnusedNodes");')

    #Make sure the camera settings are correct
    editor = 'renderView'
    cmds.renderWindowEditor( editor, e=True, colorManage=True, crc='camera1' )
    cmds.lookThru( 'camera1' )
    cmds.setAttr("defaultArnoldRenderOptions.renderDevice", 1)
    cmds.setAttr('perspShape.renderable', False)
    cmds.setAttr("frontShape.renderable", False)
    cmds.setAttr("sideShape.renderable", False)
    cmds.setAttr("topShape.renderable", False)
    cmds.setAttr("cameraShape1.renderable", True)
    cmds.setAttr("defaultArnoldRenderOptions.renderDevice", 1) #These three are computationally expensive
    cmds.setAttr("defaultArnoldRenderOptions.AASamples", 4)
    cmds.setAttr("defaultArnoldFilter.width", 2)
    mel.eval('RenderViewWindow;')

    pathOfFiles = "C:/Users/shane/Documents/Maya/2022/scripts/objects/new_objects"

    fileslist = {}
    for dirpath, subdirs, files in os.walk(pathOfFiles):
        for x in files:
            # Determine the fileType based on the directory name
            if 'coca_cola_glass_bottle' in dirpath or 'palmoliveSoap' in dirpath:
                fileType = ".fbx"
            else:
                fileType = ".obj"

            if x.endswith(fileType):
                dirpath = dirpath.replace('\\', '/')
                name = dirpath.rsplit('/', 1)[-1]
                fileslist.update({name: [dirpath, x, name]})

    selectAll = []  # A list to store all imported objects
    if len(fileslist) == 0:
        cmds.warning("No files found")
    else:
        for f in fileslist:
            obj_file = os.path.join(fileslist[f][0], fileslist[f][1])
            if os.path.isfile(obj_file):
                # Store all objects before the import
                before_import = set(cmds.ls(transforms=True))
                # Import the OBJ file with a unique namespace
                newObj = cmds.file(obj_file, i=True, namespace=fileslist[f][2], mergeNamespacesOnClash=True)
                # Get all objects after the import
                after_import = set(cmds.ls(transforms=True))
                # New objects are the ones that are in after_import but not in before_import
                new_objects = list(after_import - before_import)
                # Group new objects
                group_name = fileslist[f][2] + "_group"
                group = cmds.group(new_objects, name=group_name)

                # For each texture in the dictionary, apply material
                for key in texture_files.keys():
                    if key in f:
                        texture_path = texture_files[key]
                        object_to_assign = object_names[key]
                        apply_material_to_object(f, object_to_assign, texture_path)

                # If the imported file is 'red_bull' or 'mahatma_rice', adjust the diffuse value
                if 'red_bull' in f:
                    # Create a new phong shader
                    phong_shader = cmds.shadingNode('phong', asShader=True, name='red_bull_phongShader')
                    shading_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                                              name='red_bull_phongSG')
                    cmds.connectAttr(phong_shader + '.outColor', shading_group + '.surfaceShader', force=True)

                    # Create a file node for the texture
                    texture_path = "C:/Users/shane/Documents/Maya/2022/scripts/objects/new_objects/red_bull/Red_Bull_can_250ml_livery.png"
                    file_node = cmds.shadingNode('file', asTexture=True)
                    cmds.setAttr(file_node + '.fileTextureName', texture_path, type='string')

                    # Connect the file node to the phong shader's color attribute
                    cmds.connectAttr(file_node + '.outColor', phong_shader + '.color', force=True)

                    # Assign the shader to the objects
                    objects_to_shade = ['red_bull:can_ringgroup', 'red_bull:can_body', 'red_bull:can_facegroup2']
                    for obj in objects_to_shade:
                        if cmds.objExists(obj):
                            cmds.select(obj, replace=True)  # This ensures only the current object is selected
                            cmds.hyperShade(assign=phong_shader)
                        else:
                            print(f"Object {obj} does not exist")

                    # # Adjust the shader attributes
                    cmds.setAttr(phong_shader + '.diffuse', 0.50)
                    cmds.setAttr(phong_shader + '.specularColor', 0.2266667, 0.2266667, 0.2266667, type="double3")
                    cmds.setAttr(phong_shader + '.reflectivity', 0.2444444)
                    cmds.setAttr(phong_shader + '.reflectedColor', 0.2355556, 0.2355556, 0.2355556, type="double3")

                if 'mahatma_rice' in f:
                    shading_group_to_adjust = 'mahatma_rice:initialShadingGroup1'
                    if cmds.objExists(shading_group_to_adjust):
                        cmds.setAttr(shading_group_to_adjust + '.diffuse', 0.15)
                    else:
                        print(f'Shading group {shading_group_to_adjust} does not exist')

                if 'pringles_bbq' in f:
                    shading_group_to_adjust = 'lambert4'
                    if cmds.objExists(shading_group_to_adjust):
                        cmds.setAttr(shading_group_to_adjust + '.diffuse', 0.5)
                    else:
                        print(f'Shading group {shading_group_to_adjust} does not exist')

                if 'hunts_sauce' in f:
                    # Define the material's name
                    material_name = 'phong1'

                    # Check if the material exists in the scene
                    if cmds.objExists(material_name):
                        # Set the attributes for the material
                        # Adjust the shader attributes for hunts_sauce
                        cmds.setAttr(material_name + '.diffuse', 0.25)
                        cmds.setAttr(material_name + '.specularColor', 0.111111, 0.111111, 0.111111, type="double3")
                        cmds.setAttr(material_name + '.reflectivity', 0)
                        cmds.setAttr(material_name + '.reflectedColor', 0.137778, 0.137778, 0.137778, type="double3")

                    else:
                        print(f'Material {material_name} does not exist')

                if 'coffee' in f:
                    # Define the object's name
                    object_name = 'coffee:Meshgroup'  # Updated to the correct name

                    # Check if the object exists in the scene
                    if cmds.objExists(object_name):
                        # Center the pivot to the object's bounding box
                        cmds.xform(object_name, centerPivots=True)

                        # Scale the object to make it 30% smaller
                        cmds.scale(0.7, 0.7, 0.7, object_name, r=True)

                        # Freeze transformations to bake the new pivot and scale in
                        cmds.makeIdentity(object_name, apply=True, t=1, r=1, s=1, n=0, pn=1)
                    else:
                        print(f"Object {object_name} does not exist in the scene.")

                if 'coca_cola_glass_bottle' in f:
                    bottle_obj = 'coca_cola_glass_bottle_group'
                    cap_obj = '|coca_cola_glass_bottle_group|cap'

                    # Specify the objects to be scaled
                    upper_liquid_obj = '|coca_cola_glass_bottle_group|upperFBXASC032liquid'
                    bottom_liquid_obj = '|coca_cola_glass_bottle_group|bottomFBXASC032liquid'

                    # Scale factors
                    scale_x, scale_y, scale_z = 1.405, 1.110, 1.307

                    # Break connections for the cap object
                    for attr in ['translateX', 'translateY', 'translateZ', 'rotateX', 'rotateY', 'rotateZ', 'scaleX',
                                 'scaleY', 'scaleZ']:
                        # Check if the connection exists
                        if cmds.connectionInfo(cap_obj + '.' + attr, isDestination=True):
                            source = cmds.connectionInfo(cap_obj + '.' + attr, sourceFromDestination=True)
                            cmds.disconnectAttr(source, cap_obj + '.' + attr)

                    if cmds.objExists(bottle_obj):
                        # Scale the group down
                        cmds.scale(0.0005, 0.0005, 0.0005, bottle_obj, relative=True)

                        # Move the group up by 1.9 units
                        cmds.move(0, 1.9, 0, bottle_obj, relative=True)

                        # Get all transform descendants of the bottle object
                        all_transforms = cmds.listRelatives(bottle_obj, allDescendents=True, type='transform',
                                                            fullPath=True) or []
                        all_transforms.append(bottle_obj)  # Include the bottle object itself

                        # Apply freeze transformations to each object
                        for obj in all_transforms:
                            cmds.makeIdentity(obj, apply=True, t=1, r=1, s=1, n=0, pn=1)

                        # Scale the specific liquid objects
                        if cmds.objExists(upper_liquid_obj) and cmds.objExists(bottom_liquid_obj):
                            cmds.scale(scale_x, scale_y, scale_z, upper_liquid_obj, bottom_liquid_obj)

                            # Freeze transformations for these objects again
                            cmds.makeIdentity(upper_liquid_obj, apply=True, t=1, r=1, s=1, n=0, pn=1)
                            cmds.makeIdentity(bottom_liquid_obj, apply=True, t=1, r=1, s=1, n=0, pn=1)
                        else:
                            print("One or both liquid objects do not exist in the scene.")

                        # Directly set the attributes of the material
                        material_name = "MaterialFBXASC046004new"
                        if cmds.objExists(material_name):
                            cmds.setAttr(material_name + ".color", 0, 0, 0, type="double3")
                            cmds.setAttr(material_name + ".transparency", 0.533645, 0.533645, 0.533645,
                                         type="double3")
                            cmds.setAttr(material_name + ".reflectedColor", 0, 0, 0,
                                         type="double3")
                            cmds.setAttr(material_name + ".specularColor", 0.261682, 0.261682, 0.261682,
                                         type="double3")
                        else:
                            print(f"Material {material_name} does not exist in the scene.")

                        # Directly set the attributes of the material
                        material_name = "MaterialFBXASC046002"
                        if cmds.objExists(material_name):
                            cmds.setAttr(material_name + ".transparency", 0.308411, 0.308411, 0.308411, type="double3")
                            cmds.setAttr(material_name + ".color", 0.0280374, 0.00937969, 0,
                                         type="double3")
                        else:
                            print(f"Material {material_name} does not exist in the scene.")

                    else:
                        print(f"Object {bottle_obj} does not exist in the scene.")

                if 'palmoliveSoap' in f:
                    # Set attributes for MaterialFBXASC046008
                    material1 = 'MaterialFBXASC046008'
                    if cmds.objExists(material1):
                        cmds.setAttr(material1 + ".transparency", 0.864486, 0.864486, 0.864486, type='double3')
                        cmds.setAttr(material1 + ".reflectivity", 0.883178)
                        cmds.setAttr(material1 + ".specularColor", 0.00934579, 0.00934579, 0.00934579, type='double3')
                        cmds.setAttr(material1 + ".reflectedColor", 0, 0, 0, type='double3')
                    else:
                        print(f"Material {material1} does not exist in the scene.")

                    # Set attributes for MaterialFBXASC046004
                    material2 = 'MaterialFBXASC046004'
                    if cmds.objExists(material2):
                        cmds.setAttr(material2 + ".color", 1, 0.1404, 0, type='double3')
                        cmds.setAttr(material2 + ".transparency", 0.478318, 0.478318, 0.478318, type='double3')
                        cmds.setAttr(material2 + ".specularColor", 0.271028, 0.117073, 0.0475504, type='double3')
                        cmds.setAttr(material2 + ".reflectedColor", 0.775701, 0.775701, 0.775701, type='double3')
                    else:
                        print(f"Material {material2} does not exist in the scene.")

                    # Scale adjustments for the liquid shape
                    liquid_obj = 'liquid'
                    if cmds.objExists(liquid_obj):
                        cmds.scale(1, 1.325251, 1, liquid_obj, relative=True)
                        cmds.scale(1.330044, 1, 1, liquid_obj, relative=True)
                        cmds.scale(1, 1, 1.043754, liquid_obj, relative=True)
                        # Freeze transformations to bake the initial scaling
                        cmds.makeIdentity(liquid_obj, apply=True, t=1, r=1, s=1, n=0, pn=1)
                    else:
                        print(f"Object {liquid_obj} does not exist in the scene.")

                    # Define the materials and their associated objects
                    material_objects = {
                        'MaterialFBXASC046005': 'frontFBXASC032label',
                        'Material': 'backFBXASC032label'
                    }

                    for material_name, object_name in material_objects.items():
                        if cmds.objExists(material_name) and cmds.objExists(object_name):
                            # Create a new Lambert shader
                            lambert_shader = cmds.shadingNode('lambert', asShader=True,
                                                              name=material_name + '_lambertShader')
                            # Create a new shading group for the Lambert shader
                            shading_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                                                      name=material_name + '_lambertSG')
                            # Connect the Lambert shader to the shading group
                            cmds.connectAttr(lambert_shader + '.outColor', shading_group + '.surfaceShader', force=True)

                            # Create a file texture node and set the image path
                            file_texture_node = cmds.shadingNode('file', asTexture=True, isColorManaged=True,
                                                                 name=material_name + '_fileTexture')
                            cmds.setAttr(file_texture_node + '.fileTextureName',
                                         "C:/Users/shane/Documents/Maya/2022/scripts/objects/new_objects/palmoliveSoap/from texture.png",
                                         type='string')
                            # Connect the file texture node's outColor to the Lambert shader's color attribute
                            cmds.connectAttr(file_texture_node + '.outColor', lambert_shader + '.color', force=True)

                            # Assign the new shading group to the object
                            cmds.hyperShade(assign=shading_group)
                        else:
                            if not cmds.objExists(material_name):
                                print(f"Material {material_name} does not exist")
                            if not cmds.objExists(object_name):
                                print(f"Object {object_name} does not exist")

                    # Scale adjustments for the palmoliveSoap object
                    palmolive_soap_obj = 'palmoliveSoap_group'  # Replace with the actual name of your palmoliveSoap object
                    center_reference_obj = 'hunts_sauce_group'  # The object already at the center

                    if cmds.objExists(palmolive_soap_obj):
                        if cmds.objExists(center_reference_obj):
                            # Align palmolive_soap_obj to the center_reference_obj's position
                            cmds.matchTransform(palmolive_soap_obj, center_reference_obj, position=True)

                            # Apply scaling of 0.00075 to the entire object
                            cmds.scale(0.0005, 0.0005, 0.0005, palmolive_soap_obj, relative=True)

                            # Freeze transformations to bake the new scale and position
                            cmds.makeIdentity(palmolive_soap_obj, apply=True, t=1, r=1, s=1, n=0, pn=1)
                        else:
                            print(f"Reference object {center_reference_obj} does not exist in the scene.")
                    else:
                        print(f"Object {palmolive_soap_obj} does not exist in the scene.")

                # Modify the pop_secret_light_butter material
                if 'pop_secret_light_butter' in f:
                    material_to_adjust = 'lambert3'
                    if cmds.objExists(material_to_adjust):
                        cmds.setAttr(material_to_adjust + '.diffuse', 2.5)
                    else:
                        print(f'Material {material_to_adjust} does not exist')

                # Add the group to the selectAll list
                selectAll.append(group)
            else:
                cmds.warning("File not found for: " + f)
                continue
    # quit()

    print("SELECTALL", selectAll)

    # Cleanup, move and scale the objects
    for obj in selectAll:
        cmds.select(obj)
        cmds.move(100, 100, 100, relative=True)
        cmds.scale(100, 100, 100, obj, r=True)
        cmds.makeIdentity(obj, apply=True, scale=True)
    # Define the specified location
    specified_folder = "C:/Users/shane/Documents/Maya/2022/scripts"
    # quit()

    # Create the dataset, train, and valid directories if they don't exist
    dataset_folder = os.path.join(specified_folder, 'dataset')
    train_folder = os.path.join(dataset_folder, 'train')
    valid_folder = os.path.join(dataset_folder, 'valid')

    for folder in [dataset_folder, train_folder, valid_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    config = {
        'path': dataset_folder.replace('\\', '/'),
        'train': train_folder.replace('\\', '/'),
        'val': valid_folder.replace('\\', '/'),
        'nc': len(fileslist),
        'names': list(fileslist.keys())
    }

    # Custom representer for the names list to force inline style
    def inline_list(dumper, data):
        return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, inline_list)

    # Save the YAML file in the dataset directory
    yaml_file_path = os.path.join(dataset_folder, "data.yaml")
    with open(yaml_file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    #Save txt for classes
    classes_txt_file_path = "C:/Users/shane/Documents/Maya/2022/scripts/dataset/classes.txt"
    names = list(fileslist.keys())
    save_txt(classes_txt_file_path, names)

    # print(selectAll)
    size = len(selectAll)

    # Create a list of the third items from each value in the fileslist dictionary
    import_names = [value[2] for value in fileslist.values()]

    all_combinations = []
    for i in range(1, iterations + 1):
        all_combinations.extend(list(itertools.combinations(import_names, i)))

    # Shuffle the list of combinations
    random.shuffle(all_combinations)

    # Select the first 'total_outputs' combinations from the shuffled list
    if text_outputs.lower() == 'all':
        selected_combinations = [tuple([name + "_group" for name in combo]) for combo in all_combinations]
    else:
        total_outputs = int(text_outputs)
        selected_combinations = [tuple([name + "_group" for name in combo]) for combo in
                                 all_combinations[:total_outputs]]

    # Import Negatives
    negative_objects = {
        "bowl": "C:/Users/shane/Documents/Maya/2022/scripts/objects/negatives/bowl/bowl.obj",
        "coffee_cup": "C:/Users/shane/Documents/Maya/2022/scripts/objects/negatives/coffee_cup/coffee_cup.obj",
        "spice": "C:/Users/shane/Documents/Maya/2022/scripts/objects/negatives/spice/spice.obj"
    }

    # Import all negative objects once and move them to a relative position
    imported_negatives = []
    for obj_name, obj_path in negative_objects.items():
        if not cmds.objExists(obj_name):  # Check if the object isn't already imported
            namespace = obj_name + "_ns"
            if cmds.namespace(exists=namespace):
                counter = 1
                while cmds.namespace(exists=namespace + str(counter)):
                    counter += 1
                namespace = namespace + str(counter)

            # Import the object
            cmds.file(obj_path, i=True, namespace=namespace, mergeNamespacesOnClash=True)

            # Get all objects in the new namespace
            new_objects = cmds.ls(namespace + ":*", transforms=True)

            # Set the group name based on the object name
            group_name = obj_name + "_group"
            if cmds.objExists(group_name):
                counter = 1
                while cmds.objExists(group_name + str(counter)):
                    counter += 1
                group_name = group_name + str(counter)

            # Group the new objects
            group = cmds.group(new_objects, name=group_name)

            # Move the object to a relative position
            cmds.move(100, 100, 100, group_name, relative=True)

            # Save the name for later use
            imported_negatives.append(group_name)

            # Adjust shaders for the coffee_cup object during import
            if group_name == "coffee_cup_group":
                namespace_for_coffee_cup = "coffee_cup_ns"  # Adjust this if the namespace is dynamic
                shaders = [namespace_for_coffee_cup + ":blinn1SG1", namespace_for_coffee_cup + ":blinn2SG1"]
                reflectivity_value = 0.0577778
                specular_color = [0.0533333, 0.0533333, 0.0533333]

                for shader in shaders:
                    # Adjust reflectivity
                    reflectivity_attr = shader + ".reflectivity"
                    if cmds.objExists(reflectivity_attr):
                        cmds.setAttr(reflectivity_attr, reflectivity_value)

                    # Adjust specular color
                    specular_attr = shader + ".specularColor"
                    if cmds.objExists(specular_attr):
                        cmds.setAttr(specular_attr, specular_color[0], specular_color[1], specular_color[2],
                                     type="double3")

    for combo in selected_combinations:
            print(combo)
            combo_len = len(combo)
            #Selects objects and runs the other functions
            transforms = []
            rotations = []
            random.seed(datetime.now().timestamp())
            for rot in range(combo_len):
                rotations.append(random.randint(0, 360))

            #this places the objects, saves the transforms, and deletes the original placement
            for i in range(combo_len):
                cmds.select(clear=True)
                cmds.select(combo[i])
                cmds.select( 'pPlane2.f[0:99]', add=True )
                transforms.append(runMEL())
                cmds.select(clear=True)
                cmds.select(str(combo[i]) + '1')
                cmds.rotate(0, rotations[i], 0, os=True, r=True)
                cmds.select(clear=True)
                renderJPEG(str(combo[i]))
                renderPNG(str(combo[i]))
                cmds.delete(str(combo[i]) + '1')

            #this places all the objects back where they were
            for m in range(len(transforms)):
                cmds.select(clear=True)
                temp = transforms[m]
                temp2 = ','.join(str(x) for x in temp)
                temp2 = temp2.replace('[', '').replace(']', '').replace(' ', '')
                cmds.select(combo[m])
                cmds.select('pPlane2.f[0:99]', add=True)
                mel.eval('source "C:/Users/shane/Documents/Maya/2022/scripts/addMEL.mel"')
                mel.eval('string $tmp = python("temp2")')
                mel.eval('addMEL($tmp)')
                cmds.select(clear=True)
                cmds.select(combo[m] + '1')
                cmds.rotate(0, rotations[m], 0, os=True, r=True)

            # cmds.select( 'group*' )
            # allGroup = cmds.ls(selection=True)
            # allGroup = list(filter(lambda k: 'Id' not in k, allGroup))
            add_negative = random.random() < 0.25
            print("add_negative", add_negative)
            negative_obj = None
            if add_negative:
                negative_obj = add_negative_object(imported_negatives)
                print("adding random negative")

            # Render the final scene
            file_name = "image" + str(uuid.uuid4())
            renderJPEG(file_name)

            # Remove the negative object if it was added
            if add_negative and cmds.objExists(negative_obj):
                # print("should be deleting")
                print(negative_obj)
                cmds.delete(negative_obj)

            print("Objects loaded. Building bounding boxes.")

            bounding_boxes.detect(file_name)
            print("Process complete. Resetting.")
            for j in range(combo_len):
                cmds.delete(str(combo[j]) + '1')

            #OBJECT RENDER OPTIMIZATION
            # 1. Delete history for the combination
            cmds.delete(all=True, constructionHistory=True)

            # 2. Clear selection for the combination
            cmds.select(clear=True)

            # 3. Force garbage collection for the combination
            gc.collect()
            # quit()
    #SCENE OPTIMIZATION
    # 1. Delete history for the scene
    # cmds.delete(all=True, constructionHistory=True)
    # # 2. Optimize scene size
    # cmds.file(optimize=True)
    #
    # # 3. Limit undo queue for the scene
    # cmds.undoInfo(stateWithoutFlush=False)
    # 4. Force garbage collection for the scene
    # gc.collect()


# Set the path to the folder containing the files
files_folder = 'C:/Users/shane/Documents/Maya/2022/scripts/dataset/raw/'
# Find all the files in the folder
all_files = glob.glob(files_folder + '*.*')

# Count the total number of files
total_files = len(all_files) / 2

print("Dataset generation complete.")
# Set the path to the folder containing the files
files_folder = 'C:/Users/shane/Documents/Maya/2022/scripts/dataset/raw/'
# Find all the .txt files in the folder
txt_files = glob.glob(files_folder + '*.txt')

# Count the total number of files
total_files = len(txt_files)

print("Total number of image pairs generated:", total_files)

# Set the split ratios
train_ratio = 0.80
valid_ratio = 0.20
num_train_files = int(total_files * train_ratio)

# Randomly shuffle the txt_files list
random.shuffle(txt_files)

# Move the file pairs
for i, txt_file in enumerate(txt_files):
    # Get the corresponding jpg file name
    jpg_file = txt_file.replace('.txt', '.jpg')

    # Choose the target folder based on the index and split ratios
    if i < num_train_files:
        target_folder = 'C:/Users/shane/Documents/Maya/2022/scripts/dataset/train/'
    else:
        target_folder = 'C:/Users/shane/Documents/Maya/2022/scripts/dataset/valid/'

    # Create the images and labels folders in the target folder if they don't exist
    images_folder = os.path.join(target_folder, 'images')
    labels_folder = os.path.join(target_folder, 'labels')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Move the txt and jpg files to the images and labels folders respectively
    shutil.move(txt_file, labels_folder + '/' + os.path.basename(txt_file))
    shutil.move(jpg_file, images_folder + '/' + os.path.basename(jpg_file))

end_time = time.time()
total_time = end_time - start_time
print("Total time taken: {:.2f} seconds".format(total_time))
