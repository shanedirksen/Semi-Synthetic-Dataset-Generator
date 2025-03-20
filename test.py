import os
import maya.cmds as cmds

# define the directory path
dir_path = r'C:\Users\shane\Documents\Maya\2022\scripts\objects\new_objects'

# iterate over subdirectories
for subdirectory in os.listdir(dir_path):
    subdirectory_path = os.path.join(dir_path, subdirectory)

    # check if it's a directory and not a file
    if os.path.isdir(subdirectory_path):
        for file in os.listdir(subdirectory_path):
            # check if it's an .fbx file
            if file.endswith('.fbx'):
                file_path = os.path.join(subdirectory_path, file)

                # import the .fbx file to Maya
                cmds.file(file_path, i=True)
