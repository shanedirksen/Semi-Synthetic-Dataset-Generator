import cv2
import xml.etree.ElementTree as ET
import os
import yaml
import glob
import random
from pathlib import Path

def map_object_name(name):
    if name.startswith('coffee'):
        return 'coffee'
    elif name.startswith('palmolive'):
        return 'palmoliveSoap'
    else:
        return name


# Load the object names from the YAML file
with open("C:/Users/shane/Documents/Maya/2022/scripts/dataset/data.yaml") as f:
    data = yaml.safe_load(f)
    object_names = data['names']

# Base directory for the dataset
base_dir = "C:/Users/shane/Documents/Datasets/gmu-kitchens/"
dataset_dir = os.path.join(base_dir, 'dataset')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Create images and labels directories
Path(images_dir).mkdir(parents=True, exist_ok=True)
Path(labels_dir).mkdir(parents=True, exist_ok=True)

# Initialize set to collect objects not in YAML
not_in_yaml = set()

# Initialize a dictionary to store the IDs and names of the processed objects
processed_objects = {}

# Process each scene
for scene_num in range(1, 10):
    scene_dir = os.path.join(base_dir, f"gmu_scene_00{scene_num}")

    # Get the list of image files in this scene
    image_files = glob.glob(os.path.join(scene_dir, 'Images', '*.png'))

    # Select a hundred random image files from the list
    image_files_sample = random.sample(image_files, 100)

    # Process each image and annotation in the sample
    for image_file in image_files_sample:
        # Extract the image number from the filename
        image_num = os.path.basename(image_file)[:-4]  # Remove the '.png' extension

        image_path = os.path.join(scene_dir, 'Images', f"{image_num}.png")
        xml_path = os.path.join(scene_dir, 'Annotations', f"{image_num}.xml")

        image = cv2.imread(image_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        height, width = image.shape[:2]

        # New text file for labels
        labels_file = open(os.path.join(labels_dir, f"scene{scene_num}_{image_num}.txt"), 'w')

        # Iterate over each object in the XML file
        for obj in root.iter('object'):
            original_name = obj.find('name').text
            name = map_object_name(original_name)

            # Skip this object if it's not in the YAML file
            if name not in object_names:
                not_in_yaml.add(original_name)
                continue

            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Normalize the bounding box coordinates
            width_image = (xmax - xmin) / width
            height_image = (ymax - ymin) / height
            center_x = ((xmax + xmin) / 2) / width
            center_y = ((ymax + ymin) / 2) / height

            # Find the index of the object name in the names list
            object_index = object_names.index(name)

            # Add the object to the dictionary of processed objects
            processed_objects[name] = object_index

            # Write to the labels file
            labels_file.write(f"{object_index} {center_x} {center_y} {width_image} {height_image}\n")

        labels_file.close()

        # Save the image in the new directory with a new name
        cv2.imwrite(os.path.join(images_dir, f"scene{scene_num}_{image_num}.png"), image)

# Print the ID and name of each unique processed object, sorted by ID
for id, name in sorted(processed_objects.items(), key=lambda item: item[1]):
    print(f"ID: {id}, Name: {name}")

# Print objects not in YAML
print("Objects not included in YAML:", not_in_yaml)
