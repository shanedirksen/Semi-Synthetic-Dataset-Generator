from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageEnhance
import glob
import yaml
import os
import random
import numpy as np

#THIS IS FOR TESTING ONLY
def draw_bbox(image, boxes, width, height):
    draw = ImageDraw.Draw(image)
    for object_name, obj in boxes.items():
        left = obj.left / width
        top = obj.top / height
        right = obj.right / width
        bottom = obj.bottom / height

        draw.rectangle(((left*width, top*height), (right*width, bottom*height)), outline='red')
    return image

def random_rotate(image):
    """Randomly rotate the image within a specified range."""
    angle = random.uniform(-15, 15)  # Random angle between -15 and 15 degrees
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

def random_shear(image):
    """Randomly shear the image in the x-direction."""
    shear_factor = random.uniform(-0.3, 0.3)  # Random shear factor between -0.3 and 0.3
    width, height = image.size
    m = [1, shear_factor, -shear_factor * height / 2, 0, 1, 0]
    return image.transform((width, height), Image.AFFINE, m, resample=Image.BICUBIC)

def save_rotated_and_sheared_images(image, objects, base_path, base_name, width, height, yaml_config):
    """Save rotated and sheared variations of the image."""
    variations = {
        "_rotated1": random_rotate(image),
        "_rotated2": random_rotate(image),
        "_sheared1": random_shear(image),
        "_sheared2": random_shear(image)
    }

    for suffix, var_image in variations.items():
        var_image.save(base_path + base_name + suffix + ".jpg")
        with open(base_path + base_name + suffix + '.txt', 'w') as f:
            for object_name, obj in objects.items():
                width_image = (obj.right - obj.left) / width
                height_image = (obj.bottom - obj.top) / height
                center_x = ((obj.right + obj.left) / 2) / width
                center_y = ((obj.bottom + obj.top) / 2) / height
                object_index = yaml_config['names'].index(object_name)
                f.write(f"{object_index} {center_x} {center_y} {width_image} {height_image} \n")

def add_gaussian_noise(image, sigma=1.0):
    np_image = np.array(image)
    noise = np.random.normal(0, sigma, np_image.shape).astype(np.uint8)
    noisy_image = np_image + noise
    return Image.fromarray(np.clip(noisy_image, 0, 255))

def apply_gaussian_blur(image, radius=1):
    return image.filter(ImageFilter.GaussianBlur(radius))

def save_image_variations(image, objects, base_path, base_name, width, height, yaml_config):
    variations = {
        "": image,
        "_noisy1": add_gaussian_noise(image, sigma=10),
        "_noisy2": add_gaussian_noise(image, sigma=50),
        "_blurred": apply_gaussian_blur(image, radius=2)
    }

    for suffix, var_image in variations.items():
        var_image.save(base_path + base_name + suffix + ".jpg")
        with open(base_path + base_name + suffix + '.txt', 'w') as f:
            for object_name, obj in objects.items():
                width_image = (obj.right - obj.left) / width
                height_image = (obj.bottom - obj.top) / height
                center_x = ((obj.right + obj.left) / 2) / width
                center_y = ((obj.bottom + obj.top) / 2) / height
                object_index = yaml_config['names'].index(object_name)
                f.write(f"{object_index} {center_x} {center_y} {width_image} {height_image} \n")



def crop_image_and_boxes(raw_folder_path, orig_image, objects, width, height, arg1, n):
    crop_percentage = random.uniform(0.3, 0.6)  # Random crop size between 30% to 60%
    crop_width = int(crop_percentage * width)
    crop_height = int(crop_percentage * height)

    left_crop = random.randint(0, width - crop_width)  # Randomly choose the top-left corner of the crop
    top_crop = random.randint(0, height - crop_height)

    # Cropping the original image
    cropped_image = orig_image.crop((left_crop, top_crop, left_crop + crop_width, top_crop + crop_height))
    cropped_objects = {}

    # Adjusting the bounding boxes for the cropped image
    for object_name, obj in objects.items():
        # Calculate the intersection area
        intersect_left = max(left_crop, obj.left)
        intersect_top = max(top_crop, obj.top)
        intersect_right = min(left_crop + crop_width, obj.right)
        intersect_bottom = min(top_crop + crop_height, obj.bottom)

        if intersect_right > intersect_left and intersect_bottom > intersect_top:
            intersection_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
            bounding_box_area = (obj.right - obj.left) * (obj.bottom - obj.top)

            # Check if the intersection area is at least 25% of the bounding box area
            if intersection_area >= 0.25 * bounding_box_area:
                # Calculate new bounding box coordinates relative to cropped image
                left = max(0, obj.left - left_crop)
                top = max(0, obj.top - top_crop)
                right = min(crop_width, obj.right - left_crop)
                bottom = min(crop_height, obj.bottom - top_crop)

                # Create new square and add to the dictionary
                cropped_objects[object_name] = Square(left, top, right, bottom)

    # Save the cropped image
    cropped_filename = f"{arg1}_crop_{n}.jpg"
    cropped_image.save(raw_folder_path + cropped_filename)

    if not cropped_objects:  # If no bounding box is present in the cropped image
        return None, None, None, None, None

    return cropped_objects, cropped_image, crop_width, crop_height, cropped_filename



class Square:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def area(self):
        return (self.right - self.left) * (self.bottom - self.top)


def find_edge_pixels(alpha_image, threshold):
    left, right, top, bottom = None, None, None, None

    # Find left edge
    for x in range(alpha_image.width):
        for y in range(alpha_image.height):
            if alpha_image.getpixel((x, y)) > threshold:
                left = x
                break
        if left is not None:
            break

    # Find right edge
    for x in reversed(range(alpha_image.width)):
        for y in range(alpha_image.height):
            if alpha_image.getpixel((x, y)) > threshold:
                right = x
                break
        if right is not None:
            break

    # Find top edge
    for y in range(alpha_image.height):
        for x in range(alpha_image.width):
            if alpha_image.getpixel((x, y)) > threshold:
                top = y
                break
        if top is not None:
            break

    # Find bottom edge
    for y in reversed(range(alpha_image.height)):
        for x in range(alpha_image.width):
            if alpha_image.getpixel((x, y)) > threshold:
                bottom = y
                break
        if bottom is not None:
            break

    return left, top, right, bottom


def remove_inner_boxes(obj_dict):
    filtered_obj_dict = obj_dict.copy()
    for key1, box1 in obj_dict.items():
        is_inner = False
        for key2, box2 in obj_dict.items():
            if (box1.left > box2.left and box1.right < box2.right and
                    box1.top > box2.top and box1.bottom < box2.bottom):
                is_inner = True
                break
        if is_inner:
            del filtered_obj_dict[key1]
    return filtered_obj_dict


def detect(arg1, draw_bboxes=False):
    imdir = 'C:/Users/shane/Documents/Maya/2022/scripts/objectDetection/'
    filename = imdir + arg1 + '.jpg'

    # Load YAML file
    yaml_file_path = "C:/Users/shane/Documents/Maya/2022/scripts/dataset/data.yaml"
    with open(yaml_file_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    orig_image = Image.open(filename)
    # 10% chance to convert the image to black and white
    if random.randint(1, 100) <= 10:
        orig_image = ImageOps.grayscale(orig_image)

    width, height = orig_image.size
    threshold = 10

    print(f"Original image width: {width}, height: {height}")

    objects = {}
    ext = 'png'

    for image_file in glob.glob(imdir + '*.' + ext):
        image = Image.open(image_file).convert('RGBA')
        alpha_image = image.getchannel('A')

        # Check if there's at least one pixel above the threshold in the alpha channel
        if not any(alpha_image.getpixel((x, y)) > threshold for x in range(alpha_image.width) for y in range(alpha_image.height)):
            print(f"Warning: Skipping image {image_file} as it has no alpha values above the threshold")
            continue

        left, top, right, bottom = find_edge_pixels(alpha_image, threshold)

        # Extract object name from the PNG file
        object_name = os.path.splitext(os.path.basename(image_file))[0].replace('_group', '')

        # Store the object in the dictionary with the object_name as the key
        objects[object_name] = Square(left, top, right, bottom)

    objects = remove_inner_boxes(objects)

    # Check for 'raw' folder and create it if it doesn't exist
    raw_folder_path = 'C:/Users/shane/Documents/Maya/2022/scripts/dataset/raw/'
    if not os.path.exists(raw_folder_path):
        os.makedirs(raw_folder_path)

    # Write the original image and its annotations
    with open(raw_folder_path + arg1 + '.txt', 'w') as f:
        for object_name, obj in objects.items():
            width_image = (obj.right - obj.left) / width
            height_image = (obj.bottom - obj.top) / height
            center_x = ((obj.right + obj.left) / 2) / width
            center_y = ((obj.bottom + obj.top) / 2) / height

            # Find the index of the object name in the names list
            object_index = yaml_config['names'].index(object_name)

            # Write the bounding box to the file
            f.write(f"{object_index} {center_x} {center_y} {width_image} {height_image} \n")

    # Draw bounding boxes on the original image
    if draw_bboxes:
        orig_image = draw_bbox(orig_image, objects, width, height)

    # Save the original image
    orig_image.save(raw_folder_path + arg1 + ".jpg")
    save_rotated_and_sheared_images(orig_image, objects, raw_folder_path, arg1, width, height, yaml_config)

    # Save variations for the original image
    save_image_variations(orig_image, objects, raw_folder_path, arg1, width, height, yaml_config)

    # Flip the original image and its bounding boxes
    flipped_orig_image = ImageOps.mirror(orig_image)
    flipped_orig_objects = {k: Square(width - v.right, v.top, width - v.left, v.bottom) for k, v in objects.items()}
    # Save variations for the mirrored image
    save_image_variations(flipped_orig_image, flipped_orig_objects, raw_folder_path, arg1 + "_flipped", width, height, yaml_config)
    save_rotated_and_sheared_images(flipped_orig_image, flipped_orig_objects, raw_folder_path, arg1 + "_flipped", width,
                                    height, yaml_config)

    # Draw bounding boxes on the flipped image
    if draw_bboxes:
        flipped_orig_image = draw_bbox(flipped_orig_image, flipped_orig_objects, width, height)

    with open(raw_folder_path + arg1 + '_flipped.txt', 'w') as f:
        for object_name, obj in flipped_orig_objects.items():
            width_image = (obj.right - obj.left) / width
            height_image = (obj.bottom - obj.top) / height
            center_x = ((obj.right + obj.left) / 2) / width
            center_y = ((obj.bottom + obj.top) / 2) / height

            # Find the index of the object name in the names list
            object_index = yaml_config['names'].index(object_name)

            # Write the bounding box to the file
            f.write(f"{object_index} {center_x} {center_y} {width_image} {height_image} \n")

    # Save the flipped original image
    flipped_orig_image.save(raw_folder_path + arg1 + "_flipped.jpg")

    # Create 5 cropped images
    for i in range(5):
        attempts = 0
        while attempts < 100:
            cropped_objects, cropped_image, crop_width, crop_height, cropped_filename = crop_image_and_boxes(
                raw_folder_path, orig_image,
                objects,
                width,
                height,
                arg1, i)
            if cropped_objects:  # If a cropped image with bounding boxes was returned
                # Draw bounding boxes on the cropped image
                if draw_bboxes:
                    cropped_image = draw_bbox(cropped_image, cropped_objects, crop_width, crop_height)

                # Write annotation file for the cropped image
                with open(raw_folder_path + cropped_filename.replace('.jpg', '.txt'), 'w') as f:
                    for object_name, obj in cropped_objects.items():
                        width_image = (obj.right - obj.left) / crop_width
                        height_image = (obj.bottom - obj.top) / crop_height
                        center_x = ((obj.right + obj.left) / 2) / crop_width
                        center_y = ((obj.bottom + obj.top) / 2) / crop_height

                        # Find the index of the object name in the names list
                        object_index = yaml_config['names'].index(object_name)

                        # Write the bounding box to the file
                        f.write(f"{object_index} {center_x} {center_y} {width_image} {height_image} \n")

                # Save the cropped image
                cropped_image.save(raw_folder_path + cropped_filename)
                # Save variations for the cropped image
                save_image_variations(cropped_image, cropped_objects, raw_folder_path,
                                      cropped_filename.replace('.jpg', ''), crop_width, crop_height, yaml_config)

                break
            attempts += 1
            if not cropped_objects:
                print("We tried a hundred times brutha")
                continue

        # If no cropped image with bounding boxes was found after 100 attempts, skip the current iteration
        if not cropped_objects:
            continue

        # Flip the cropped image and its bounding boxes
        flipped_image = ImageOps.mirror(cropped_image)
        flipped_objects = {k: Square(crop_width - v.right, v.top, crop_width - v.left, v.bottom) for k, v in
                           cropped_objects.items()}

        # Draw bounding boxes on the flipped image
        if draw_bboxes:
            flipped_image = draw_bbox(flipped_image, flipped_objects, crop_width, crop_height)

        with open(raw_folder_path + cropped_filename.replace('.jpg', '_flipped.txt'), 'w') as f:
            for object_name, obj in flipped_objects.items():
                width_image = (obj.right - obj.left) / crop_width
                height_image = (obj.bottom - obj.top) / crop_height
                center_x = ((obj.right + obj.left) / 2) / crop_width
                center_y = ((obj.bottom + obj.top) / 2) / crop_height

                # Find the index of the object name in the names list
                object_index = yaml_config['names'].index(object_name)

                # Write the bounding box to the file
                f.write(f"{object_index} {center_x} {center_y} {width_image} {height_image} \n")

        # Save the flipped image
        flipped_image.save(raw_folder_path + cropped_filename.replace('.jpg', '_flipped.jpg'))
        # Save variations for the flipped cropped image
        save_image_variations(flipped_image, flipped_objects, raw_folder_path,
                              cropped_filename.replace('.jpg', '_flipped'), crop_width, crop_height, yaml_config)

    # Delete all PNG and JPG files in the objectDetection folder
    for image_file in glob.glob('C:/Users/shane/Documents/Maya/2022/scripts/objectDetection/*.*'):
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            os.remove(image_file)


def test():
    detect("bottles", 9)