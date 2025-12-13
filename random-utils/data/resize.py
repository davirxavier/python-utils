import os
import shutil
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path

def adjust_bounding_boxes(xml_file, original_size, target_size, resize_mode):
    print(f"Adjusting bounding boxes for {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    original_width, original_height = original_size
    target_width, target_height = target_size

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        if not bndbox:
            continue

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        if resize_mode == 'fit-longest':
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            crop_left = (new_width - target_width) / 2
            crop_top = (new_height - target_height) / 2
            crop_right = crop_left + target_width
            crop_bottom = crop_top + target_height

            xmin = (xmin * scale) - crop_left
            ymin = (ymin * scale) - crop_top
            xmax = (xmax * scale) - crop_left
            ymax = (ymax * scale) - crop_top

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(target_width, xmax)
            ymax = min(target_height, ymax)

        elif resize_mode == 'fit-shortest':
            scale = max(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            padding_left = (target_width - new_width) // 2
            padding_top = (target_height - new_height) // 2

            xmin = xmin * scale + padding_left
            ymin = ymin * scale + padding_top
            xmax = xmax * scale + padding_left
            ymax = ymax * scale + padding_top

            xmax = min(xmax, target_width)
            ymax = min(ymax, target_height)

        elif resize_mode == 'squash':
            scale_width = target_width / original_width
            scale_height = target_height / original_height
            xmin = xmin * scale_width
            ymin = ymin * scale_height
            xmax = xmax * scale_width
            ymax = ymax * scale_height

        else:
            raise ValueError("Invalid resize_mode. Use 'fit-shortest', 'fit-longest', or 'squash'.")

        bndbox.find('xmin').text = str(round(xmin))
        bndbox.find('ymin').text = str(round(ymin))
        bndbox.find('xmax').text = str(round(xmax))
        bndbox.find('ymax').text = str(round(ymax))

    size = root.find('size')
    if size is not None:
        width_elem = size.find('width')
        height_elem = size.find('height')
        if width_elem is not None:
            width_elem.text = str(target_width)
        if height_elem is not None:
            height_elem.text = str(target_height)

    return tree


def resize_image(image_path, target_size, resize_mode):
    print(f"Resizing image {image_path} to {target_size} using {resize_mode} mode.")
    img = Image.open(image_path)
    original_size = img.size
    width, height = original_size

    target_width, target_height = target_size

    if resize_mode == 'fit-longest':
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        left = (new_width - target_width) / 2
        top = (new_height - target_height) / 2
        right = (new_width + target_width) / 2
        bottom = (new_height + target_height) / 2

        img = img.crop((left, top, right, bottom))

    elif resize_mode == 'fit-shortest':
        scale = max(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        new_img.paste(img, (left, top))
        img = new_img

    elif resize_mode == 'squash':
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    else:
        raise ValueError("Invalid resize_mode. Use 'fit-shortest', 'fit-longest', or 'squash'.")

    print(f"Resized image size: {img.size}")
    return img, original_size


def process_image(image_path, output_image_path, output_xml_path, target_size, resize_mode):
    print(f"Processing image {image_path} -> Output: {output_image_path} -> XML: {output_xml_path}")
    img, original_size = resize_image(image_path, target_size, resize_mode)
    img.save(output_image_path, 'JPEG', quality=90, optimize=True)

    xml_file = image_path.with_suffix('.xml')
    if xml_file.exists():
        print(f"Found XML: {xml_file}, adjusting bounding boxes.")
        adjusted_xml = adjust_bounding_boxes(xml_file, original_size, target_size, resize_mode)
        adjusted_xml.write(output_xml_path)
    else:
        print(f"No XML file found for {image_path.stem}, skipping bounding box adjustment.")


def process_folder(folder, output_folder, target_size, resize_mode):
    print(f"Processing folder: {folder} -> Output folder: {output_folder}")

    # Walk through the folder structure
    for root, dirs, files in os.walk(folder):
        # Convert the root to an absolute path to avoid path resolution issues
        root_path = Path(root).resolve()
        output_folder_path = Path(output_folder).resolve()

        # Skip if the current directory is exactly the output folder
        if root_path == output_folder_path:
            print(f"Skipping the output folder itself: {root}")
            continue

        # Normal processing for directories outside the output folder
        relative_root = os.path.relpath(root, folder)
        output_subfolder = Path(output_folder) / relative_root
        print(f"Creating output folder: {output_subfolder}")
        output_subfolder.mkdir(parents=True, exist_ok=True)

        # Process each file in the directory
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = Path(root) / file
                output_image_path = output_subfolder / (image_path.stem + '.jpg')
                output_xml_path = output_subfolder / (image_path.stem + '.xml')

                print(f"Found image file: {file} -> {image_path}")
                print(f"Output image path: {output_image_path}")
                print(f"Output XML path: {output_xml_path}")

                if output_image_path.exists():
                    print(f"Skipping {file}, already processed.")
                    continue

                print(f"Looking for XML: {output_xml_path}")
                if not output_xml_path.exists():
                    print(f"Warning: No XML file found for {image_path.stem}. Skipping bounding box adjustment.")

                try:
                    process_image(image_path, output_image_path, output_xml_path, target_size, resize_mode)
                except Exception as e:
                    print(f"Error processing {file}: {e}")


def resize_images(folder, output_folder, target_size, resize_mode):
    target_size = (target_size, target_size)
    print(f"Starting resize for {folder} -> {output_folder} with target size {target_size} and mode {resize_mode}")
    process_folder(folder, output_folder, target_size, resize_mode)


# folder = '/home/xav/Pictures/sofa/custom_model/data'
# output_folder = '/home/xav/Pictures/sofa/custom_model/data/processed'
# target_size = 160
# resize_mode = 'squash'
#
# resize_images(folder, output_folder, target_size, resize_mode)
