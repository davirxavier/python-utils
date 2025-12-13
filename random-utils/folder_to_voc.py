import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image


def create_voc_xml(image_path, label, out_dir):
    """Creates a simplified VOC XML annotation file for the image (with just labels)."""
    # Get image details
    img = Image.open(image_path)
    img_width, img_height = img.size
    img_filename = os.path.basename(image_path)
    img_name = os.path.splitext(img_filename)[0]

    # Create the XML structure
    annotation = ET.Element("annotation")

    # Folder element
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(out_dir)

    # Filename element
    filename = ET.SubElement(annotation, "filename")
    filename.text = img_filename

    # Size element (width and height)
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_width)
    height = ET.SubElement(size, "height")
    height.text = str(img_height)

    # Object element (label only, no bounding boxes)
    obj = ET.SubElement(annotation, "object")

    # Name of the object (label)
    name = ET.SubElement(obj, "name")
    name.text = label

    # Write the XML to file
    xml_path = os.path.join(out_dir, f"{img_name}.xml")
    tree = ET.ElementTree(annotation)
    tree.write(xml_path)


def get_unique_image_name(image_path, out_dir):
    """Generate a unique filename if the image already exists in the output directory."""
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    counter = 1
    unique_name = base_name

    # Check if the file already exists and find a unique name
    while os.path.exists(os.path.join(out_dir, unique_name)):
        unique_name = f"{name}_{counter}{ext}"
        counter += 1

    return unique_name


def scan_and_copy_images(root_dir, out_dir):
    """Scans the root directory for images, copies them to out_dir, and generates simplified VOC XMLs."""
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Walk through the root directory recursively
    for label_folder in os.listdir(root_dir):
        label_folder_path = os.path.join(root_dir, label_folder)
        if label_folder_path == out_dir:
            continue

        # Only proceed if it's a directory
        if os.path.isdir(label_folder_path):
            for image_name in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_name)

                # If it's an image file (based on extension)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Determine the output path and ensure unique filename
                    out_image_name = get_unique_image_name(image_path, out_dir)
                    out_image_path = os.path.join(out_dir, out_image_name)

                    # Copy the image to the output folder with a unique name
                    shutil.copy(image_path, out_image_path)

                    # Generate simplified VOC XML annotation
                    create_voc_xml(out_image_path, label_folder, out_dir)
                    print(f"Processed: {out_image_name} -> {label_folder}")


if __name__ == "__main__":
    root_dir = os.getcwd()  # Current working directory
    out_dir = os.path.join(root_dir, 'out')  # Output directory

    scan_and_copy_images(root_dir, out_dir)
    print(f"Processing complete. Images and XMLs saved to {out_dir}")
