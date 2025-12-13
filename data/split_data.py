import os
import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_voc_xml(xml_file: Path):
    """
    Parses the VOC XML file to extract the class labels.

    Args:
        xml_file (Path): Path to the VOC XML file.

    Returns:
        list: A list of class labels present in the image.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    labels = []
    for object_elem in root.findall("object"):
        label = object_elem.find("name").text
        labels.append(label)

    return labels


def split_data(root_dir: str, test_size: float = 0.2):
    """
    Scans the 'input' folder in the provided root directory, finds all image and VOC annotation files,
    and splits them into training and testing folders, ensuring an even class distribution.

    Args:
        root_dir (str): The root directory where the 'input' folder is located.
        test_size (float): The fraction of data to use for testing (default is 0.2, or 20%).
    """

    input_dir = Path(root_dir) / "input"
    if not input_dir.exists():
        print(f"Error: 'input' folder not found in {root_dir}.")
        return

    # Paths for training and testing directories
    train_dir = Path(root_dir) / "training"
    test_dir = Path(root_dir) / "testing"

    # If the training or testing directories already exist, do nothing
    if train_dir.exists() and test_dir.exists():
        print("Training and Testing directories already exist. No changes made.")
        return

    # Create training and testing directories if they do not exist
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files and corresponding annotation files
    image_files = []
    annotation_files = []

    for image_file in input_dir.rglob("*"):
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            # Find corresponding annotation file (assuming .xml files for VOC annotations)
            annotation_file = image_file.with_suffix(".xml")
            if annotation_file.exists():
                image_files.append(image_file)
                annotation_files.append(annotation_file)

    # Group files by the labels in the annotation XML files
    data_by_class = defaultdict(list)

    for image_file, annotation_file in zip(image_files, annotation_files):
        labels = parse_voc_xml(annotation_file)
        for label in labels:
            data_by_class[label].append((image_file, annotation_file))

    # Split data by class (to ensure even distribution)
    train_data = []
    test_data = []

    for label, items in data_by_class.items():
        # Shuffle data for this class
        random.shuffle(items)

        # Split data for this class
        split_index = int(len(items) * (1 - test_size))
        train_data.extend(items[:split_index])
        test_data.extend(items[split_index:])

    # Shuffle the final datasets
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Copy files to the respective folders
    def copy_files(file_list, target_dir):
        for image_file, annotation_file in file_list:
            # Copy image files
            shutil.copy(str(image_file), target_dir / image_file.name)
            # Copy corresponding annotation files
            shutil.copy(str(annotation_file), target_dir / annotation_file.name)

    copy_files(train_data, train_dir)
    copy_files(test_data, test_dir)

    print(f"Data split completed. {len(train_data)} images for training, {len(test_data)} images for testing.")

# Example usage:
# if __name__ == "__main__":
#     root_dir = "/home/xav/Pictures/sofa/custom_model/classification_data"
#     path = Path(root_dir)
#     resize.resize_images(str(path / 'input_resize'), str(path / 'input'), 160, 'fit-shortest')
#     split_data(root_dir, 0.2)
