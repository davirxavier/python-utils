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
    Background-only images (no XML or empty XML) are also included.
    """

    input_dir = Path(root_dir) / "input"
    if not input_dir.exists():
        print(f"Error: 'input' folder not found in {root_dir}.")
        return

    train_dir = Path(root_dir) / "training"
    test_dir = Path(root_dir) / "testing"

    if train_dir.exists() and test_dir.exists():
        print("Training and Testing directories already exist. No changes made.")
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    image_files = []

    for image_file in input_dir.rglob("*"):
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            image_files.append(image_file)

    data_by_class = defaultdict(list)
    background_items = []

    for image_file in image_files:
        annotation_file = image_file.with_suffix(".xml")

        if not annotation_file.exists():
            background_items.append((image_file, None))
            continue

        labels = parse_voc_xml(annotation_file)

        if len(labels) == 0:
            background_items.append((image_file, annotation_file))
            continue

        for label in labels:
            data_by_class[label].append((image_file, annotation_file))

    train_data = []
    test_data = []

    for label, items in data_by_class.items():
        random.shuffle(items)
        split_index = int(len(items) * (1 - test_size))
        train_data.extend(items[:split_index])
        test_data.extend(items[split_index:])

    # Split background images (not stratified)
    random.shuffle(background_items)
    split_index = int(len(background_items) * (1 - test_size))
    train_data.extend(background_items[:split_index])
    test_data.extend(background_items[split_index:])

    random.shuffle(train_data)
    random.shuffle(test_data)

    def copy_files(file_list, target_dir):
        for image_file, annotation_file in file_list:
            shutil.copy(str(image_file), target_dir / image_file.name)
            if annotation_file is not None:
                shutil.copy(str(annotation_file), target_dir / annotation_file.name)

    copy_files(train_data, train_dir)
    copy_files(test_data, test_dir)

    print(
        f"Data split completed. "
        f"{len(train_data)} images for training, "
        f"{len(test_data)} images for testing."
    )


# Example usage:
# if __name__ == "__main__":
#     root_dir = "/home/xav/Pictures/sofa/custom_model/classification_data"
#     path = Path(root_dir)
#     resize.resize_images(str(path / 'input_resize'), str(path / 'input'), 160, 'fit-shortest')
#     split_data(root_dir, 0.2)
