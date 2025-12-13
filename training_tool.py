# ---------------------------------------------------------------------------------------------------------------------------
# TRAINING TOOL
# ---------------------------------------------------------------------------------------------------------------------------
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import tensorflow as tf

from data import resize
from data.split_data import split_data


# -------------------------------
# Data preparation
# -------------------------------

def parse_voc_xml(xml_path):
    """
    Parse a VOC annotation file into category labels for image classification.
    Handles cases where there are no bounding boxes in the annotation.
    """
    bboxes = []  # Bounding boxes will remain empty in case of no boxes
    labels = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        # Extract the label (class name) for the object
        label = obj.find("name").text

        # Check if there is a bounding box
        bbox = obj.find("bndbox")
        if bbox is not None:
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            bboxes.append((xmin, ymin, xmax, ymax))

        labels.append(label)

    return bboxes, labels


def collect_data(root_dir):
    """
    Recursively scan 'training' and 'testing' folders inside root_dir for images
    and corresponding VOC XML annotation files. Converts bounding boxes to
    [x_min, y_min, x_max, y_max].

    Returns two lists:
    - training_data: A list of tuples (image_path, (bboxes, labels))
    - testing_data: A list of tuples (image_path, (bboxes, labels)) or an empty list if no testing folder is found
    """
    training_data = []
    testing_data = []

    valid_ext = {".jpg", ".jpeg", ".png"}

    # Define training and testing directories
    training_dir = Path(root_dir) / "training"
    testing_dir = Path(root_dir) / "testing"

    # Function to collect data from a given directory
    def collect_from_directory(data_list, folder_path):
        for dirpath, _, filenames in os.walk(folder_path):
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext not in valid_ext:
                    continue

                img_path = Path(dirpath) / fn
                xml_path = Path(dirpath) / (Path(fn).stem + ".xml")

                # If no annotation, treat as empty bbox image
                if xml_path.exists():
                    ann = parse_voc_xml(xml_path)
                else:
                    print(f"Annotation missing for {img_path}, using empty boxes.")
                    ann = ([], [])

                data_list.append({"img_path": str(img_path), "annotations": ann})

    # Collect training data
    if training_dir.exists():
        collect_from_directory(training_data, training_dir)
    else:
        print(f"Training directory {training_dir} not found.")

    if testing_dir.exists():
        collect_from_directory(testing_data, testing_dir)
    else:
        print(f"Testing directory {testing_dir} not found, returning empty testing data.")

    return training_data, testing_data


def split_training_data(data, split_ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]


def validate_dataset(data, expected_h, expected_w, is_object_detection=True):
    """
    Validate dataset images and bounding boxes or labels based on the dataset type.

    Checks:
      - For image classification datasets:
          - image shape
          - number of channels
          - decode failures
      - For object detection datasets:
          - image shape
          - number of channels
          - decode failures
          - invalid bbox geometry
          - bbox out of bounds
    """

    errors = {
        "bad_shape": [],
        "bad_channels": [],
        "bbox_oob": [],
        "bbox_invalid": [],
        "decode_fail": [],
    }

    for idx, item in enumerate(data):
        img_path = item["img_path"]
        annotations = item["annotations"]

        # Try decoding the image
        img = safe_decode_image(img_path)
        if img is None:
            errors["decode_fail"].append(img_path)
            print(f"[DECODE FAIL] {img_path}")
            continue

        img = tf.convert_to_tensor(img, dtype=tf.float32)
        h, w, c = img.shape

        # Validate image shape
        if h != expected_h or w != expected_w:
            errors["bad_shape"].append((img_path, (h, w)))
            print(f"[BAD SHAPE] {img_path} -> ({h}, {w}) expected ({expected_h}, {expected_w})")

        # # Validate number of channels
        # if c != expected_channels:
        #     errors["bad_channels"].append((img_path, c))
        #     print(f"[BAD CHANNELS] {img_path} -> {c} channels, expected {expected_channels}")

        if is_object_detection:
            bboxes_list = annotations[0] or []
            bboxes = tf.convert_to_tensor(bboxes_list, dtype=tf.float32)

            # Validate bounding boxes
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = box.numpy()

                # Invalid geometry: x2 <= x1 or y2 <= y1
                if x2 <= x1 or y2 <= y1:
                    errors["bbox_invalid"].append((img_path, i, box.numpy()))
                    print(f"[INVALID BBOX] {img_path} idx {i}: {box.numpy()} (x2<=x1 or y2<=y1)")

                # Bounding box out of bounds
                if (
                        x1 < 0 or y1 < 0 or
                        x2 > w or y2 > h
                ):
                    errors["bbox_oob"].append((img_path, i, box.numpy(), (h, w)))
                    print(f"[BBOX OOB] {img_path} idx {i}: {box.numpy()} exceeds image bounds ({h}, {w})")

    print("\nValidation Complete.")
    print("Bad shape images:", len(errors["bad_shape"]))
    print("Bad channel images:", len(errors["bad_channels"]))
    print("Bboxes OOB:", len(errors["bbox_oob"]))
    print("Invalid bboxes:", len(errors["bbox_invalid"]))
    print("Decode failures:", len(errors["decode_fail"]))

    return errors


def has_errors(errors):
    return any(len(v) > 0 for v in errors.values())


def augment_sample(img, data, is_object_detection=False, is_grayscale=False):
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.9, 1.15)

    if not is_grayscale:
        img = tf.image.random_saturation(img, 0.9, 1.15)

    if not is_object_detection:

        # ----- Random horizontal flip -----
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)

        # ----- Random zoom -----
        if tf.random.uniform(()) > 0.5:
            scale = tf.random.uniform([], 0.9, 1.2)

            # dynamic shape safe extraction
            shape = tf.shape(img)
            h, w = shape[0], shape[1]

            new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

            resized = tf.image.resize(img, (new_h, new_w))
            img = tf.image.resize_with_crop_or_pad(resized, h, w)

    return img, data


def safe_decode_image(img_path, grayscale=False):
    """Read and decode the image file only if it's valid."""
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3, try_recover_truncated=True)
        img = tf.image.convert_image_dtype(img, tf.float32)

        if grayscale:
            img = tf.image.rgb_to_grayscale(img)

        return img
    except Exception as e:
        raise Exception(f"Unexpected error for image {img_path}: {e}")


def create_tf_dataset(data, input_shape, label_map, is_object_detection, augmentation=True):
    target_w, target_h, target_c = input_shape
    grayscale = target_c == 1

    errors = validate_dataset(data, target_w, target_h, is_object_detection)
    if has_errors(errors):
        raise ValueError(f"Dataset validation failed.\nSummary: {errors}")

    img_paths = [item["img_path"] for item in data]
    annotations = [item["annotations"] for item in data]
    num_classes = len(label_map)

    annotations_tf = []
    for ann in annotations:
        bboxes = ann[0]
        label_names = ann[1]
        labels = [label_map[x] for x in label_names]

        if is_object_detection:
            if len(bboxes) == 0:
                bboxes_tf = tf.zeros((0, 4), dtype=tf.float32)
            else:
                bboxes_tf = tf.convert_to_tensor(bboxes, dtype=tf.float32)

            if len(labels) == 0:
                labels_tf = tf.zeros((0,), dtype=tf.int32)
            else:
                labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)

            annotations_tf.append((bboxes_tf, labels_tf))
            continue

        class_index = labels[0]
        labels_tf = tf.one_hot(class_index, num_classes)
        annotations_tf.append(labels_tf)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, annotations_tf))

    def process_item(img_path, annotations):
        img = safe_decode_image(img_path, grayscale)
        ret = (img, annotations)
        return ret

    dataset = dataset.map(process_item, num_parallel_calls=tf.data.AUTOTUNE)

    if augmentation:
        dataset = dataset.map(lambda img, data: augment_sample(img, data, is_object_detection, grayscale), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def generate_label_map(train_ds, test_ds):
    label_list = []
    label_map = {}
    label_counter = 0

    for dataset in [train_ds, test_ds]:
        for item in dataset:
            labels = item["annotations"][1]
            for l in labels:
                if l not in label_map:
                    label_list.append(l)
                    label_map[l] = label_counter
                    label_counter += 1

    return label_map, label_list


def summarize_dataset(dataset, name):
    """
    Print a simple readable summary of a dataset that stores:
      - image file paths in dataset.image_paths
      - annotations in dataset.annotations (list of lists of bbox dicts)
    """
    num_items = len(dataset)
    print("----- Dataset Summary -----")
    print(f"Name: {name}")
    print(f"Total samples: {num_items}")

    # Count objects and classes
    num_objects = 0
    class_counts = {}

    for data in dataset:
        cls_list = data["annotations"][1]
        for cls in cls_list:
            num_objects += 1
            class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"Total objects: {num_objects}")

    print("Classes:")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]}")

    # Show first example
    # if num_items > 0:
    #     print("\nExample entry:")
    #     print(f"  image_path: {dataset[0]["img_path"]}")
    #     print(f"  annotations ([bboxes], [labels]): {dataset[0]["annotations"]}")

    print("---------------------------")


def process_images(root_folder, target_size, resize_mode, test_split=0.2):
    resize.resize_images(os.path.join(root_folder, "input_resize"), os.path.join(root_folder, "input"), target_size,
                         resize_mode)
    split_data(root_folder, test_split)
