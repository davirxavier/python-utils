import json
import os
from PIL import Image


# Function to load JSON data from a file
def load_json_from_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to get image dimensions (width, height) using Pillow
def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None


# Function to convert to TensorFlow format
def convert_to_tensorflow_format(input_json, images_dir):
    output_json = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}  # To store the category ids

    annotation_id = 1  # Start annotation IDs from 1

    # Process each file entry
    for file in input_json["files"]:
        # Construct full path to image (assuming it's in the same folder as the JSON)
        image_path = os.path.join(images_dir, file["path"])

        # Get image dimensions
        width, height = get_image_dimensions(image_path)

        if width is None or height is None:
            print(f"Skipping image due to missing dimensions: {image_path}")
            continue  # Skip the file if dimensions are unavailable

        image_data = {
            "id": file["name"],
            "file_name": file["path"],
            "width": width,
            "height": height
        }

        # Add the image entry
        output_json["images"].append(image_data)

        # Process each bounding box
        for box in file["boundingBoxes"]:
            label = box["label"]
            # Check if label exists in category map
            if label not in category_map:
                category_id = len(category_map) + 1  # Assign a new category id
                category_map[label] = category_id
                output_json["categories"].append({
                    "id": category_id,
                    "name": label,
                    "supercategory": "none"  # Assuming no supercategory for simplicity
                })

            # Compute bounding box coordinates in [x_min, y_min, x_max, y_max]
            x_min = box["x"]
            y_min = box["y"]
            x_max = box["x"] + box["width"]
            y_max = box["y"] + box["height"]

            # Add annotation
            annotation = {
                "id": annotation_id,
                "image_id": file["name"],
                "category_id": category_map[label],
                "bbox": [x_min, y_min, box["width"], box["height"]],
                "area": box["width"] * box["height"],
                "iscrowd": 0
            }
            output_json["annotations"].append(annotation)
            annotation_id += 1

    return output_json


# Main function to process the JSON file
def main(input_file, output_file):
    # Get the directory of the input JSON file
    images_dir = os.path.dirname(input_file)

    # Load the input JSON data
    input_json = load_json_from_file(input_file)

    # Convert to TensorFlow format
    tensorflow_format_json = convert_to_tensorflow_format(input_json, images_dir)

    # Write the output JSON to a file
    with open(output_file, "w") as outfile:
        json.dump(tensorflow_format_json, outfile, indent=4)

    print(f"Converted JSON saved to {output_file}")


# Example usage
if __name__ == "__main__":
    input_file = "/home/xav/Pictures/sofa/esp32-cat-detector-fomo-export/training/info.labels"  # Path to your input JSON file (in the same folder as images)
    output_file = "tensorflow_format.json"  # Path for the output JSON file
    main(input_file, output_file)
