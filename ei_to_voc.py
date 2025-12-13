import json
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image


# Load JSON data from file
def load_json_from_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Get image dimensions
def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None


# Create VOC XML tree
def create_voc_xml(filename, width, height, boxes):
    annotation = Element("annotation")

    folder = SubElement(annotation, "folder")
    folder.text = "images"

    fname = SubElement(annotation, "filename")
    fname.text = filename

    size = SubElement(annotation, "size")
    w = SubElement(size, "width")
    w.text = str(width)
    h = SubElement(size, "height")
    h.text = str(height)
    d = SubElement(size, "depth")
    d.text = "3"

    for box in boxes:
        obj = SubElement(annotation, "object")

        name = SubElement(obj, "name")
        name.text = box["label"]

        pose = SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = SubElement(obj, "bndbox")
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(box["x"])
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(box["y"])
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(box["x"] + box["width"])
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(box["y"] + box["height"])

    xml_str = tostring(annotation)
    pretty_xml = parseString(xml_str).toprettyxml(indent="  ")
    return pretty_xml


# Convert JSON annotations to VOC XML files
def convert_to_voc(input_json, images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in input_json["files"]:
        image_path = os.path.join(images_dir, file["path"])
        width, height = get_image_dimensions(image_path)

        if width is None:
            print(f"Skipping {image_path}")
            continue

        xml_content = create_voc_xml(
            filename=file["path"],
            width=width,
            height=height,
            boxes=file["boundingBoxes"]
        )

        base = os.path.splitext(os.path.basename(file["path"]))[0]
        xml_path = os.path.join(output_dir, f"{base}.xml")

        with open(xml_path, "w") as f:
            f.write(xml_content)

        print(f"Saved: {xml_path}")


# Main
def main(input_file):
    images_dir = os.path.dirname(input_file)
    output_dir = os.path.join(os.getcwd(), "voc_out")  # <-- ALWAYS voc_out at CWD

    input_json = load_json_from_file(input_file)
    convert_to_voc(input_json, images_dir, output_dir)

    print(f"VOC files saved to: {output_dir}")


if __name__ == "__main__":
    input_file = "/home/xav/Pictures/sofa/esp32-cat-detector-fomo-export/training/info.labels"
    main(input_file)
