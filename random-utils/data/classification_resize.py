import os
from PIL import Image, ImageFile, ExifTags
import xml.etree.ElementTree as ET

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Utility functions
# -----------------------------

def load_image_pil(path):
    try:
        with Image.open(path) as img:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_val = exif.get(orientation, None)
                    if orientation_val == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_val == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_val == 8:
                        img = img.rotate(90, expand=True)
            except Exception:
                pass
            return img.convert('RGB')
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def parse_voc_annotations(xml_path):
    objects = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            objects.append({'label': label, 'bbox': (xmin, ymin, xmax, ymax)})
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return objects

def crop_bbox_overlap_ratio(crop_box, bbox):
    """
    Returns fraction of CROP area covered by intersection with bbox.
    crop_box, bbox = (xmin, ymin, xmax, ymax)
    """
    x1 = max(crop_box[0], bbox[0])
    y1 = max(crop_box[1], bbox[1])
    x2 = min(crop_box[2], bbox[2])
    y2 = min(crop_box[3], bbox[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    crop_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
    return inter_area / crop_area if crop_area > 0 else 0

def bbox_intersects(crop_box, bbox):
    x1 = max(crop_box[0], bbox[0])
    y1 = max(crop_box[1], bbox[1])
    x2 = min(crop_box[2], bbox[2])
    y2 = min(crop_box[3], bbox[3])
    return x1 < x2 and y1 < y2

def sliding_window_pil(image, step=50, window_size=(200, 200), sofa_area=None, skip_per_row=0):
    crops = []
    w, h = image.size
    win_w, win_h = window_size

    for y in range(0, h - win_h + 1, step):
        row_kept = 0
        for x in range(0, w - win_w + 1, step):
            if sofa_area is not None:
                sx, sy, sw, sh = sofa_area
                if x + win_w < sx or x > sx + sw or y + win_h < sy or y > sy + sh:
                    continue

            if row_kept < skip_per_row:
                row_kept += 1
                continue

            crop_box = (x, y, x + win_w, y + win_h)
            crops.append({'box': crop_box, 'coords': (x, y)})
            row_kept += 1
    return crops

# -----------------------------
# Main processing function
# -----------------------------

def process_images_voc_dynamic_threshold(window_size=(384, 384), step=75, resize_to=(160, 160),
                                         sofa_area=(576, 0, 320, 1024), skip_per_row=0,
                                         overlap_thresholds=None, default_overlap_threshold=0.08):
    """
    overlap_thresholds: dict like {'cat':0.05, 'human':0.2}
    default_overlap_threshold: used for any label not in overlap_thresholds
    """
    if overlap_thresholds is None:
        overlap_thresholds = {}

    cwd = os.getcwd()
    output_root = os.path.join(cwd, "out")
    os.makedirs(output_root, exist_ok=True)

    unknown_folder = os.path.join(output_root, "unknown")
    os.makedirs(unknown_folder, exist_ok=True)
    triage_folder = os.path.join(output_root, "triage")
    os.makedirs(triage_folder, exist_ok=True)

    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if d.lower() != 'out']

        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            input_path = os.path.join(root, file)
            pil_img = load_image_pil(input_path)
            if pil_img is None:
                continue

            base_name = '.'.join(file.split('.')[:-1])
            xml_path = os.path.join(root, base_name + ".xml")
            objects = parse_voc_annotations(xml_path) if os.path.exists(xml_path) else []

            crops = sliding_window_pil(pil_img, step=step, window_size=window_size,
                                       sofa_area=sofa_area, skip_per_row=skip_per_row)

            for i, crop in enumerate(crops):
                crop_box = crop['box']
                labels_hit = []
                under_threshold = False

                for obj in objects:
                    if bbox_intersects(crop_box, obj['bbox']):
                        ratio = crop_bbox_overlap_ratio(crop_box, obj['bbox'])
                        threshold = overlap_thresholds.get(obj['label'], default_overlap_threshold)
                        if ratio >= threshold:
                            labels_hit.append(obj['label'])
                        else:
                            under_threshold = True

                if labels_hit:
                    for label in labels_hit:
                        label_folder = os.path.join(output_root, label)
                        os.makedirs(label_folder, exist_ok=True)
                        try:
                            crop_img = pil_img.crop(crop_box)
                            if resize_to:
                                crop_img = crop_img.resize(resize_to, Image.LANCZOS)
                            save_path = os.path.join(label_folder, f"{base_name}_crop{i}.jpg")
                            crop_img.save(save_path, format='JPEG', quality=95)
                        except Exception as e:
                            print(f"Error saving crop {save_path}: {e}")
                elif under_threshold:
                    try:
                        crop_img = pil_img.crop(crop_box)
                        if resize_to:
                            crop_img = crop_img.resize(resize_to, Image.LANCZOS)
                        save_path = os.path.join(triage_folder, f"{base_name}_crop{i}.jpg")
                        crop_img.save(save_path, format='JPEG', quality=95)
                    except Exception as e:
                        print(f"Error saving crop {save_path}: {e}")
                else:
                    try:
                        crop_img = pil_img.crop(crop_box)
                        if resize_to:
                            crop_img = crop_img.resize(resize_to, Image.LANCZOS)
                        save_path = os.path.join(unknown_folder, f"{base_name}_crop{i}.jpg")
                        crop_img.save(save_path, format='JPEG', quality=95)
                    except Exception as e:
                        print(f"Error saving crop {save_path}: {e}")

# -----------------------------
# Example calling arguments
# -----------------------------
overlap_thresholds = {
    'cat': 0.05,
    'human': 0.2
}

process_images_voc_dynamic_threshold(
    window_size=(384, 384),
    step=75,
    resize_to=(160, 160),
    sofa_area=(576, 0, 320, 1024),
    skip_per_row=0,
    overlap_thresholds=overlap_thresholds,
    default_overlap_threshold=0.08
)
