import os
import random

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def shuffle_reduce(keep_count):
    cwd = os.getcwd()

    items = []  # (image, xml_or_None)

    for filename in os.listdir(cwd):
        name, ext = os.path.splitext(filename)

        if ext.lower() in IMAGE_EXTENSIONS:
            xml_name = name + ".xml"
            xml_path = os.path.join(cwd, xml_name)

            if os.path.isfile(xml_path):
                items.append((filename, xml_name))
            else:
                items.append((filename, None))

    if len(items) < keep_count:
        raise ValueError(
            f"Requested {keep_count} images, but only {len(items)} found."
        )

    random.shuffle(items)

    keep_items = items[:keep_count]
    delete_items = items[keep_count:]

    for img, xml in delete_items:
        os.remove(os.path.join(cwd, img))
        if xml:
            os.remove(os.path.join(cwd, xml))

    print(f"Kept {len(keep_items)} images.")
    print(f"Deleted {len(delete_items)} images.")
    print(f"Deleted XMLs where present.")


shuffle_reduce(1500)
