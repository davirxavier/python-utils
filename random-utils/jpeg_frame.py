import os
from pathlib import Path
from PIL import Image, ImageFile

# Allow Pillow to load truncated images instead of failing
ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_frames(img_path, output_root, rel_path, x, y, width, height, hor, ver):
    """
    Extract n×k frames from the image and save to the mirrored output directory.
    Includes error handling for truncated/corrupted images.
    """
    try:
        img = Image.open(img_path)
        img.load()  # Force loading to trigger errors early
    except OSError as e:
        print(f"[ERROR] Cannot open image {img_path}: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error opening {img_path}: {e}")
        return

    # Create output directory matching input structure
    output_dir = output_root / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = img_path.stem

    for i in range(hor):
        for j in range(ver):
            startx = x + (i * width / 2)
            starty = y + (j * height / 2)
            endx = startx + width
            endy = starty + height

            frame = img.crop((startx, starty, endx, endy))

            try:
                out_name = f"{basename}_r{i}_c{j}.jpg"
                frame.save(output_dir / out_name)
            except OSError as e:
                print(f"[ERROR] Failed to crop/save frame for {img_path}: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected crop error for {img_path}: {e}")

def main():
    # === Set your parameters here ===
    x = 544
    y = 0
    width = 416
    height = 416
    hor = 1  # horizontal frames
    ver = 4  # vertical frames
    # =================================

    root = Path.cwd()
    output_root = root / "output"
    output_root.mkdir(exist_ok=True)

    jpg_files = list(root.rglob("*.jpg"))

    if not jpg_files:
        print("No .jpg files found.")
        return

    for img_path in jpg_files:
        rel_path = img_path.relative_to(root)

        print(f"Processing: {img_path}")
        extract_frames(img_path, output_root, rel_path, x, y, width, height, hor, ver)

    print(f"Done! Output saved under: {output_root}")


if __name__ == "__main__":
    main()
