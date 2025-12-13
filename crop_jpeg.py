import os
from PIL import Image, ImageFile

# Allow Pillow to load truncated images when possible
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== CONFIGURE YOUR CROP HERE =====
x = 384   # left
y = 0     # top
w = 896   # width
h = 896   # height
# ====================================

def main():
    root_dir = os.getcwd()

    for dirpath, _, filenames in os.walk(root_dir):
        # Skip output folders to avoid infinite recursion
        if dirpath.endswith("_cropped"):
            continue

        # Find images in this folder
        images = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg"))]
        if not images:
            continue

        # Create output folder for this directory
        output_folder = dirpath + "_cropped"
        os.makedirs(output_folder, exist_ok=True)

        for filename in images:
            input_path = os.path.join(dirpath, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    cropped = img.crop((x, y, x + w, y + h))
                    cropped.save(output_path, quality=95)

                print(f"{input_path}  →  {output_path}")

            except Exception as e:
                print(f"[WARNING] Skipping corrupted image: {input_path}")
                print(f"          Error: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
