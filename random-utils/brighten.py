import os
from PIL import Image, ImageEnhance

def increase_brightness(input_path, output_path, factor=1.3):
    """Increase brightness of an image and save it."""
    try:
        img = Image.open(input_path)
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bright_img.save(output_path)
        print(f"Processed: {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def main():
    root = os.getcwd()
    out_dir = os.path.join(root, "out")

    for dirpath, _, filenames in os.walk(root):
        # Skip the output directory to avoid infinite loops
        if dirpath.startswith(out_dir):
            continue

        for filename in filenames:
            if filename.lower().endswith((".jpg", ".jpeg")):
                input_path = os.path.join(dirpath, filename)

                # Build output path by replacing root with out/ root
                relative_path = os.path.relpath(input_path, root)
                output_path = os.path.join(out_dir, relative_path)

                increase_brightness(input_path, output_path)

if __name__ == "__main__":
    main()
