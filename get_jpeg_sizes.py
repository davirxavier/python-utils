import os
from PIL import Image

def main():
    root = os.getcwd()
    unique_sizes = set()
    image_files = []

    # Recursively walk the directory tree
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.lower().endswith((".jpg", ".jpeg")):
                path = os.path.join(dirpath, file)
                image_files.append(path)

    # Read sizes
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                size = img.size  # (width, height)
                unique_sizes.add(size)
        except Exception as e:
            print(f"Could not read {img_path}: {e}")

    # Print results
    print("\n=== Unique Image Sizes Found ===\n")
    if not unique_sizes:
        print("No JPEG images found.")
        return

    for w, h in sorted(unique_sizes):
        print(f"{w} x {h}")

    print(f"\nTotal unique sizes: {len(unique_sizes)}")
    print(f"Total images scanned: {len(image_files)}")

if __name__ == "__main__":
    main()
