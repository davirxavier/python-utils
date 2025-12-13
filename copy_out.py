import os
import shutil

# Define the output directory
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

# Walk through the current working directory recursively
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.lower().endswith(".jpg"):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(output_dir, file)

            # Handle duplicate file names by adding a suffix
            counter = 1
            base_name, ext = os.path.splitext(file)
            while os.path.exists(dst_path):
                dst_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                counter += 1

            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} -> {dst_path}")

print("All JPG files have been copied.")
