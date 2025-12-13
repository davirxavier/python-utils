import os
import shutil
from PIL import Image

# --- CONFIGURATION ---
width = 384
esp32_quality = 3   # ESP32-CAM style quality (0=best, 63=lowest)
triage_folder = "triage"  # Folder for problematic files
# ----------------------

# Convert ESP32 quality (0-63) to Pillow quality (1-95)
def esp32_to_pillow_quality(esp_quality):
    return max(1, min(95, int(95 - (esp_quality / 63) * 90)))

pillow_quality = esp32_to_pillow_quality(esp32_quality)

def convert_rgb888_to_jpg(file_path, output_path, width, quality):
    with open(file_path, "rb") as f:
        data = f.read()

    # Check if file size is divisible by width*3
    if len(data) % (width * 3) != 0:
        print(f"Moving {file_path} to triage: file size {len(data)} is not divisible by width*3")
        # Ensure triage folder exists
        os.makedirs(triage_folder, exist_ok=True)
        shutil.move(file_path, os.path.join(triage_folder, os.path.basename(file_path)))
        return

    height = len(data) // (width * 3)

    # Create image from raw RGB data
    img = Image.frombytes("RGB", (width, height), data)
    img.save(output_path, "JPEG", quality=quality)
    print(f"Saved {output_path} ({width}x{height}) with quality={quality}")

def process_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".rgb888"):
                input_path = os.path.join(root, file)
                output_path = os.path.splitext(input_path)[0] + ".jpg"
                convert_rgb888_to_jpg(input_path, output_path, width, pillow_quality)

if __name__ == "__main__":
    current_folder = os.getcwd()  # Run in execution folder
    process_folder(current_folder)