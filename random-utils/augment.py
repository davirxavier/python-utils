#!/usr/bin/env python3
"""
augment.py

Recursively find images (excluding output folders),
apply 3 photometric augmentations per image:
  - random combination of brightness/contrast/gamma/noise
Save in separate output folders while keeping folder structure.
Filenames are kept identical for label compatibility.
"""

import os
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import random

# ---------------- Config ----------------
INPUT_ROOT = Path.cwd()

OUTPUT_ROOTS = {
    "photometric_1": Path.cwd() / "out_photometric_1",
    "photometric_2": Path.cwd() / "out_photometric_2",
    "photometric_3": Path.cwd() / "out_photometric_3",
}

# Photometric ranges
BRIGHTNESS_RANGE = (0.85, 1.3)
CONTRAST_RANGE  = (0.8, 1.2)
GAMMA_RANGE     = (0.85, 1.2)
NOISE_STD       = (0, 3)

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VERBOSE = True
# ----------------------------------------

# Reset output folders
for out_path in OUTPUT_ROOTS.values():
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ----- Helper functions -----
def make_gamma_table(gamma):
    """Return 256-length lookup table for a single channel."""
    return [int((i / 255.0) ** gamma * 255.0 + 0.5) for i in range(256)]

def apply_gamma(img, gamma_value):
    if img.mode != "RGBA":
        table = make_gamma_table(gamma_value)
        return img.point(table * len(img.getbands()))
    else:
        # Split channels
        r, g, b, a = img.split()
        table = make_gamma_table(gamma_value)
        r = r.point(table)
        g = g.point(table)
        b = b.point(table)
        return Image.merge("RGBA", (r, g, b, a))

def add_noise(img, std):
    if std <= 0:
        return img
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)

def photometric_augment(img):
    out = img.copy()
    ops = [
        lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(*BRIGHTNESS_RANGE)),
        lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(*CONTRAST_RANGE)),
        lambda im: apply_gamma(im, random.uniform(*GAMMA_RANGE)),
        lambda im: add_noise(im, random.uniform(*NOISE_STD)),
    ]

    # Randomly pick ops to apply
    chosen_ops = [op for op in ops if random.random() < 0.6]
    if not chosen_ops:
        chosen_ops = [random.choice(ops)]  # Ensure at least one

    for op in chosen_ops:
        out = op(out)
    return out

def process_and_save(img_path, out_root):
    try:
        rel_path = img_path.relative_to(INPUT_ROOT)
    except Exception:
        rel_path = Path(img_path.name)

    out_dir = out_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(img_path)
    except Exception as e:
        log(f"[WARN] Cannot open image {img_path}: {e}")
        return

    ext = rel_path.suffix.lower()
    try:
        aug_img = photometric_augment(img)
        out_path = out_dir / rel_path.name  # Keep original filename

        # Convert to RGB for JPEG if needed
        if ext in (".jpg", ".jpeg") and aug_img.mode in ("RGBA", "P"):
            aug_img = aug_img.convert("RGB")

        aug_img.save(out_path)
    except Exception as e:
        log(f"[ERROR] Augment failed for {img_path}: {e}")

def should_skip_dir(path):
    try:
        p = Path(path).resolve()
        return any(out.resolve() == p or out.resolve() in p.parents for out in OUTPUT_ROOTS.values())
    except Exception:
        return False

def main():
    total = 0
    for root, dirs, files in os.walk(INPUT_ROOT):
        if should_skip_dir(root):
            continue

        for file in files:
            if Path(file).suffix.lower() in VALID_EXT:
                img_path = Path(root) / file
                # Apply 3 photometric augmentations
                for out_root in OUTPUT_ROOTS.values():
                    process_and_save(img_path, out_root)
                total += 1

    log(f"Done. Processed {total} original images into 3 photometric augmented sets.")

if __name__ == "__main__":
    main()
