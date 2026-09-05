import os
import glob
import random
import cv2
import numpy as np

SRC_DIR = "dataset/clean_xrays"
OUT_DIR = "dataset/processed"
IMG_SIZE = (256, 256)

CLEAN_DIR = os.path.join(OUT_DIR, "clean")
CORRUPTED_DIR = os.path.join(OUT_DIR, "corrupted")
MASKS_DIR = os.path.join(OUT_DIR, "masks")

for d in [CLEAN_DIR, CORRUPTED_DIR, MASKS_DIR]:
    os.makedirs(d, exist_ok=True)

SAMPLE_TEXTS = [
    "R", "L", "AP", "PA", "LATERAL", "PORTABLE", 
    "ER", "CHEST UPRIGHT", "ID: 948201", "24-OCT-2023",
    "NOT FOR DIAGNOSIS", "SEATED", "ER-04"
]

def create_synthetic_pair(clean_gray):
    h, w = clean_gray.shape
    corrupted = clean_gray.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    num_labels = random.randint(2, 5)
    for _ in range(num_labels):
        text = random.choice(SAMPLE_TEXTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = random.uniform(0.5, 0.8)
        thickness = random.randint(1, 2)
        
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size

        x = random.randint(5, max(5, w - text_w - 10))
        y = random.randint(text_h + 10, h - 10)

        # Varying text color (bright white, mid gray, or dark marker)
        text_color = random.choice([255, 230, 180, 40])
        alpha = random.uniform(0.7, 1.0)

        # Temporary overlay for alpha blending
        overlay = corrupted.copy()
        cv2.putText(overlay, text, (x, y), font, scale, text_color, thickness)
        cv2.addWeighted(overlay, alpha, corrupted, 1 - alpha, 0, corrupted)

        # Bounding box mask to match EasyOCR box behavior
        pad = random.randint(2, 5)
        x1 = max(0, x - pad)
        y1 = max(0, y - text_h - pad)
        x2 = min(w, x + text_w + pad)
        y2 = min(h, y + baseline + pad)
        mask[y1:y2, x1:x2] = 255

    return corrupted, mask

def process_dataset(samples_per_image=4):
    image_paths = glob.glob(os.path.join(SRC_DIR, "*.*"))
    if not image_paths:
        print(f"Error: Place clean source images in '{SRC_DIR}'.")
        return

    print(f"Processing {len(image_paths)} images...")
    count = 0
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        clean_resized = cv2.resize(img, IMG_SIZE)
        for aug in range(samples_per_image):
            corrupted, mask = create_synthetic_pair(clean_resized)
            file_id = f"sample_{idx:05d}_{aug}"
            cv2.imwrite(os.path.join(CLEAN_DIR, f"{file_id}.png"), clean_resized)
            cv2.imwrite(os.path.join(CORRUPTED_DIR, f"{file_id}.png"), corrupted)
            cv2.imwrite(os.path.join(MASKS_DIR, f"{file_id}.png"), mask)
            count += 1

    print(f"Finished generating {count} paired samples.")

if __name__ == "__main__":
    process_dataset()