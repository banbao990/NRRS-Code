# Read all jpg and png images in the folder
# Remove high-frequency information, simple processing
# If it is an RGBA image, set the color of all points with A=1.0 to the average color of these points
# If it is an RGB image, set the color of all points to the average color of these points
# After processing, save to the `textures-lowf` folder

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DIR = os.path.join(PROJECT_DIR, "common", "assets", "scenes-nrrs", "sponza-caustic-lowf", "textures")

import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing


def process_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image: {file_path}")
        return

    if img.shape[2] == 4:  # RGBA
        alpha_channel = img[:, :, 3]
        mask = (alpha_channel == 255)
        if np.any(mask):
            avg_color = cv2.mean(img[:, :, :3], mask.astype(np.uint8))[:3]
            img[mask] = [int(c) for c in avg_color] + [255]
    elif img.shape[2] == 3:  # RGB
        avg_color = cv2.mean(img)[:3]
        img[:, :] = [int(c) for c in avg_color]

    save_dir = os.path.join(os.path.dirname(DIR), "textures-lowf")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    image_files = [os.path.join(DIR, f) for f in os.listdir(DIR) if f.lower().endswith(('.jpg', '.png'))]

    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap_unordered(process_image, image_files), total=len(image_files)))
