import os
import cv2
import numpy as np
import concurrent.futures  # new import
from tqdm import tqdm  # new import

# Parameters (could be replaced by argparse)
DATASET_ROOT = "/home/caio.dasilva/datasets/brc"  # dataset root folder
OUTPUT_ROOT = "/home/caio.dasilva/datasets/brc_512"  # output folder
CROP_SIZE = 512

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def compute_crop_coords(img_shape, bbox):
    H, W = img_shape[:2]
    x, y, w, h = bbox
    bbox_center_x = x + w // 2
    bbox_center_y = y + h // 2

    # initial crops centered on bbox center
    init_x0 = bbox_center_x - CROP_SIZE // 2
    init_y0 = bbox_center_y - CROP_SIZE // 2

    # allowed range for x: ensure bbox is inside crop
    allowed_min_x = x + w - CROP_SIZE
    allowed_max_x = x
    # adjust based on image boundaries
    allowed_min_x = clamp(allowed_min_x, 0, W - CROP_SIZE)
    allowed_max_x = clamp(allowed_max_x, 0, W - CROP_SIZE)
    crop_x0 = clamp(init_x0, allowed_min_x, allowed_max_x)

    # same for y axis
    allowed_min_y = y + h - CROP_SIZE
    allowed_max_y = y
    allowed_min_y = clamp(allowed_min_y, 0, H - CROP_SIZE)
    allowed_max_y = clamp(allowed_max_y, 0, H - CROP_SIZE)
    crop_y0 = clamp(init_y0, allowed_min_y, allowed_max_y)

    return crop_x0, crop_y0, crop_x0 + CROP_SIZE, crop_y0 + CROP_SIZE

def process_pair(silhouette_path, frame_path, out_silh_path, out_frame_path):
    silh = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(frame_path)
    if silh is None or frame is None:
        print(f"Skipping pair: {silhouette_path} or {frame_path} not found")
        return

    # Find white region in silhouette (assume white is 255)
    coords = cv2.findNonZero(silh)
    if coords is None:
        print(f"No white region in silhouette: {silhouette_path}")
        return
    x, y, w, h = cv2.boundingRect(coords)
    bbox = (x, y, w, h)
    # Scale down images if white part is larger than crop size
    if w > CROP_SIZE or h > CROP_SIZE:
        scale = CROP_SIZE / max(w, h)
        silh = cv2.resize(silh, (0, 0), fx=scale, fy=scale)
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        x = int(x * scale)
        y = int(y * scale)
        w = int(w * scale)
        h = int(h * scale)
        bbox = (x, y, w, h)

    crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_coords(silh.shape, bbox)

    silh_crop = silh[crop_y0:crop_y1, crop_x0:crop_x1]
    frame_crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]

    os.makedirs(os.path.dirname(out_silh_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_frame_path), exist_ok=True)
    cv2.imwrite(out_silh_path, silh_crop)
    cv2.imwrite(out_frame_path, frame_crop)

def main():
    # ...existing code...
    silhouettes_dir = os.path.join(DATASET_ROOT, "silhouettes")
    frames_dir = os.path.join(DATASET_ROOT, "frames")
    tasks = []
    for root, dirs, files in os.walk(silhouettes_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                silh_path = os.path.join(root, file)
                rel_path = os.path.relpath(silh_path, silhouettes_dir)
                frame_path = os.path.join(frames_dir, rel_path)  # corresponding frame image
                out_silh_path = os.path.join(OUTPUT_ROOT, "silhouettes", rel_path)
                out_frame_path = os.path.join(OUTPUT_ROOT, "frames", rel_path)
                tasks.append((silh_path, frame_path, out_silh_path, out_frame_path))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda args: process_pair(*args), tasks), total=len(tasks)))
    # ...existing code...

if __name__ == "__main__":
    main()
