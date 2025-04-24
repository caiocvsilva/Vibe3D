import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import torch
import concurrent.futures

def generate_silhouette_with_mediapipe(img):
    """
    Uses MediaPipe Selfie Segmentation to generate a binary mask.
    """
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        # MediaPipe expects RGB
        results = segmenter.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask
        # Threshold to binary
        sil = (mask > 0.1).astype(np.uint8) * 255
        return sil

def process_camera(frames_dir, silhouette_dir, frame_files):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to read {frame_path}")
                continue
            results = segmenter.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mask = results.segmentation_mask
            sil = (mask > 0.1).astype(np.uint8) * 255
            sil_path = os.path.join(silhouette_dir, frame_file)
            cv2.imwrite(sil_path, sil)

def split_list(lst, n):
    """Splits a list into n approximately equal parts."""
    k, m = divmod(len(lst), n)
    return (lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))

def process_dataset(root, num_gpus=2):
    """
    Multi-GPU (multi-process) version: splits the workload by camera across GPUs.
    Each process handles a subset of cameras.
    """
    subjects = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    camera_tasks = []
    for subject in subjects:
        subject_path = os.path.join(root, subject)
        cameras = sorted([d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))])
        for camera in cameras:
            frames_dir = os.path.join(subject_path, camera, "frames")
            silhouette_dir = os.path.join(subject_path, camera, "silhouette")
            if not os.path.exists(frames_dir):
                continue
            os.makedirs(silhouette_dir, exist_ok=True)
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if len(frame_files) == 0:
                continue
            camera_tasks.append((frames_dir, silhouette_dir, frame_files))
    # Split tasks across two GPUs (actually two processes, since MediaPipe is CPU-only or uses its own GPU context)
    splits = list(split_list(camera_tasks, num_gpus))
    def worker(task_list, device_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        for frames_dir, silhouette_dir, frame_files in tqdm(task_list, desc=f"GPU {device_id} cameras"):
            process_camera(frames_dir, silhouette_dir, frame_files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, task_list in enumerate(splits):
            futures.append(executor.submit(worker, task_list, i))
        for f in futures:
            f.result()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate silhouettes for frames dataset using MediaPipe segmentation (multi-GPU).")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs/processes to use")
    args = parser.parse_args()
    process_dataset(args.root, num_gpus=args.gpus)