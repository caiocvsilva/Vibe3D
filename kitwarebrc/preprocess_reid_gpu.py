import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from collections import defaultdict

from kitbrc.brc import Dataset
from kitbrc.annotations.bounding_boxes import Tracklets

import torch
import torchreid
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
CONTROL_DIR = '/home/caio.dasilva/datasets/brc2_rotate/'  # Control images, one per subject
VIDEO_MANIFEST_PATH = '/blue/sarkar.sudeep/caio.dasilva/datasets/brc2-field/video_manifest.txt'
CLIP_ID_TO_FILENAME_PATH = '/blue/sarkar.sudeep/mauricio.segundo/KITBRC2-new/KITBRC2/docs/KITBRC2-clipID-to-filename.txt'
SUBJECT_ID_TO_CLIP_ID_PATH = '/blue/sarkar.sudeep/mauricio.segundo/KITBRC2-new/KITBRC2/docs/KITBRC2-subjectID-to-clipID.txt'
CAMERA_BBOX_BOUNDS_PATH = 'camera_bbox_bounds.json'
OUTPUT_DIR_TEMPLATE = '/home/caio.dasilva/datasets/extracted_brc2/{subject_id}/{camera}/frames'
SIMILARITY_PERCENTILE = 85

# --- REID MODEL AND FEATURE EXTRACTION ---

def load_reid_model(device):
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True
    )
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, transform

def extract_feature(image_path, transform, device, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

def load_control_features(transform, device, model):
    # For each subject_id, gather all "set2" images and compute their features
    control_feats = defaultdict(list)
    for fname in os.listdir(CONTROL_DIR):
        if (fname.lower().endswith(('.jpg', '.png'))) and ('set2' in fname.lower()):
            subject_id = fname.split('_')[0]
            path = os.path.join(CONTROL_DIR, fname)
            feat = extract_feature(path, transform, device, model)
            control_feats[subject_id].append(feat)
    return control_feats

# --- MAIN VIDEO PROCESSING ---

def process_video(video, control_features, model, transform, device):
    video_path = video['file_path']
    subject_id = video['subject_id']
    camera = video['camera']

    if subject_id not in control_features or len(control_features[subject_id]) == 0:
        print(f"[GPU {device}] No control features for subject {subject_id}, skipping video.")
        return

    tracklets = video['tracklets']
    filtered_annotations = tracklets.filter()
    if filtered_annotations is None or filtered_annotations.empty:
        print("No annotations found for the video.")
        return

    output_dir = Path(OUTPUT_DIR_TEMPLATE.format(subject_id=subject_id, camera=camera))
    if os.path.isdir(output_dir):  # Already processed
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    similarities = []
    frame_features = []
    frame_info = []

    # 1. Extract features for all annotated frames first (for percentile thresholding)
    for frame_count in tqdm(range(total_frames), desc=f"Extracting features ({subject_id}/{camera})", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in filtered_annotations.index:
            tracklet = filtered_annotations.loc[frame_count]
            x1, y1, x2, y2 = int(tracklet['TL_x']), int(tracklet['TL_y']), int(tracklet['BR_x']), int(tracklet['BR_y'])

            # Calculate center and ensure 512x512 crop
            width, height = x2 - x1, y2 - y1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            new_width = max(512, width)
            new_height = max(512, height)
            x1 = center_x - new_width // 2
            x2 = center_x + new_width // 2
            y1 = center_y - new_height // 2
            y2 = center_y + new_height // 2
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            cropped_frame = frame[y1:y2, x1:x2]

            # Blank 512x512 image
            output_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            cropped_h, cropped_w = cropped_frame.shape[:2]
            if cropped_h > 512:
                offset = (cropped_h - 512) // 2
                cropped_frame = cropped_frame[offset:offset+512, :]
                cropped_h = 512
            if cropped_w > 512:
                offset = (cropped_w - 512) // 2
                cropped_frame = cropped_frame[:, offset:offset+512]
                cropped_w = 512
            start_y = (512 - cropped_h) // 2
            start_x = (512 - cropped_w) // 2
            output_frame[start_y:start_y+cropped_h, start_x:start_x+cropped_w] = cropped_frame

            # Save temp image for feature extraction
            temp_img_path = output_dir / f'_tmp_{frame_count:04d}.png'
            cv2.imwrite(str(temp_img_path), output_frame)
            # Extract feature (using PIL for transform)
            feat = extract_feature(str(temp_img_path), transform, device, model)
            # Similarity to all control features
            sims = [cosine_similarity([ctrl_feat], [feat])[0][0] for ctrl_feat in control_features[subject_id]]
            sim = max(sims)
            similarities.append(sim)
            frame_features.append(feat)
            frame_info.append((frame_count, output_frame, temp_img_path))
        else:
            continue

    # Calculate adaptive threshold for this video/subject
    if not similarities:
        cap.release()
        print(f"No annotated frames with features for video {video_path}")
        return

    adaptive_threshold = np.percentile(similarities, SIMILARITY_PERCENTILE)

    # 2. Save only frames above threshold
    for idx, (sim, (frame_count, output_frame, temp_img_path)) in enumerate(zip(similarities, frame_info)):
        if sim >= adaptive_threshold:
            frame_output_path = output_dir / f'frame_{frame_count:04d}.png'
            cv2.imwrite(str(frame_output_path), output_frame)
        # Remove temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    cap.release()

def run_videos(videos_subset, gpu_id):
    # Set environment for CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f'cuda:0')  # CUDA_VISIBLE_DEVICES shifts visible devices, so always use cuda:0
    model, transform = load_reid_model(device)
    control_features = load_control_features(transform, device, model)
    for _, video in tqdm(videos_subset.iterrows(), desc=f'[GPU {gpu_id}] Videos'):
        process_video(video, control_features, model, transform, device)

def main():
    # Load dataset
    dataset = Dataset(
        video_manifest=VIDEO_MANIFEST_PATH,
        camera_bbox_bounds=CAMERA_BBOX_BOUNDS_PATH,
        brc=2,
        clip_id_to_filename=CLIP_ID_TO_FILENAME_PATH,
        subject_id_to_clip_id=SUBJECT_ID_TO_CLIP_ID_PATH
    )

    num_gpus = 2
    videos = dataset.videos
    splits = np.array_split(videos, num_gpus)
    processes = []
    for gpu_id, videos_subset in enumerate(splits):
        p = multiprocessing.Process(target=run_videos, args=(videos_subset, gpu_id))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
