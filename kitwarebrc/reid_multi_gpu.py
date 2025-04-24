#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torchreid
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import csv
from collections import defaultdict
from torch.multiprocessing import Process, set_start_method

CONTROL_DIR = '/home/caio.dasilva/datasets/brc2_rotate/'
TEST_DIR = 'output/'
RESULT_CSV_TEMPLATE = 'reid_results_gpu{}.csv'

SIMILARITY_THRESHOLD = 0.6

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def load_control_features_multi(transform, device, model):
    control_feats = defaultdict(list)
    for fname in os.listdir(CONTROL_DIR):
        if (fname.lower().endswith(('.jpg', '.png'))) and ('set2' in fname.lower()):
            subject_id = fname.split('_')[0]
            path = os.path.join(CONTROL_DIR, fname)
            feat = extract_feature(path, transform, device, model)
            control_feats[subject_id].append(feat)
    return control_feats

def extract_feature(image_path, transform, device, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

def process_subjects(subjects_chunk, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Process on GPU {gpu_id} handling {len(subjects_chunk)} subjects.")

    torchreid.utils.set_random_seed(42)

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

    control_features = load_control_features_multi(transform, device, model)

    result_csv = RESULT_CSV_TEMPLATE.format(gpu_id)
    with open(result_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['subject_id', 'camera', 'frame_path', 'similarity', 'match'])

        for subject_id in tqdm(subjects_chunk, desc=f'[GPU {gpu_id}] Subjects'):
            subject_path = os.path.join(TEST_DIR, subject_id)
            if not os.path.isdir(subject_path) or subject_id not in control_features:
                continue

            subject_feats = control_features[subject_id]
            cameras = sorted(os.listdir(subject_path))

            for camera in cameras:
                camera_path = os.path.join(subject_path, camera, 'frames')
                if not os.path.isdir(camera_path):
                    continue

                frame_list = [f for f in os.listdir(camera_path)
                              if f.lower().endswith(('.jpg', '.png'))]

                frame_paths = []
                similarities = []

                for frame in tqdm(frame_list, desc=f"{subject_id}/{camera}", leave=False):
                    frame_path = os.path.join(camera_path, frame)
                    frame_rel_path = os.path.relpath(frame_path)

                    try:
                        frame_feat = extract_feature(frame_path, transform, device, model)
                        sims = [cosine_similarity([ctrl_feat], [frame_feat])[0][0] for ctrl_feat in subject_feats]
                        sim = max(sims)

                        frame_paths.append((frame, frame_path, frame_rel_path))
                        similarities.append(sim)
                    except Exception as e:
                        print(f"[GPU {gpu_id}] Error during collection: {frame_path}: {e}")

                if not similarities:
                    continue

                adaptive_threshold = np.percentile(similarities, 85)

                for (frame, frame_path, frame_rel_path), sim in zip(frame_paths, similarities):
                    is_match = sim >= adaptive_threshold
                    if not is_match:
                        try:
                            os.remove(frame_path)
                        except Exception as e:
                            print(f"[GPU {gpu_id}] Failed to delete {frame_path}: {e}")

                    writer.writerow([subject_id, camera, frame_rel_path, sim, is_match])
                    csvfile.flush()

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    subjects = sorted(os.listdir(TEST_DIR))
    subject_chunks = chunkify(subjects, 3)

    processes = []
    for gpu_id in range(3):
        p = Process(target=process_subjects, args=(subject_chunks[gpu_id], gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Multi-GPU processing complete. You can merge the result CSVs if needed.")

