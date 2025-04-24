#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ReID frames


# In[10]:


# Cell 1: Imports and Setup
import os
import torch
import torchreid
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.notebook import tqdm
import csv
from collections import defaultdict


# In[4]:


# === Paths and Config ===
CONTROL_DIR = '/home/caio.dasilva/datasets/brc2_rotate/'
TEST_DIR = 'output/'
RESULT_CSV = 'reid_results.csv'
SIMILARITY_THRESHOLD = 0.6


# In[5]:


# === Device Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model Setup ===
torchreid.utils.set_random_seed(42)
# torchreid.utils.set_logger()

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


# In[8]:


# === Feature extraction function ===
def extract_feature(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

# === Load multiple control features per subject ===
def load_control_features_multi():
    control_feats = defaultdict(list)

    for fname in os.listdir(CONTROL_DIR):
        if fname.lower().endswith(('.jpg', '.png')):
            subject_id = fname.split('_')[0]
            path = os.path.join(CONTROL_DIR, fname)
            feat = extract_feature(path)
            control_feats[subject_id].append(feat)

    return control_feats

# === Load already processed frames ===
def load_existing_results():
    if os.path.exists(RESULT_CSV):
        df = pd.read_csv(RESULT_CSV)
        return set(df['frame_path'].tolist())
    return set()


# In[ ]:


# Cell 3: Main Loop with Multi-Control Matching + Adaptive Threshold

control_features = load_control_features_multi()
processed_frames = load_existing_results()

with open(RESULT_CSV, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    if os.stat(RESULT_CSV).st_size == 0:  # Write header if CSV is empty
        writer.writerow(['subject_id', 'camera', 'frame_path', 'similarity', 'match'])

    subjects = sorted(os.listdir(TEST_DIR))

    for subject_id in tqdm(subjects, desc='Subjects', unit='subject'):
        subject_path = os.path.join(TEST_DIR, subject_id)
        if not os.path.isdir(subject_path) or subject_id not in control_features:
            continue

        subject_feats = control_features[subject_id]  # list of feature vectors
        cameras = sorted(os.listdir(subject_path))

        for camera in cameras:
            camera_path = os.path.join(subject_path, camera, 'frames')
            if not os.path.isdir(camera_path):
                continue

            frame_list = [f for f in os.listdir(camera_path)
                          if f.lower().endswith(('.jpg', '.png'))]

            frame_paths = []
            similarities = []

            # First pass: collect similarities
            for frame in tqdm(frame_list, desc=f"{subject_id}/{camera} collecting", leave=False, unit='frame'):
                frame_path = os.path.join(camera_path, frame)
                frame_rel_path = os.path.relpath(frame_path)

                if frame_rel_path in processed_frames:
                    continue

                try:
                    frame_feat = extract_feature(frame_path)

                    sims = [cosine_similarity([ctrl_feat], [frame_feat])[0][0] for ctrl_feat in subject_feats]
                    sim = max(sims)

                    frame_paths.append((frame, frame_path, frame_rel_path))
                    similarities.append(sim)
                except Exception as e:
                    print(f"Error during collection: {frame_path}: {e}")

            if not similarities:
                continue

            # Adaptive threshold: 85th percentile
            adaptive_threshold = np.percentile(similarities, 85)

            # Second pass: apply threshold and log
            for (frame, frame_path, frame_rel_path), sim in zip(frame_paths, similarities):
                is_match = sim >= adaptive_threshold

                if not is_match:
                    try:
                        os.remove(frame_path)
                    except Exception as e:
                        print(f"Failed to delete {frame_path}: {e}")

                writer.writerow([subject_id, camera, frame_rel_path, sim, is_match])
                csvfile.flush()

