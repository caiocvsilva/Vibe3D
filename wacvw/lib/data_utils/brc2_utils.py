import sys
sys.path.append('.')

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm
import glob

from lib.core.config import VIBE_DB_DIR, BRC_DIR, BRC2_DIR, BRC2_SMPL_DIR
from lib.models.vibe import VIBE_Demo
from lib.data_utils.img_utils import convert_cvimg_to_tensor

def read_data(root_folder, subset, gt_betas_dir, debug=False):
    # Only use these two camera IDs
    valid_cameras = {"G318", "G2302"}
    dataset = { 'vid_name': [], 'id': [], 'img_name': [], 'pose': [], 'shape': [] }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = VIBE_Demo(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)
    pretrained_file = '/blue/sarkar.sudeep/caio.dasilva/Vibe3D/vibe_model_wo_3dpw.pth.tar'
    if torch.cuda.is_available():
        ckpt = torch.load(pretrained_file)
    else:
        ckpt = torch.load(pretrained_file, map_location=torch.device('cpu'))
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # All folders under root are subjectIDs
    subjects = sorted([x for x in os.listdir(root_folder) if osp.isdir(osp.join(root_folder, x))])

    print('subjects: ', subjects)

    split_index = 70
    for i, subject_id in enumerate(tqdm(subjects, desc="Processing subjects")):
        # Partition by index, not int(subject_id)
        if (subset == 'train' and i > split_index) or (subset == 'test' and i <= split_index):
            continue

        subject_folder = osp.join(root_folder, subject_id)
        cameras = sorted([x for x in os.listdir(subject_folder) if osp.isdir(osp.join(subject_folder, x))])
        cameras = [c for c in cameras if c in valid_cameras]

        # Only proceed if BOTH desired cameras exist for this subject and each has frames
        subject_has_both_cameras = True
        for camera_id in valid_cameras:
            frames_dir = osp.join(subject_folder, camera_id, 'frames')
            if not osp.exists(frames_dir):
                subject_has_both_cameras = False
                break
            frames = [x for x in os.listdir(frames_dir) if x.endswith('.png')]
            if not frames:
                subject_has_both_cameras = False
                break
        if not subject_has_both_cameras:
            continue

        for camera_id in cameras:
            # Load the subject's ground truth beta file from the identity folder.
            beta_files = glob.glob(osp.join(gt_betas_dir, f'{subject_id}.npy'))
            if not beta_files:
                print('no beta files')
                continue
            beta_file = beta_files[0]
            if not osp.exists(beta_file):
                print('no beta files2')
                continue

            subject_beta = np.load(beta_file, allow_pickle=True)

            frames_dir = osp.join(subject_folder, camera_id, 'frames')
            if not osp.exists(frames_dir):
                print('no frames')
                continue
            frames = sorted([x for x in os.listdir(frames_dir) if x.endswith('.png')])

            img_paths = []
            batch = []
            gt_betas_list = []

            for frame in frames:
                img_path = osp.join(frames_dir, frame)
                try:
                    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        print(f"Warning: Failed to load image {img_path}, skipping.")
                        continue
                    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    image = convert_cvimg_to_tensor(image)
                    batch.append(image)
                    img_paths.append(img_path)
                    gt_betas_list.append(subject_beta)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}, skipping.")
                    continue

            if not batch:
                print('no batch')
                continue

            img_paths_array = np.array(img_paths)
            batch = torch.stack(batch, dim=0)
            batch = batch.unsqueeze(1)
            batch = batch.to(device)
            batch_size, seqlen = batch.shape[:2]

            with torch.no_grad():
                output = model(batch)[-1]

            pred_pose = output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1).cpu().numpy()
            pred_betas = np.array(gt_betas_list)

            vid_name = f"{subject_id}_{camera_id}"
            dataset['vid_name'].append(np.array([vid_name]*len(frames)))
            dataset['id'].append(np.array([int(subject_id)]*len(frames)))  # Use string subject_id
            dataset['img_name'].append(img_paths_array)
            dataset['pose'].append(pred_pose)
            dataset['shape'].append(pred_betas)
            print('added subject')

    print('dataset: ', dataset)
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])

    return dataset

if __name__ == '__main__':
    debug = False
    gt_betas_dir = BRC2_SMPL_DIR

    dataset = read_data(BRC2_DIR, 'train', gt_betas_dir, debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'brc2_train_db.pt'))

    dataset = read_data(BRC2_DIR, 'test', gt_betas_dir, debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'brc2_test_db.pt'))