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

    # Partition by list order (not integer value)
    split_index = 70
    for i, subject_id in enumerate(tqdm(subjects, desc="Processing subjects")):
        # Partition by index, not int(subject_id)
        if (subset == 'train' and i > split_index) or (subset == 'test' and i <= split_index):
            continue

        cameras = sorted([x for x in os.listdir(osp.join(root_folder, subject_id)) if osp.isdir(osp.join(root_folder, subject_id, x))])

        for camera_id in tqdm(cameras, desc="Processing cameras", leave=False):
            # Load the subject's ground truth beta file from the identity folder.
            # beta_file = osp.join(gt_betas_dir, f'{subject_id}.npy')
            beta_files = glob.glob(osp.join(gt_betas_dir, f'{subject_id}_*.npy'))
            if not beta_files:
                continue
            beta_file = beta_files[0]
            if not osp.exists(beta_file):
                continue
                
            # subject_beta = np.load(beta_file, allow_pickle=True).item()['betas']
            subject_beta = np.load(beta_file, allow_pickle=True)
            # print(subject_beta)
            # input('press enter')

            frames_dir = osp.join(root_folder, subject_id, camera_id, 'frames')
            if not osp.exists(frames_dir):
                continue
            frames = sorted([x for x in os.listdir(frames_dir) if x.endswith('.png')])

            img_paths = []
            batch = []
            gt_betas_list = []

            # for frame in frames:
            #     img_path = osp.join(frames_dir, frame)
            #     img_paths.append(img_path)
            #     print(img_paths)
            #     image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            #     image = convert_cvimg_to_tensor(image)
            #     batch.append(image)
            #     gt_betas_list.append(subject_beta)
                
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

            # Use subjectID_cameraID as vid_name
            vid_name = f"{subject_id}_{camera_id}"
            dataset['vid_name'].append(np.array([vid_name]*len(frames)))
            dataset['id'].append(np.array([subject_id]*len(frames)))  # Use string subject_id
            dataset['img_name'].append(img_paths_array)
            dataset['pose'].append(pred_pose)
            dataset['shape'].append(pred_betas)

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])

    return dataset

if __name__ == '__main__':
    debug = False
    # gt_betas_dir = osp.join(BRC_DIR, 'betas')
    gt_betas_dir = BRC2_SMPL_DIR

    dataset = read_data(BRC2_DIR, 'train', gt_betas_dir, debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'brc2_train_db.pt'))

    dataset = read_data(BRC2_DIR, 'test', gt_betas_dir, debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'brc2_test_db.pt'))