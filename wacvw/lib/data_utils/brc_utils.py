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

from lib.core.config import VIBE_DB_DIR, BRC_DIR
from lib.models.vibe import VIBE_Demo
from lib.data_utils.img_utils import convert_cvimg_to_tensor

#torch.set_num_threads(8)

def read_data(root_folder, subset, gt_betas_dir, debug=False):
    dataset = { 'vid_name': [], 'id': [], 'img_name': [], 'sil_name': [], 'pose': [], 'shape': [] }

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

    identities = sorted([x for x in os.listdir(osp.join(root_folder, 'frames'))])
    for identity in tqdm(identities, desc="Processing identities"):
        #print(identity)

        idx = int(identity)
        if subset == 'train' and idx > 144 or subset == 'test' and idx <= 144:
            continue

        conditions = sorted([x for x in os.listdir(osp.join(root_folder, 'frames', identity))])
        for condition in tqdm(conditions, desc="Processing conditions", leave=False):
            # print('\t', condition)
            # sys.stdout.flush()
            # Load the subject's ground truth beta file from the identity folder.
            beta_file = osp.join(gt_betas_dir, identity + '.npy')
            subject_beta = np.load(beta_file, allow_pickle=True).item()['betas']  # convert numpy array to dict if needed
            # Access the betas:
            # subject_beta = subject_beta['betas']
            # print('betas:', betas)
            # print('subject_beta', subject_beta)
            # print keys
            # print(subject_beta.keys())
            # print('subject_beta', subject_beta.shape)
            # input("Press Enter to continue...")
            frames = sorted([x for x in os.listdir(osp.join(root_folder, 'frames', identity, condition))])
            # print(f"Processing {identity} {condition} with {len(frames)} frames")

            img_paths = []
            sil_paths = []
            batch = []
            gt_betas_list = []  # list to collect ground truth betas
            for frame in frames:
                img_path = osp.join(root_folder, 'frames', identity, condition, frame)
                img_paths.append(img_path)
                sil_path = osp.join(root_folder, 'silhouettes', identity, condition, frame)
                sil_paths.append(sil_path)

                image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                image = convert_cvimg_to_tensor(image)
                batch.append(image)
                # print("batch", len(batch))
                
                # Append the subject's beta for each frame.
                gt_betas_list.append(subject_beta)
                # print('gt_betas_list', len(gt_betas_list))

            img_paths_array = np.array(img_paths)
            sil_paths_array = np.array(sil_paths)

            batch = torch.stack(batch, dim=0)
            # print('shape', batch.shape)
            # reshape rom (16,3,1920,1080) to (16,1,3,1920,1080)

            batch = batch.unsqueeze(1)
            # print('shape', batch.shape)
            batch = batch.to(device)
            # print('shape', batch.shape)
            batch_size, seqlen = batch.shape[:2]

            # print(f"Processing {identity} {condition} with {batch_size} frames")


            with torch.no_grad():
                output = model(batch)[-1]

            pred_pose = output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1).cpu().numpy()
            pred_betas = output['theta'][:,:,75:].reshape(batch_size * seqlen, -1).cpu().numpy()
            pred_betas = np.mean(pred_betas, 0, keepdims=True)
            pred_betas = pred_betas.repeat(pred_pose.shape[0], 0)

            # print('pred_pose', pred_pose.shape)
            # print('pred_betas', pred_betas.shape)

            # Use ground truth betas instead of model predictions.
            pred_betas = np.array(gt_betas_list)

            # print('pred_betas', pred_betas.shape)
            # input("Press Enter to continue...")

            dataset['vid_name'].append(np.array([identity+'_'+condition]*len(frames)))
            dataset['id'].append(np.array([idx]*len(frames)))
            dataset['img_name'].append(img_paths_array)
            dataset['sil_name'].append(sil_paths_array)
            dataset['pose'].append(pred_pose)
            dataset['shape'].append(pred_betas)

    # print(subset)
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        # print(k, dataset[k].shape, dataset[k].dtype)

    check = np.unique(dataset['vid_name'])
    # print(check.shape)

    return dataset

if __name__ == '__main__':
    debug = False
    # assume ground truth betas are stored in a directory 'betas' under BRC_DIR
    gt_betas_dir = osp.join(BRC_DIR, 'betas')

    dataset = read_data(BRC_DIR, 'train', gt_betas_dir, debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'brc_train_db.pt'))

    dataset = read_data(BRC_DIR, 'test', gt_betas_dir, debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'brc_test_db.pt'))
