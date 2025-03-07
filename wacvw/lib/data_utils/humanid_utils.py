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

from lib.core.config import VIBE_DB_DIR, HUMANID_DIR
from lib.models.vibe import VIBE_Demo
from lib.data_utils.img_utils import convert_cvimg_to_tensor

#torch.set_num_threads(8)

def read_data(root_folder, subset, debug=False):
	dataset = { 'vid_name': [], 'id': [], 'img_name': [], 'sil_name': [], 'pose': [], 'shape': [] }

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = VIBE_Demo(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)
	pretrained_file = '/blue/sarkar.sudeep/mauricio.segundo/vibe/vibe_data/vibe_model_wo_3dpw.pth.tar'
	if torch.cuda.is_available():
		ckpt = torch.load(pretrained_file)
	else:
		ckpt = torch.load(pretrained_file, map_location=torch.device('cpu'))
	ckpt = ckpt['gen_state_dict']
	model.load_state_dict(ckpt, strict=False)
	model.eval()

	folders = sorted([x for x in os.listdir(osp.join(root_folder, 'cropped_frames'))])
	for folder in folders:
		print(folder)
		surfaces = sorted([x for x in os.listdir(osp.join(root_folder, 'cropped_frames', folder, 'imagery'))])
		for surface in surfaces:
			print('\t', surface)
			videos = sorted([x for x in os.listdir(osp.join(root_folder, 'cropped_frames', folder, 'imagery', surface))])
			for video in videos:
				print('\t\t', video)
				sys.stdout.flush()

				idx = int(video[:5])

				frames = sorted([x for x in os.listdir(osp.join(root_folder, 'cropped_frames', folder, 'imagery', surface, video))])

				img_paths = []
				sil_paths = []
				batch = []
				for frame in frames:
					img_path = osp.join(root_folder, 'cropped_frames', folder, 'imagery', surface, video, frame)
					img_paths.append(img_path)
					sil_path = osp.join(root_folder, 'cropped_silhouettes', folder, 'imagery', surface, video, frame)
					sil_paths.append(sil_path)

					image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
					image = convert_cvimg_to_tensor(image)
					batch.append(image)

				img_paths_array = np.array(img_paths)
				sil_paths_array = np.array(sil_paths)

				batch = torch.stack(batch, dim=0)
				batch = batch.unsqueeze(0)
				batch = batch.to(device)
				batch_size, seqlen = batch.shape[:2]

				with torch.no_grad():
					output = model(batch)[-1]

				pred_pose = output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1).cpu().numpy()
				pred_betas = output['theta'][:,:,75:].reshape(batch_size * seqlen, -1).cpu().numpy()
				pred_betas = np.mean(pred_betas, 0, keepdims=True)
				pred_betas = pred_betas.repeat(pred_pose.shape[0], 0)

				dataset['vid_name'].append(np.array([folder+'_'+surface+'_'+video]*len(frames)))
				dataset['id'].append(np.array([idx]*len(frames)))
				dataset['img_name'].append(img_paths_array)
				dataset['sil_name'].append(sil_paths_array)
				dataset['pose'].append(pred_pose)
				dataset['shape'].append(pred_betas)

	print(subset)
	for k in dataset.keys():
		dataset[k] = np.concatenate(dataset[k])
		print(k, dataset[k].shape, dataset[k].dtype)

	check = np.unique(dataset['vid_name'])
	print(check.shape)

	return dataset

if __name__ == '__main__':
	#parser = argparse.ArgumentParser()
	#parser.add_argument('--dir', type=str, help='dataset directory', default=CASIAB_DIR)
	#args = parser.parse_args()

	debug = False

	dataset = read_data(HUMANID_DIR, 'all', debug=debug)
	joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'humanid_all_db.pt'))

	#dataset = read_data(HUMANID_DIR, 'test', debug=debug)
	#joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'humanid_test_db.pt'))
