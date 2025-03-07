# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import cv2
import torch
import joblib
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lib.core.config import VIBE_DB_DIR
from lib.data_utils.img_utils import split_into_chunks, convert_cvimg_to_tensor

class HumanID(Dataset):
	def __init__(self, seqlen, subset='train'):
		self.seqlen = seqlen
		self.subset = subset

		self.stride = seqlen

		with open(osp.join(VIBE_DB_DIR, 'humanid_'+subset+'.txt'), 'r') as fp:
			valid_files = set(fp.read().splitlines())

		self.db = self.load_db()

		mask = []
		for name in self.db['vid_name']:
			if name in valid_files:
				mask.append(True)
			else:
				mask.append(False)

		for key, val in self.db.items():
			self.db[key] = val[mask]

		self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)

		print(np.unique(self.db['vid_name']).shape)

		#self.db['vid_name'] = self.db['vid_name'][mask]
		#print(np.unique(self.db['vid_name']).shape)

		del self.db['vid_name']

		self.idx = []
		self.idx_list = {}
		for i, (start, end) in enumerate(self.vid_indices):
			j = np.unique(self.db['id'][start:end+1])[0]
			self.idx.append(j)
			if j not in self.idx_list:
				self.idx_list[j] = []
			self.idx_list[j].append(i)
		#print(len(self.idx_list), 'identities')
		#for k,v in self.idx_list.items():
		#	print(k, len(v))

		print(f'HumanID dataset number of videos: {len(self.vid_indices)}')

	def __len__(self):
		return len(self.vid_indices)

	def __getitem__(self, index):
		return self.get_single_item(index)

	def load_db(self):
		db_file = osp.join(VIBE_DB_DIR, 'humanid_all_db.pt')
		db = joblib.load(db_file)
		return db

	def get_single_item(self, index):
		start_index, end_index = self.vid_indices[index]

		#sil = []
		img = []
		for img_name, sil_name in zip(self.db['img_name'][start_index:end_index+1], self.db['sil_name'][start_index:end_index+1]):
			i = cv2.imread(img_name, cv2.IMREAD_COLOR)
			#s = cv2.imread(sil_name, cv2.IMREAD_GRAYSCALE)
			img.append(convert_cvimg_to_tensor(i))
			#sil.append(s)
		img = torch.stack(img, dim=0)
		#sil = np.expand_dims(np.stack(sil, axis=0).astype(np.float32)/255.0, 1)

		target = {
			'images': img,
			#'silhouettes': torch.from_numpy(sil),
			'id': torch.from_numpy(np.unique(self.db['id'][start_index:end_index+1])),
			'shape': torch.from_numpy(self.db['shape'][start_index:end_index+1]),
			'pose': torch.from_numpy(self.db['pose'][start_index:end_index+1])
		}
		return target

class TripletSampler(torch.utils.data.sampler.Sampler):
	def __init__(self, db):
		self.db = db
		self.next = None

	def __len__(self):
		return 2*len(self.db)

	def __iter__(self):
		lrand = torch.randperm(len(self.db))
		l = []
		for x in lrand:
			l.append(x)
			idx = self.db.idx[x]
			y = torch.randint(len(self.db.idx_list[idx]), (1,))
			l.append(self.db.idx_list[idx][y])
		return iter(l)

class ValidationSampler(torch.utils.data.sampler.Sampler):
	def __init__(self, db, max_img_per_id=50):
		self.l = []
		for k, v in db.idx_list.items():
			if len(v) < max_img_per_id:
				self.l += v
			else:
				step = int(len(v)/max_img_per_id)
				ls = [v[i] for i in range(0, len(v), step)]
				self.l += ls[:max_img_per_id]

	def __len__(self):
		return len(self.l)

	def __iter__(self):
		return iter(self.l)

#sil_db = HumanID(seqlen=20, subset='train')
"""
sil_loader = DataLoader(dataset=sil_db, batch_size=32, num_workers=8, sampler=TripletSampler(sil_db))
val_loader = DataLoader(dataset=sil_db, batch_size=32, num_workers=8, sampler=ValidationSampler(sil_db))
print(len(sil_loader), len(val_loader))

for target in sil_loader:
	print(target['images'].shape, target['silhouettes'].shape, target['id'].shape, target['shape'].shape, target['pose'].shape)
	print(target['images'].type(), target['silhouettes'].type(), target['id'].type(), target['shape'].type(), target['pose'].type())
	print(target['id'])
	break

for target in val_loader:
	print(target['images'].shape, target['silhouettes'].shape, target['id'].shape, target['shape'].shape, target['pose'].shape)
	print(target['images'].type(), target['silhouettes'].type(), target['id'].type(), target['shape'].type(), target['pose'].type())
	print(target['id'])
	break
"""

