import cv2
import torch
import joblib
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lib.core.config import VIBE_DB_DIR
from lib.data_utils.img_utils import split_into_chunks, convert_cvimg_to_tensor

class BRC2(Dataset):
    def __init__(self, seqlen, subset='train'):
        self.seqlen = seqlen
        self.subset = subset

        self.stride = 10

        self.db = self.load_db()

        self.idx_shape = {}
        for i in range(len(self.db['id'])):
            if self.db['id'][i] not in self.idx_shape:
                self.idx_shape[self.db['id'][i]] = []
            self.idx_shape[self.db['id'][i]].append(i)
        self.shape = {}
        # for key, val in self.idx_shape.items():
        #     self.shape[key] = np.mean(np.take(self.db['shape'], val, axis=0), axis=0, keepdims=True)
        for key, val in self.idx_shape.items():
            valid_val = [i for i in val if i < len(self.db['shape'])]
            if not valid_val:
                print(f"WARNING: No valid indices for {key}")
                continue
            self.shape[key] = np.mean(np.take(self.db['shape'], valid_val, axis=0), axis=0, keepdims=True)
        print(len(self.shape), 'identities with mean shape!')

        self.vid_indices = []
        self.idx = []
        self.idx_list = {}
        i = 0
        j = 1
        video_count = 0
        valid_video = 0
        while i < len(self.db['vid_name']):
            video_count += 1
            flag = False
            while j < len(self.db['vid_name']) and self.db['vid_name'][i] == self.db['vid_name'][j]:
                assert self.db['id'][i] == self.db['id'][j]
                j += 1
            while j - i >= self.seqlen // 3:
                flag = True
                self.vid_indices.append((i, min(i + self.seqlen - 1, j - 1)))
                self.idx.append(self.db['id'][i])
                if self.db['id'][i] not in self.idx_list:
                    self.idx_list[self.db['id'][i]] = []
                self.idx_list[self.db['id'][i]].append(len(self.vid_indices) - 1)
                if j - i < self.seqlen:
                    break
                i += self.stride
            if flag:
                valid_video += 1
            else:
                print('ignoring', self.db['vid_name'][i], 'with', j - i, 'frames.')
            i = j

        del self.db['vid_name']

        print(f'brc dataset number of videos: {video_count}')
        print(f'brc dataset number of used videos: {valid_video}')
        print(f'brc dataset number of video pieces: {len(self.vid_indices)}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(VIBE_DB_DIR, 'brc2_' + self.subset + '_db.pt')
        db = joblib.load(db_file)
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        img = []
        for img_name in self.db['img_name'][start_index:end_index + 1]:
            i = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img.append(convert_cvimg_to_tensor(i))
        t = transforms.ToTensor()
        while len(img) < self.seqlen:
            img.append(t(np.zeros(i.shape, np.uint8)))
        img = torch.stack(img, dim=0)

        target = {
            'images': img,
            'length': torch.from_numpy(np.asarray([end_index - start_index + 1])),
            'id': torch.from_numpy(np.asarray([self.idx[index]])),
            'shape': torch.from_numpy(np.repeat(self.shape[self.idx[index]], self.seqlen, axis=0)),
            'pose': torch.from_numpy(np.concatenate(
                (self.db['pose'][start_index:end_index + 1],
                 np.zeros((self.seqlen - (end_index - start_index + 1), 72), dtype=np.float32)), axis=0))
        }
        return target