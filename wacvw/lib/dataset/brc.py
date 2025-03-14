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

class BRC(Dataset):
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
        for key, val in self.idx_shape.items():
            self.shape[key] = np.mean(np.take(self.db['shape'], val, axis=0), axis=0, keepdims=True)
        print(len(self.shape), 'identities with mean shape!')

        self.vid_indices = []
        self.idx = []
        self.idx_list = {}
        i=0
        j=1
        video_count=0
        valid_video=0
        while i < len(self.db['vid_name']):
            video_count+=1
            flag=False
            while j < len(self.db['vid_name']) and self.db['vid_name'][i] == self.db['vid_name'][j]:
                assert self.db['id'][i] == self.db['id'][j]
                j+=1
            while j-i >= self.seqlen//3:
                flag=True
                self.vid_indices.append((i,min(i+self.seqlen-1, j-1)))
                self.idx.append(self.db['id'][i])
                if self.db['id'][i] not in self.idx_list:
                    self.idx_list[self.db['id'][i]] = []
                self.idx_list[self.db['id'][i]].append(len(self.vid_indices)-1)
                if j-i < self.seqlen:
                    break
                i+=self.stride
            if flag:
                valid_video+=1
            else:
                print('ignoring', self.db['vid_name'][i], 'with', j-i, 'frames.')
            i=j

        del self.db['vid_name']

        print(f'brc dataset number of videos: {video_count}')
        print(f'brc dataset number of used videos: {valid_video}')
        print(f'brc dataset number of video pieces: {len(self.vid_indices)}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(VIBE_DB_DIR, 'brc_'+self.subset+'_db.pt')
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
        t = transforms.ToTensor()
        while len(img) < self.seqlen:
            img.append(t(np.zeros(i.shape, np.uint8)))
        img = torch.stack(img, dim=0)
        #sil = np.expand_dims(np.stack(sil, axis=0).astype(np.float32)/255.0, 1)

        #print(img.shape, mask.shape, np.unique(self.db['id'][start_index:end_index+1]))

        #print(self.idx[index], np.unique(self.db['id'][start_index:end_index+1]))
        target = {
            'images': img,
            #'silhouettes': torch.from_numpy(sil),
            'length': torch.from_numpy(np.asarray([end_index-start_index+1])),
            #'id': torch.from_numpy(np.unique(self.db['id'][start_index:end_index+1])),
            'id': torch.from_numpy(np.asarray([self.idx[index]])),
            #'shape': torch.from_numpy(np.concatenate((self.db['shape'][start_index:end_index+1], np.zeros((self.seqlen-(end_index-start_index+1), 10), dtype=np.float32)), axis=0)),
            'shape': torch.from_numpy(np.repeat(self.shape[self.idx[index]], self.seqlen, axis=0)),
            'pose': torch.from_numpy(np.concatenate((self.db['pose'][start_index:end_index+1], np.zeros((self.seqlen-(end_index-start_index+1), 72), dtype=np.float32)), axis=0))
        }
        return target

"""
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

print('Load train')
train_db = CasiaB(seqlen=60, subset='train')
print(len(train_db))

print('Load test')
test_db = CasiaB(seqlen=60, subset='test')
print(len(test_db))

sil_loader = DataLoader(dataset=train_db, batch_size=8, num_workers=1, sampler=TripletSampler(train_db))
val_loader = DataLoader(dataset=train_db, batch_size=8, num_workers=1, sampler=ValidationSampler(train_db))
print(len(sil_loader), len(val_loader))

for target in sil_loader:
    seqsize = torch.min(target['length']).item()
    target['images'] = target['images'][:,:seqsize]
    ###target['mask'] = target['mask'][:,:seqsize]
    target['shape'] = target['shape'][:,:seqsize]
    target['pose'] = target['pose'][:,:seqsize]
    print(target['images'].shape, target['id'].shape, target['shape'].shape, target['pose'].shape)
    print(target['images'].type(), target['id'].type(), target['shape'].type(), target['pose'].type())
    print(target['id'])
    print(target['length'])
    ###print(target['mask'])
    break

for target in val_loader:
    seqsize = torch.min(target['length']).item()
    target['images'] = target['images'][:,:seqsize]
    ###target['mask'] = target['mask'][:,:seqsize]
    target['shape'] = target['shape'][:,:seqsize]
    target['pose'] = target['pose'][:,:seqsize]
    print(target['images'].shape, target['id'].shape, target['shape'].shape, target['pose'].shape)
    print(target['images'].type(), target['id'].type(), target['shape'].type(), target['pose'].type())
    print(target['id'])
    print(target['length'])
    ###print(target['mask'])
    break
"""
