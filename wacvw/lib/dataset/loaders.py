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

from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler
from lib.dataset import *

import torch
from torch.nn.utils.rnn import pad_sequence

def pad_if_necessary(list_of_items):
    # Check if all items are tensors/arrays of the same shape
    shapes = [item.shape for item in list_of_items if hasattr(item, 'shape')]
    if len(set(shapes)) == 1:
        return torch.stack(list_of_items)
    # If not, pad along first dimension
    return pad_sequence(list_of_items, batch_first=True)

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        items = [item[key] for item in batch]
        if isinstance(items[0], torch.Tensor) and items[0].dim() > 0:  # likely a sequence
            collated[key] = pad_if_necessary(items)
        else:
            collated[key] = items  # leave as list
    return collated


def get_data_loaders(cfg):
    """
    def get_2d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    def get_3d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(set='train', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    # ===== 2D keypoint datasets =====
    train_2d_dataset_names = cfg.TRAIN.DATASETS_2D
    train_2d_db = get_2d_datasets(train_2d_dataset_names)

    data_2d_batch_size = int(cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.DATA_2D_RATIO)
    data_3d_batch_size = cfg.TRAIN.BATCH_SIZE - data_2d_batch_size

    train_2d_loader = DataLoader(
        dataset=train_2d_db,
        batch_size=data_2d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    # ===== 3D keypoint datasets =====
    train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
    train_3d_db = get_3d_datasets(train_3d_dataset_names)

    train_3d_loader = DataLoader(
        dataset=train_3d_db,
        batch_size=data_3d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    """

    # ===== Motion Discriminator dataset =====
    # motion_disc_db = AMASS(seqlen=cfg.DATASET.SEQLEN)
    # motion_disc_loader = DataLoader(dataset=motion_disc_db, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    motion_disc_loader = None

    # ===== Generator dataset =====
    if cfg.TRAIN.DATASET == 'HumanID':
        img_db = HumanID(seqlen=cfg.DATASET.SEQLEN, subset='train')
    elif cfg.TRAIN.DATASET == 'CasiaB':
        img_db = CasiaB(seqlen=cfg.DATASET.SEQLEN, subset='train')
    elif cfg.TRAIN.DATASET == 'BRC':
        img_db = BRC(seqlen=cfg.DATASET.SEQLEN, subset='train')
    elif cfg.TRAIN.DATASET == 'BRC2':
        img_db = BRC2(seqlen=cfg.DATASET.SEQLEN, subset='train')
    img_loader = DataLoader(dataset=img_db, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, sampler=TripletSampler(img_db), collate_fn=custom_collate_fn)
    img_val_loader = DataLoader(dataset=img_db, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, sampler=ValidationSampler(img_db), collate_fn=custom_collate_fn)

    """
    # ===== Evaluation dataset =====
    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(set='val', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    """

    return img_loader, img_val_loader, motion_disc_loader
    #return train_2d_loader, train_3d_loader, motion_disc_loader, valid_loader
