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

import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
VIBE_DB_DIR = '/blue/sarkar.sudeep/caio.dasilva/Vibe3D/VIBE/data/vibe_db/'
# AMASS_DIR = '/blue/sarkar.sudeep/mauricio.segundo/vibe/amass/'
AMASS_DIR = '/blue/sarkar.sudeep/mauricio.segundo/AMASS'
# HUMANID_DIR = '/blue/sarkar.sudeep/mauricio.segundo/HumanID/'
HUMANID_DIR = '/blue/sarkar.sudeep/mauricio.segundo/HUMANID/'
# CASIAB_DIR = '/blue/sarkar.sudeep/mauricio.segundo/CasiaB/'
CASIAB_DIR = '/blue/sarkar.sudeep/mauricio.segundo/CASIAB/'
BRC_DIR='/blue/sarkar.sudeep/caio.dasilva/datasets/brc_512'
# BRC2_DIR='/blue/sarkar.sudeep/caio.dasilva/Vibe3D/kitwarebrc/output/'
BRC2_DIR='/home/caio.dasilva/datasets/extracted_brc2'
# BRC2_SMPL_DIR='/blue/sarkar.sudeep/caio.dasilva/datasets/brc2_gt_smpl_num/'
BRC2_SMPL_DIR='/blue/sarkar.sudeep/caio.dasilva/datasets/brc2_gt_smpl/'

INSTA_DIR = 'data/insta_variety'
MPII3D_DIR = 'data/mpi_inf_3dhp'
THREEDPW_DIR = 'data/3dpw'
PENNACTION_DIR = 'data/penn_action'
POSETRACK_DIR = 'data/posetrack'
VIBE_DATA_DIR = '/blue/sarkar.sudeep/caio.dasilva/Vibe3D/VIBE/data/vibe_data/'


# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = True
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 2 #8
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = -1

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
# cfg.TRAIN.DATASET = 'CasiaB'
cfg.TRAIN.DATASET = 'BRC2'
cfg.TRAIN.DATASETS_2D = ['Insta']
cfg.TRAIN.DATASETS_3D = ['MPII3D']
cfg.TRAIN.DATASET_EVAL = 'ThreeDPW'
cfg.TRAIN.BATCH_SIZE = 4 #4
cfg.TRAIN.DATA_2D_RATIO = 0.5
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 100
cfg.TRAIN.PRETRAINED_REGRESSOR = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.RESUME = ''
cfg.TRAIN.NUM_ITERS_PER_EPOCH = 1000
cfg.TRAIN.LR_PATIENCE = 5

# <====== generator optimizer
cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 0.00005
cfg.TRAIN.GEN_WD = 0.0
cfg.TRAIN.GEN_MOMENTUM = 0.9

# <====== motion discriminator optimizer
cfg.TRAIN.MOT_DISCR = CN()
cfg.TRAIN.MOT_DISCR.OPTIM = 'Adam'
cfg.TRAIN.MOT_DISCR.LR = 0.0001
cfg.TRAIN.MOT_DISCR.WD = 0.0001
cfg.TRAIN.MOT_DISCR.MOMENTUM = 0.9
cfg.TRAIN.MOT_DISCR.UPDATE_STEPS = 1
cfg.TRAIN.MOT_DISCR.FEATURE_POOL = 'attention'
cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE = 1024
cfg.TRAIN.MOT_DISCR.NUM_LAYERS = 2
cfg.TRAIN.MOT_DISCR.ATT = CN()
cfg.TRAIN.MOT_DISCR.ATT.SIZE = 1024
cfg.TRAIN.MOT_DISCR.ATT.LAYERS = 3
cfg.TRAIN.MOT_DISCR.ATT.DROPOUT = 0.2

cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 16
cfg.DATASET.OVERLAP = 0.5

cfg.LOSS = CN()
cfg.LOSS.KP_2D_W = 300.
cfg.LOSS.KP_3D_W = 300.
cfg.LOSS.SHAPE_W = 0.06
cfg.LOSS.POSE_W = 60.0
cfg.LOSS.D_MOTION_LOSS_W = 0.5

cfg.MODEL = CN()

cfg.MODEL.TEMPORAL_TYPE = 'gru'

# GRU model hyperparams
cfg.MODEL.TGRU = CN()
cfg.MODEL.TGRU.NUM_LAYERS = 2
cfg.MODEL.TGRU.ADD_LINEAR = True
cfg.MODEL.TGRU.RESIDUAL = True
cfg.MODEL.TGRU.HIDDEN_SIZE = 1024
cfg.MODEL.TGRU.BIDIRECTIONAL = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file
