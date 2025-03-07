import os
import sys
import cv2
import math
import time
import torch
import joblib
import numpy as np
import sklearn.metrics

from lib.models import MotionDiscriminator
from lib.models.vibe import VIBE_Demo
from lib.dataset.loaders import get_data_loaders
from lib.data_utils.img_utils import convert_cvimg_to_tensor
from lib.utils.utils import move_dict_to_device
from lib.utils.geometry import batch_rodrigues

from ztriplet import TripletSemiHardLoss

from lib.core.config import cfg

# ========= Discriminative head ========= #
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GaitHead(torch.nn.Module):
    def __init__(self, input_size, seq_size, hidden_size=2048, output_size=512, dropout=0.0):
        super(GaitHead, self).__init__()
        self.dropout = torch.nn.Dropout2d(dropout)

        # MLP to map 10-dimensional vector to 512-dimensional embedding
        self.mlp1_fc1 = torch.nn.Linear(input_size, hidden_size, False)
        self.mlp1_bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.mlp1_fc2 = torch.nn.Linear(hidden_size, output_size, False)
        self.mlp1_bn2 = torch.nn.BatchNorm1d(output_size)

        # Transformer to map 50x72 sequence to 512-dimensional embedding
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, output_size))
        self.trf_emb = torch.nn.Linear(seq_size, output_size)
        self.trf_bn1 = torch.nn.BatchNorm1d(output_size)
        self.trf_pos_emb = PositionalEncoding(output_size, 0.1)
        encoder_layers = torch.nn.TransformerEncoderLayer(output_size, 8, hidden_size, 0.1)
        self.trf_enc = torch.nn.TransformerEncoder(encoder_layers, 6)

        # MLP to map concatenated embeddings to a single 512-dimensional embedding
        self.mlp2_fc1 = torch.nn.Linear(2*output_size, hidden_size, False)
        self.mlp2_bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.mlp2_fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        h = torch.transpose(h, 0, 1)

        x = self.dropout(self.mlp1_bn1(torch.nn.functional.relu(self.mlp1_fc1(x))))
        x = self.dropout(self.mlp1_bn2(torch.nn.functional.relu(self.mlp1_fc2(x))))

        h_size = h.size()
        h = torch.reshape(h, (-1, h_size[2]))
        h = self.trf_bn1(torch.nn.functional.relu(self.trf_emb(h)))
        h = torch.reshape(h, (h_size[0], h_size[1], -1))

        batch_class_token = self.class_token.expand(-1, h.shape[1], -1)
        h = torch.cat([batch_class_token, h], dim=0)

        h = self.trf_pos_emb(h)
        h = self.dropout(self.trf_enc(h)[0])

        out = torch.cat((x, h), 1)
        out = self.dropout(self.mlp2_bn1(torch.nn.functional.relu(self.mlp2_fc1(out))))
        out = self.mlp2_fc2(out)

        return torch.nn.functional.normalize(out, p=2, dim=-1)

class SMPLloss(torch.nn.Module):
    def __init__(self, e_pose_loss_weight, e_shape_loss_weight):
        super(SMPLloss, self).__init__()
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight

    def forward(self, preds, target):
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

        pred_theta = preds['theta']
        real_shape, pred_shape = reduce(target['shape']), reduce(pred_theta[:, :, 75:])
        real_pose, pred_pose = reduce(target['pose']), reduce(pred_theta[:, :, 3:75])
        loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)

        return loss_shape * self.e_shape_loss_weight + loss_pose * self.e_pose_loss_weight

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        loss_regr_pose = torch.nn.functional.mse_loss(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = torch.nn.functional.mse_loss(pred_betas_valid, gt_betas_valid)
        return loss_regr_pose, loss_regr_betas

class DISCloss(torch.nn.Module):
    def __init__(self, d_motion_loss_weight):
        super(DISCloss, self).__init__()
        self.d_motion_loss_weight = d_motion_loss_weight

    def forward(self, preds, mocap_data, motion_discriminator):
        e_motion_disc_loss = self.enc_loss(motion_discriminator(preds['theta'][:, :, 6:75]))
        e_motion_disc_loss = e_motion_disc_loss * self.d_motion_loss_weight

        fake_motion = preds['theta'].detach()
        real_motion = mocap_data['theta']
        fake_disc_value = motion_discriminator(fake_motion[:, :, 6:75])
        real_disc_value = motion_discriminator(real_motion[:, :, 6:75])
        _, _, d_motion_disc_loss = self.dec_loss(real_disc_value, fake_disc_value)
        d_motion_disc_loss = d_motion_disc_loss * self.d_motion_loss_weight

        return e_motion_disc_loss, d_motion_disc_loss

    def enc_loss(self, disc_value):
        k = disc_value.shape[0]
        return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k

    def dec_loss(self, real_disc_value, fake_disc_value):
        ka = real_disc_value.shape[0]
        kb = fake_disc_value.shape[0]
        lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
        return la, lb, la + lb


def eval(generator, gait_head, data_loader, device):
    batch_size = 32
    generator.eval()
    gait_head.eval()

    with torch.no_grad():
        X = {}
        for j, batch_img in enumerate(data_loader):
            seqsize = torch.min(batch_img['length']).item()
            batch_img['images'] = batch_img['images'][:,:seqsize]
            batch_img['shape'] = batch_img['shape'][:,:seqsize]
            batch_img['pose'] = batch_img['pose'][:,:seqsize]

            print('Batch', j, 'of', len(data_loader), '/', seqsize)
            sys.stdout.flush()

            move_dict_to_device(batch_img, device)

            preds = generator(batch_img['images'])[-1]
            siz = preds['rotmat'].size()
            #fv = gait_head(torch.mean(preds['theta'][:, :, 75:], 1), torch.transpose(torch.reshape(preds['rotmat'][:, :, 1:], (siz[0], siz[1], -1)), 0, 1))
            fv = gait_head(torch.mean(preds['theta'][:, :, 75:], 1), torch.reshape(preds['rotmat'][:, :, 1:], (siz[0], siz[1], -1)))
            #fv = gait_head(torch.mean(preds['theta'][:, :, 75:], 1), torch.transpose(preds['theta'][:, :, 3:75], 0, 1))
            fv_numpy = fv.cpu().numpy()
            for i in range(len(fv_numpy)):
                if batch_img['id'][i].item() not in X:
                    X[batch_img['id'][i].item()] = []
                X[batch_img['id'][i].item()].append(fv_numpy[i:i+1])
        for y in X.keys():
            X[y] = np.concatenate(X[y], axis=0)

    hist_gen = [0]*2001
    hist_imp = [0]*2001

    start = time.time()
    keys = sorted(list(X.keys()))
    for i in range(len(keys)):
        print('Identity', i, 'of', len(keys))
        sys.stdout.flush()

        d = sklearn.metrics.pairwise.cosine_similarity(X[keys[i]])
        d = np.int32(np.clip(d+1.0, 0.0, 2.0)*1000.0+0.5)
        for r in range(d.shape[0]):
            for c in range(r+1, d.shape[1]):
                hist_gen[d[r,c]] += 1

        for j in range(i+1,len(keys)):
            d = sklearn.metrics.pairwise.cosine_similarity(X[keys[i]], X[keys[j]])
            d = np.int32(np.clip(d+1.0, 0.0, 2.0)*1000.0+0.5)
            for r in range(d.shape[0]):
                for c in range(r+1, d.shape[1]):
                    hist_imp[d[r,c]] += 1

    sum_gen = sum(hist_gen)
    hist_gen = [x/sum_gen for x in hist_gen]
    for i in range(1,2001):
        hist_gen[i] += hist_gen[i-1]
    sum_imp = sum(hist_imp)
    hist_imp = [x/sum_imp for x in hist_imp]
    for i in range(1999,-1,-1):
        hist_imp[i] += hist_imp[i+1]

    eer = 100.0
    for i in range(1,2001):
        if hist_gen[i] > hist_imp[i]:
            eer = 100.0*(hist_gen[i]+hist_gen[i-1]+hist_imp[i]+hist_imp[i-1])/4.0
            print('\nEER:', eer)
            break

    generator.train()
    gait_head.train()
    return eer

def main():
    start = time.time()

    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg)
    # [0] img_train [1] img_val [2] mocap_train

    print('loaders in', time.time()-start, 'seconds')
    start = time.time()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_ids = [0,1] # [int(x) for x in os.getenv('CUDA_VISIBLE_DEVICES').split(',')]

    print('device in', time.time()-start, 'seconds')
    sys.stdout.flush()
    start = time.time()

    # ========= Define VIBE model ========= #
    generator = VIBE_Demo(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)
    generator = torch.nn.DataParallel(generator, device_ids=device_ids)

    print('generator in', time.time()-start, 'seconds')
    sys.stdout.flush()
    start = time.time()

    # ========= Load pretrained weights ========= #
    pretrained_file = '/blue/sarkar.sudeep/mauricio.segundo/vibe/vibe_data/vibe_model_w_3dpw.pth.tar'
    ckpt = torch.load(pretrained_file, map_location=device)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    generator.load_state_dict(ckpt, strict=False)
    #generator.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    sys.stdout.flush()

    print('initialization in', time.time()-start, 'seconds')
    sys.stdout.flush()
    start = time.time()

    motion_discriminator = MotionDiscriminator(rnn_size=1024, input_size=69, num_layers=2, output_size=1, feature_pool='attention', attention_size=1024, attention_layers=3, attention_dropout=0.2).to(device)
    motion_discriminator = torch.nn.DataParallel(motion_discriminator, device_ids=device_ids)

    print('discriminator in', time.time()-start, 'seconds')
    sys.stdout.flush()
    start = time.time()

    #gait_head = GaitHead(10, 72, 2048, 512).to(device)
    gait_head = GaitHead(10, 207, 2048, 512).to(device)
    gait_head = torch.nn.DataParallel(gait_head, device_ids=device_ids)
    #gait_head.load_state_dict(torch.load('/blue/sarkar.sudeep/mauricio.segundo/models/headonly-gaithead-100.pytorch'))

    print('gait head in', time.time()-start, 'seconds')
    sys.stdout.flush()
    start = time.time()

    # ========= Generator optimizer ========= #
    optimizer = torch.optim.Adam(lr=cfg.TRAIN.GEN_LR, params=[p for name, p in gait_head.named_parameters()]+[p for name, p in generator.named_parameters()], weight_decay=cfg.TRAIN.GEN_WD)

    disc_optimizer = torch.optim.Adam(lr=cfg.TRAIN.MOT_DISCR.LR, params=motion_discriminator.parameters(), weight_decay=cfg.TRAIN.MOT_DISCR.WD)

    print('optimizer in', time.time()-start, 'seconds')
    sys.stdout.flush()

    smpl_loss = SMPLloss(cfg.LOSS.POSE_W, cfg.LOSS.SHAPE_W).to(device)
    adv_loss = DISCloss(cfg.LOSS.D_MOTION_LOSS_W)

    img_iter = iter(data_loaders[0])
    mocap_iter = iter(data_loaders[2])

    start = time.time()

    eval(generator, gait_head, data_loaders[1], device)

    print('eval in', time.time()-start, 'seconds')
    sys.stdout.flush()

    for epoch in range(cfg.TRAIN.END_EPOCH):
        print('Epoch #'+str(epoch+1))
        sys.stdout.flush()

        start = time.time()
        for iteration in range(cfg.TRAIN.NUM_ITERS_PER_EPOCH):
            try:
                batch_img = next(img_iter)
                if batch_img['images'].size(0) < cfg.TRAIN.BATCH_SIZE:
                    img_iter = iter(data_loaders[0])
                    batch_img = next(img_iter)
            except StopIteration:
                img_iter = iter(data_loaders[0])
                batch_img = next(img_iter)

            try:
                batch_mocap = next(mocap_iter)
                if batch_mocap['theta'].size(0) < cfg.TRAIN.BATCH_SIZE:
                    mocap_iter = iter(data_loaders[2])
                    batch_mocap = next(mocap_iter)
            except StopIteration:
                mocap_iter = iter(data_loaders[2])
                batch_mocap = next(mocap_iter)

            seqsize = torch.min(batch_img['length']).item()
            batch_img['images'] = batch_img['images'][:,:seqsize]
            batch_img['shape'] = batch_img['shape'][:,:seqsize]
            batch_img['pose'] = batch_img['pose'][:,:seqsize]

            batch_mocap['theta'] = batch_mocap['theta'][:,:seqsize]

            move_dict_to_device(batch_img, device)
            move_dict_to_device(batch_mocap, device)

            print('Iteration', iteration, 'of', cfg.TRAIN.NUM_ITERS_PER_EPOCH, '/', seqsize)
            sys.stdout.flush()

            preds = generator(batch_img['images'])[-1]
            siz = preds['rotmat'].size()      
            #feats = gait_head(torch.mean(preds['theta'][:, :, 75:], 1), torch.transpose(preds['theta'][:, :, 3:75], 0, 1))
            #feats = gait_head(torch.mean(preds['theta'][:, :, 75:], 1), torch.transpose(torch.reshape(preds['rotmat'][:, :, 1:], (siz[0], siz[1], -1)), 0, 1))
            feats = gait_head(torch.mean(preds['theta'][:, :, 75:], 1), torch.reshape(preds['rotmat'][:, :, 1:], (siz[0], siz[1], -1)))

            body_loss = smpl_loss(preds, batch_img)
            gait_loss = TripletSemiHardLoss(batch_img['id'], feats, device)

            gen_loss, disc_loss = adv_loss(preds, batch_mocap, motion_discriminator)

            loss = body_loss + gait_loss + gen_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

        print('train epoch in', time.time()-start, 'seconds')

        start = time.time()
        eval(generator, gait_head, data_loaders[1], device)
        print('eval in', time.time()-start, 'seconds')
        sys.stdout.flush()

        if (epoch+1)%5 == 0:
            torch.save(gait_head.state_dict(), '/blue/sarkar.sudeep/mauricio.segundo/models/'+cfg.TRAIN.DATASET+'-all-adv-gaithead-'+str(epoch+1)+'.pytorch')
            torch.save(generator.state_dict(), '/blue/sarkar.sudeep/mauricio.segundo/models/'+cfg.TRAIN.DATASET+'-all-adv-generator-'+str(epoch+1)+'.pytorch')

    print('Training done!')

if __name__ == '__main__':
    main()

