import os
import sys
import cv2
import math
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics
from lib.models.vibe import VIBE_Demo
from lib.data_utils.img_utils import convert_cvimg_to_tensor


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


for epoch in ['5', '10', '15', '20', '25', '30', '35', '40', '45']:
    dataset='CasiaB'
    #epoch='5'
    generator_name = '/blue/sarkar.sudeep/mauricio.segundo/models/'+dataset+'-all-generator-'+epoch+'.pytorch'
    gait_head_name = '/blue/sarkar.sudeep/mauricio.segundo/models/'+dataset+'-all-gaithead-'+epoch+'.pytorch'

    seqlen = 36
    stepsize = 18

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_ids = [0]

    generator = VIBE_Demo(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)
    generator = torch.nn.DataParallel(generator, device_ids=device_ids)
    generator.load_state_dict(torch.load(generator_name))
    generator.eval()

    gait_head = GaitHead(10, 207, 2048, 512).to(device)
    gait_head = torch.nn.DataParallel(gait_head, device_ids=device_ids)
    gait_head.load_state_dict(torch.load(gait_head_name))
    gait_head.eval()

    print('OK')

    def get_embedding(folder, files):
        img = []
        for img_name in files:
            i = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_COLOR)
            if i is None:
                print(os.path.join(folder, img_name))
            img.append(convert_cvimg_to_tensor(i))
        img = torch.stack(img, dim=0)

        embedding = []
        for i in range(0, len(img), stepsize):
            batch = torch.unsqueeze(img[i:i+seqlen], 0)
            batch = batch.to(device)

            with torch.no_grad():
                preds = generator(batch)[-1]
                siz = preds['rotmat'].size()
                feats = gait_head(torch.mean(preds['theta'][:, :, 75:], 1), torch.reshape(preds['rotmat'][:, :, 1:], (siz[0], siz[1], -1)))
            embedding.append(feats.cpu().numpy())

            if i+seqlen > len(img):
                break

        embedding = np.concatenate(embedding, axis=0)
        embedding = np.mean(embedding, 0)

        return embedding

    viewpoints = ['000','018','036','054','072','090','108','126','144','162','180']
    conditions = ['nm', 'bg', 'cl']
    identities = ['075','076','077','078','079','080','081','082','083','084','085','086','087','088','089','090','091','092','093','094','095','096','097','098','099','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124']

    path = '/blue/sarkar.sudeep/mauricio.segundo/CasiaB/cropped_frames/'

    gallery = {}
    for viewpoint in viewpoints:
        gallery[viewpoint] = {}
        for identity in identities:
            gallery[viewpoint][identity] = []

        with open('./subsets-casiab/Gallery'+viewpoint, 'r') as fp:
            folders = [x.split()[0] for x in fp.read().splitlines()]

        for folder in folders:
            print(viewpoint, folder, end='\r')
            sys.stdout.flush()
            files = sorted(os.listdir(os.path.join(path, folder)))
            if len(files) <= stepsize:
                print('skip', folder)
                continue
            identity = folder.split('/')[0]
            gallery[viewpoint][identity].append(get_embedding(os.path.join(path, folder), files))

    for key, val in gallery.items():
        print(key, end=':')
        for key2, val2 in val.items():
            print(key2,'(',len(val2),')',sep='', end=' ')
        print()

    for condition in conditions:
        print(condition)
        print('Probe', end='\t')
        for viewpoint in viewpoints:
            print(viewpoint, end='\t')
        print('avg')

        for viewpoint in viewpoints:
            with open('./subsets-casiab/Probe'+viewpoint+'-'+condition, 'r') as fp:
                folders = [x.split()[0] for x in fp.read().splitlines()]

            probe_ids = []
            probe_emb = []
            for folder in folders:
                files = sorted(os.listdir(os.path.join(path, folder)))
                if len(files) <= stepsize:
                    print('skip', folder)
                    continue
                probe_ids.append(folder.split('/')[0])
                probe_emb.append(get_embedding(os.path.join(path, folder), files))

            print(viewpoint, end='\t')
            avg = 0.0
            for gallery_viewpoint in viewpoints:
                rank1 = 0
                for identity, embedding in zip(probe_ids, probe_emb):
                    if len(gallery[gallery_viewpoint][identity]) == 0:
                        continue
                    gen_sim = max([sklearn.metrics.pairwise.cosine_similarity(embedding.reshape(1, -1), gallery_embedding.reshape(1, -1)) for gallery_embedding in gallery[gallery_viewpoint][identity]])
                    flag = True
                    for gallery_identity in identities:
                        if identity != gallery_identity and len(gallery[gallery_viewpoint][gallery_identity]) > 0:
                            imp_sim = max([sklearn.metrics.pairwise.cosine_similarity(embedding.reshape(1, -1), gallery_embedding.reshape(1, -1)) for gallery_embedding in gallery[gallery_viewpoint][gallery_identity]])
                            if imp_sim >= gen_sim:
                                flag = False
                                break
                    if flag:
                        rank1+=1
                print('{:.4f}'.format(rank1/len(folders)), end='\t')
                if viewpoint != gallery_viewpoint:
                    avg += rank1/len(folders)
            print('{:.4f}'.format(avg/(len(viewpoints)-1))) # average excluding identical viewpoint (usually reported in papers)
            sys.stdout.flush()
        print()

    print('DONE\n\n\n')

