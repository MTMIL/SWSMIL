import os
import torch
from tqdm import tqdm
import numpy as np
import random
import sys
import argparse
import torch.nn as nn
from torch.nn import functional as F
from utils.metrics import calculate_full_metrics
from utils.preprocess import return_splits
from torch.utils.data import DataLoader
from dataset.dataset import M2Dataset
from models.instance_models import ABMIL, CLAM_MB, CLAM_SB, AttentionGated, TransMIL, MeanMIL, MaxMIL, DSMIL # , DTFD
from modules import rrt
# 插入进度条
from tqdm import tqdm
import logging
from torch.utils.data import Dataset

def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)
    torch.backends.cudnn.deterministic = True
    sys.setrecursionlimit(10000)

class M2Dataset_new(Dataset):
    def __init__(self, feature):
        self.feature = feature
        
    def __getitem__(self, idx):
        
        return self.feature[:, idx, :]

    def __len__(self):
        return self.feature.shape[1]

def m2_patch_pred(model, loader, device, num_classes, label_path):
    model.eval()
    idx_list = []
    logits = torch.Tensor()
    targets = torch.Tensor()
    new_logits = torch.Tensor()
    new_targets = torch.Tensor()
    with torch.no_grad():
        for _, sample in enumerate(loader):
            slide_id, feat = sample['slide_id'], sample['feat']
            feat = feat.to(device)
            logit = model(feat)

            test_list = [item.split('.')[0] for item in os.listdir(label_path)]
            if slide_id[0] in test_list:
                logits = torch.cat((logits, logit.detach().cpu()), dim=0)
                target = torch.from_numpy(np.load(os.path.join(label_path, slide_id[0]+'.npy')))
                targets = torch.cat((targets, target.cpu()), dim=0)
    for idx, t in enumerate(targets):
        if t == -1:
            continue
        else:
            idx_list.append(idx)
    new_logits = logits[idx_list]
    new_targets = targets[idx_list]
    acc, f1, roc_auc, precision, recall, mat = calculate_full_metrics(new_logits, new_targets, num_classes, confusion_mat=True)
    return acc, f1, roc_auc, precision, recall, mat


def m2_patch_pred_mine(model, loader, device, num_classes, label_path, batchsize=512):
    model.eval()
    idx_list = []
    logits = torch.Tensor()
    targets = torch.Tensor()
    new_logits = torch.Tensor()
    new_targets = torch.Tensor()
    with torch.no_grad():
        for _, sample in tqdm(enumerate(loader)):
            slide_id, feat = sample['slide_id'], sample['feat']
            test_list = [item.split('.')[0] for item in os.listdir(label_path)]
            if slide_id[0] in test_list:
                datas = M2Dataset_new(feat)
                loader_new = DataLoader(datas, batch_size=batchsize, shuffle=False, num_workers=0)
                for _, new_feat in tqdm(enumerate(loader_new)):
                    new_feat = new_feat.to(device)
                    # new_feat = new_feat.squeeze(1)
                    logit = model(new_feat)
                    logits = torch.cat((logits, logit.detach().cpu()), dim=0)
                target = torch.from_numpy(np.load(os.path.join(label_path, slide_id[0]+'.npy')))
                targets = torch.cat((targets, target.cpu()), dim=0)
    for idx, t in enumerate(targets):
        if t == -1:
            continue
        else:
            idx_list.append(idx)
    new_logits = logits[idx_list]
    new_targets = targets[idx_list]
    acc, f1, roc_auc, precision, recall, mat = calculate_full_metrics(new_logits, new_targets, num_classes, confusion_mat=True)
    return acc, f1, roc_auc, precision, recall, mat

def get_args():
    parser = argparse.ArgumentParser(description='HOMIL main parameters')

    parser.add_argument('--device_ids', type=str, default=2, help='gpu devices for training')
    parser.add_argument('--seed', type=int, default=10, help='random seed')  # 10
    parser.add_argument('--fold', type=int, default=2, help='fold number')
    parser.add_argument('--num_classes', type=int, default=2, help='classification number')

    parser.add_argument('--csv_dir', type=str, default='./csv/c16', help='csv dir to load data')
    parser.add_argument('--feat_dir', type=str, default='./data/MILs/c16/feat0',help='test dir for feat')
    parser.add_argument('--label_path', type=str, default='./data/MILs/c16/label', help='instance label')
    
    parser.add_argument('--ckpt_dir', type=str, default='./camelyon16/transmil/trans_0/camelyon16.pth',help='dir to load models')
    parser.add_argument('--logging_dir', type=str, default='./logging',help='dir to load models')
    parser.add_argument('--name', type=str, default='debug',help='dir to load models')
    parser.add_argument('--MIL_model', type=str, default='TransMIL',
                        # choices=['ABMIL', 'CLAM_SB', 'CLAM_MB', 'TransMIL', 'GatedABMIL', 'MeanMIL', 'MaxMIL', 'DSMIL', 'DTFD'],
                        help='MIL model to use')

    args = parser.parse_args()
    return args

def get_logger(logging_dir, name):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(logging_dir, '{}.log'.format(name)),
                            level=20,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


if __name__ == '__main__':
    args = get_args()
    args.logging_dir = os.path.join(args.logging_dir, args.MIL_model, str(args.fold))
    get_logger(args.logging_dir, args.name)
    args.device = torch.device('cuda:{}'.format(args.device_ids))
    # print('Using GPU ID: {}'.format(args.device_ids))
    # set random seed
    set_seed(args.seed)
    # print('Using Random Seed: {}'.format(str(args.seed)))

    fold = args.fold
    num_classes = args.num_classes
    csv_dir = args.csv_dir
    csv_path = os.path.join(csv_dir,'Fold_{}.csv'.format(fold))
    feat_dir = args.feat_dir
    label_path = args.label_path
    M2_model_dir = args.ckpt_dir

    _, _, test_dataset = return_splits(csv_path=csv_path, test=True)

    test_dset = M2Dataset(test_dataset, feat_dir)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)

    if 'GatedABMIL' == args.MIL_model:
        model = AttentionGated(n_classes=args.num_classes)
    elif 'ABMIL' == args.MIL_model:
        model = ABMIL(n_classes=args.num_classes)
    elif 'DSMIL' == args.MIL_model:
        model = DSMIL(n_classes=args.num_classes)
    elif 'CLAM_SB' == args.MIL_model:
        model = CLAM_SB(size_arg="small", k_sample=8, n_classes=args.num_classes)
    elif 'CLAM_MB' == args.MIL_model:
        model = CLAM_MB(size_arg="small", k_sample=8, n_classes=args.num_classes)
    elif 'MeanMIL' == args.MIL_model:
        model = MeanMIL(n_classes=args.num_classes)
    elif 'MaxMIL' == args.MIL_model:
        model = MaxMIL(n_classes=args.num_classes)
    elif 'DTFD' == args.MIL_model:
        model = DTFD(args.num_classes, 1024, m_dim=1024, numLayer_Res=0)
    elif 'TransMIL' == args.MIL_model:
        model = TransMIL(n_classes=args.num_classes)
    elif 'AttentionGated' == args.MIL_model:
        model = AttentionGated(n_classes=args.num_classes)
    elif args.MIL_model == 'rrt_mil':
        model_params = {
        'n_classes': 2,
        'dropout': 0.25,
        'act': 'relu',
        'region_num': 8,
        'pos': 'none',
        'pos_pos': 0,
        'pool': 'avg',
        'peg_k': 7,
        'drop_path': 0,
        'n_layers': 1,
        'n_heads': 8,
        'attn': 'rrt',
        'da_act': 'tanh',
        'trans_dropout': 0.1,
        'ffn': False,
        'mlp_ratio': 4.,
        'trans_dim': 64,
        'epeg': True,
        'min_region_num': 0,
        'qkv_bias': True,
        'conv_k': 15,
        'conv_2d': False,
        'conv_bias': True,
        'conv_type': 'attn',
        'region_attn': 'native',
        'peg_1d': False}
        model = rrt.RRT(**model_params)
    else:
        raise NotImplementedError
    model = model.to(args.device)
    if 'DTFD' == args.MIL_model:
        ckpt = torch.load(M2_model_dir, map_location='cpu')
        classifier_ckpt = ckpt['classifier']
        model.load_state_dict(classifier_ckpt, strict=False)
        dimReduction_ckpt = ckpt['dim_reduction']
        model.load_state_dict(dimReduction_ckpt, strict=False)
    # elif 'TransMIL' == args.MIL_model:
    #     ckpt = torch.load(M2_model_dir, map_location='cpu')['state_dict']
    #     weights_dict = {}
    #     for k, v in ckpt.items():
    #         new_k = k.replace('model.', '') if 'model.' in k else k
    #         weights_dict[new_k] = v
    #     model.load_state_dict(weights_dict)
    else:
        model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'), strict=False)
    acc, f1, roc_auc, precision, recall, mat = m2_patch_pred_mine(model, test_loader, args.device, num_classes, label_path=label_path)
    print('acc:{:.4f}, auc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}'.format(acc, roc_auc, f1, precision, recall))
    print(mat)
    logging.info('acc:{:.4f}, auc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}'.format(acc, roc_auc, f1, precision, recall))
    logging.info(mat)
