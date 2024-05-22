import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import time
import math
import h5py
import torch
import random
import faiss
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from sklearn import manifold
from collections import Counter
# from sklearn.cluster import AgglomerativeClustering
from .metrics import draw_metrics
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from .preprocess import set_transforms
from dataset.dataset import M2Dataset, M1Dataset, Extract_Feat_Dataset
from .train_utils import m2_train_epoch, m2_pred, m2_patch_pred, m1_train_epoch, m1_pred, feat_extraction
from models.MIL_models import ABMIL, Feat_Classifier, CLAM_SB, CLAM_MB, Joint_ABMIL, Joint_Feat_Classifier, Aux_Model, resnet50, TransMIL, DSMIL


def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)
    torch.backends.cudnn.deterministic = True
    sys.setrecursionlimit(10000)


class EarlyStopping:
    def __init__(self, model_path, patience=7, warmup_epoch=0, verbose=False):
        self.patience = patience
        self.warmup_epoch = warmup_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.model_path = model_path

    def __call__(self, epoch, val_loss, model, val_acc=None):
        flag = False
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            flag = True
        if val_acc is not None:
            if self.best_acc is None or val_acc >= self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, model, status='acc')
                self.counter = 0
                flag = True
        if flag:
            return self.counter
        self.counter += 1
        print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
        if self.counter >= self.patience and epoch > self.warmup_epoch:
            self.early_stop = True
        return self.counter

    def save_checkpoint(self, score, model, status='loss'):
        """Saves model when validation loss or validation acc decrease."""
        if status == 'loss':
            pre_score = self.val_loss_min
            self.val_loss_min = score
        else:
            pre_score = self.val_acc_max
            self.val_acc_max = score
        torch.save(model.state_dict(), self.model_path)
        if self.verbose:
            print('Valid {} ({} --> {}).  Saving model ...{}'.format(status, pre_score, score, self.model_path))


def adjust_feat_aug(args, mixup_rate=2, discard_factor=0.1):
    mixup_num, discard_rate = args.mixup_num, args.discard_rate
    if args.feat_aug_method == 'dynamic':
        if args.round_id >= args.feat_aug_warmup_round:
            mixup_num = min(args.mixup_num_th,
                            pow(mixup_rate, args.mixup_num + args.round_id - args.feat_aug_warmup_round))
            discard_rate = min(args.discard_rate_th,
                               discard_rate + discard_factor * (args.round_id - args.feat_aug_warmup_round))
    return mixup_num, discard_rate

def tsne_vis(X,y,auc,path):
    X = np.array(X)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    figure = plt.figure(figsize=(10, 8), dpi=80)

    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='.', c=y, cmap='coolwarm', alpha=0.8, s=100)
    # plt.text(0.5, 1, 'AUC on Test Dataset: {}'.format(auc), ha='center', va='center', fontsize=12,
    #          bbox=dict(facecolor='white', alpha=0.5))
    plt.xticks([])
    plt.yticks([])
    save_path = os.path.join(path,'tsne.png')
    plt.savefig(save_path)


def M2_updating(args):
    print('----------------M2_updating starts---------------')
    start = time.time()
    device = args.device
    ts_writer = args.writer
    round_id = args.round_id
    MIL_model = args.MIL_model
    num_classes = args.num_classes
    lr = args.M2_max_lr
    min_lr = args.M2_min_lr
    dataset = args.dataset
    train_dataset, val_dataset = dataset['train'], dataset['val']
    feat_dir = args.pretrained_feat_dir if round_id == 0 else os.path.join(args.feat_dir, 'round_{}'.format(round_id))
    joint = args.joint
    fixed_feat_dir = args.fixed_feat_dir if joint else None
    joint_warmup_epochs = args.joint_warmup_epochs if joint else 0
    dropout_rate = args.dropout_rate if joint else 0
    mixup_num, discard_rate = adjust_feat_aug(args, mixup_rate=2, discard_factor=0.1)

    train_dset = M2Dataset(train_dataset, feat_dir, discard_rate, mixup_num, fixed_feat_dir=fixed_feat_dir)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
    train_dset1 = M2Dataset(train_dataset, feat_dir, fixed_feat_dir=fixed_feat_dir)
    train_loader1 = DataLoader(train_dset1, batch_size=1, shuffle=False, num_workers=0)
    val_dset = M2Dataset(val_dataset, feat_dir, fixed_feat_dir=fixed_feat_dir)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)
    if args.test:
        test_dataset = dataset['test']
        test_dset = M2Dataset(test_dataset, feat_dir, fixed_feat_dir=fixed_feat_dir)
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.M2_model_dir, exist_ok=True)
    if joint:
        M2_model_dir = os.path.join(args.M2_model_dir, 'joint_{}_model_{}.pth'.format(MIL_model, round_id))
        if 'ABMIL' in MIL_model:
            model = Joint_ABMIL(n_classes=num_classes)
        else:
            raise NotImplementedError
    else:
        M2_model_dir = os.path.join(args.M2_model_dir, '{}_model_{}.pth'.format(MIL_model, round_id))
        if 'ABMIL' in MIL_model:
            model = ABMIL(n_classes=num_classes)
        elif 'CLAM_SB' in MIL_model:
            model = CLAM_SB(size_arg="small", k_sample=8, n_classes=num_classes, instance_loss_fn=criterion)
        elif 'CLAM_MB' in MIL_model:
            model = CLAM_MB(size_arg="small", k_sample=8, n_classes=num_classes, instance_loss_fn=criterion)
        elif 'TransMIL' in MIL_model:
            model = TransMIL(n_classes=num_classes)
        elif 'DSMIL' in MIL_model:
            model = DSMIL(n_classes=num_classes)
        else:
            raise NotImplementedError
    model = model.to(device)
    warmup_epoch = 0
    if not os.path.exists(M2_model_dir):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        if joint:
            warmup_early_stopping = EarlyStopping(model_path=M2_model_dir, patience=args.joint_patience, verbose=True)
            for warmup_epoch in range(1, joint_warmup_epochs + 1):
                m2_train_epoch(round_id, warmup_epoch, model, optimizer, train_loader, criterion, device, num_classes,
                               MIL_model, dropout_rate=1, joint=True)
                loss, acc, _, _, _, _ = m2_pred(round_id, model, val_loader, criterion, device, num_classes, MIL_model,
                                                True)
                counter = warmup_early_stopping(warmup_epoch, loss, model, acc)
                if warmup_early_stopping.early_stop:
                    print('Early Stop!')
                    break
                # adjust learning rate
                if counter > 0 and counter % 7 == 0 and lr > min_lr:
                    lr = lr / 3 if lr / 3 >= min_lr else min_lr
                    for params in optimizer.param_groups:
                        params['lr'] = lr
            model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'))
            loss, acc, auc, mat, _, f1 = m2_pred(round_id, model, train_loader1, criterion, device, num_classes,
                                                 MIL_model, True)
            draw_metrics(ts_writer, 'Train_WarmUp', num_classes, loss, acc, auc, mat, f1, round_id)
            loss, acc, auc, mat, _, f1 = m2_pred(round_id, model, val_loader, criterion, device, num_classes,
                                                 MIL_model, True)
            draw_metrics(ts_writer, 'Val_WarmUp', num_classes, loss, acc, auc, mat, f1, round_id)
            if args.test:
                loss, acc, auc, mat, _, f1 = m2_pred(round_id, model, test_loader, criterion, device, num_classes,
                                                     MIL_model, True)
                draw_metrics(ts_writer, 'Test_WarmUp', num_classes, loss, acc, auc, mat, f1, round_id)

            model = Joint_ABMIL(n_classes=num_classes, dropout=0.5).to(device)
            model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'))
            lr = args.M2_max_lr
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        early_stopping = EarlyStopping(model_path=M2_model_dir, patience=args.M2_patience, verbose=True)
        for m2_epoch in range(warmup_epoch + 1, warmup_epoch + args.M2_epochs + 1):
            m2_train_epoch(round_id, m2_epoch, model, optimizer, train_loader, criterion, device, num_classes,
                           MIL_model, dropout_rate=dropout_rate, joint=joint)
            loss, acc, _, _, _, _ = m2_pred(round_id, model, val_loader, criterion, device, num_classes, MIL_model,
                                            joint)
            counter = early_stopping(m2_epoch, loss, model, acc)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
            # adjust learning rate
            if counter > 0 and counter % 7 == 0 and lr > min_lr:
                lr = lr / 3 if lr / 3 >= min_lr else min_lr
                for params in optimizer.param_groups:
                    params['lr'] = lr

    model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'))
    if 'TransMIL' not in MIL_model:
        loss, acc, auc, mat, train_attns, f1 = m2_pred(round_id, model, train_loader1, criterion, device, num_classes,
                                                       MIL_model, joint, 'Train')
        draw_metrics(ts_writer, 'Train_M2', num_classes, loss, acc, auc, mat, f1, round_id)
        loss, acc, auc, mat, val_attns, f1 = m2_pred(round_id, model, val_loader, criterion, device, num_classes,
                                                     MIL_model, joint, 'Val')
        draw_metrics(ts_writer, 'Val_M2', num_classes, loss, acc, auc, mat, f1, round_id)
        if args.test:
            loss, acc, test_auc, mat, test_attns, f1 = m2_pred(round_id, model, test_loader, criterion, device, num_classes,
                                                          MIL_model, joint, 'Test')
            draw_metrics(ts_writer, 'Test', num_classes, loss, acc, test_auc, mat, f1, round_id)
    else:
        loss, acc, auc, mat, f1 = m2_pred(round_id, model, train_loader1, criterion, device, num_classes,
                                                       MIL_model, joint, 'Train')
        draw_metrics(ts_writer, 'Train_M2', num_classes, loss, acc, auc, mat, f1, round_id)
        loss, acc, auc, mat, f1 = m2_pred(round_id, model, val_loader, criterion, device, num_classes,
                                                     MIL_model, joint, 'Val')
        draw_metrics(ts_writer, 'Val_M2', num_classes, loss, acc, auc, mat, f1, round_id)
        if args.test:
            loss, acc, test_auc, mat, f1 = m2_pred(round_id, model, test_loader, criterion, device,
                                                               num_classes,
                                                               MIL_model, joint, 'Test')
            draw_metrics(ts_writer, 'Test', num_classes, loss, acc, test_auc, mat, f1, round_id)
    if joint:
        patch_model = Joint_Feat_Classifier(n_classes=num_classes).to(device)
    else:
        patch_model = Feat_Classifier(n_classes=num_classes).to(device)
    if 'CLAM_SB' in MIL_model:
        model_dict = patch_model.state_dict()
        pretrain_model = torch.load(M2_model_dir, map_location='cpu')
        state_dict = {k: v for k, v in pretrain_model.items() if
                      k == 'attention_net.0.weight' or k == 'attention_net.0.bias' or k in model_dict.keys()}
        state_dict["attention_net.weight"] = state_dict.pop("attention_net.0.weight")
        state_dict['attention_net.bias'] = state_dict.pop('attention_net.0.bias')
        model_dict.update(state_dict)
        patch_model.load_state_dict(model_dict)
    else:
        patch_model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'), strict=False)
    train_probs = m2_patch_pred(patch_model, train_loader1, device, joint)
    val_probs = m2_patch_pred(patch_model, val_loader, device, joint)
    obj = {'train_attns': train_attns, 'train_probs': train_probs,
           'val_attns': val_attns, 'val_probs': val_probs}
    # if args.test:
    #     test_probs = m2_patch_pred(patch_model, test_loader, device, joint, status='test')
        # obj.update({'test_attns': test_attns, 'test_probs': test_probs})
    if args.tsne:
        model.eval()
        feature_x = []
        label_y = []
        with torch.no_grad():
            with tqdm(total=len(train_loader1)) as pbar:
                for _, sample in enumerate(train_loader1):
                    _, feat, target = sample['slide_id'], sample['feat'], sample['target']
                    feat = feat.to(device)
                    _, _, bag_feat = model(feat,need_feature=True)
                    feature_x.append(bag_feat.cpu().squeeze(0).numpy().tolist())
                    label_y += target.numpy().tolist()
                    pbar.update(1)
            with tqdm(total=len(val_loader)) as pbar:
                for _, sample in enumerate(val_loader):
                    _, feat, target = sample['slide_id'], sample['feat'], sample['target']
                    feat = feat.to(device)
                    _, _, bag_feat = model(feat,need_feature=True)
                    feature_x.append(bag_feat.cpu().squeeze(0).numpy().tolist())
                    label_y += target.numpy().tolist()
                    pbar.update(1)
            with tqdm(total=len(test_loader)) as pbar:
                for _, sample in enumerate(test_loader):
                    _, feat, target = sample['slide_id'], sample['feat'], sample['target']
                    feat = feat.to(device)
                    _, _, bag_feat = model(feat,need_feature=True)
                    feature_x.append(bag_feat.cpu().squeeze(0).numpy().tolist())
                    label_y += target.numpy().tolist()
                    pbar.update(1)
        print(len(feature_x))
        print(len(label_y))
        tsne_vis(feature_x,label_y, test_auc, args.vis_dir)
    end = time.time()
    print('M2 use time: ', end - start)

    return obj

def run_kmeans(x, nmb_clusters, soft_centroids=1):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        (list: ids of data in each cluster, float: loss value)
    """
    n_data, d = x.shape
    print(n_data,d)

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    D, I = index.search(x, soft_centroids)

    return I, D, cluster



def distance(cluster1,cluster2,mode = 0):
    total_dis = 0
    m,n = len(cluster1),len(cluster2)
    for ms in cluster1:
        for ns in cluster2:
            if mode == 1:
                a, b = np.linalg.norm(ms), np.linalg.norm(ns)
                ms, ns = ms.reshape(-1),ns.reshape(-1)
                cos = np.dot(ms, ns) / (a * b)
                total_dis += cos
            else:
                dis = np.linalg.norm(ms-ns)
                total_dis += dis
    average_dis = total_dis / (m * n)
    return average_dis


def AgglomerativeClustering(features,labels,target,max_label):
    matrix = np.zeros((max_label, max_label))
    for m in range(max_label):
        for n in range(m, max_label):
            if m == n:
                break
            else:
                matrix[m, n] = distance(features[str(m)], features[str(n)])
    return matrix


def E_step(args, obj):
    dataset = args.dataset
    round_id = args.round_id
    coord_dir = args.coord_dir
    K0 = args.K0
    topk_coord_dir = os.path.join(args.topk_coord_dir, 'round_{}'.format(round_id))
    os.makedirs(topk_coord_dir, exist_ok=True)

    new_obj = {}
    print('------------------E stage starts-----------------')
    start = time.time()

    dset_patch = {}
    attns, probs = {}, {}
    attns.update(obj['train_attns'])
    attns.update(obj['val_attns'])
    probs.update(obj['train_probs'])
    probs.update(obj['val_probs'])

    topk_features = None
    slide_to_label = {}
    slide_to_label.update(dataset['train'])
    slide_to_label.update(dataset['val'])

    with tqdm(total=len(attns)) as pbar:
        for slide_id, attn in attns.items():
            slide_label = slide_to_label[slide_id]

            attn = torch.from_numpy(attn).squeeze(0)
            prob = torch.from_numpy(probs[slide_id])
            prob = torch.transpose(prob, 1, 0)
            prob = prob[slide_label]
            attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
            score = prob * attn

            slide_patch_num = len(score)
            if round_id == 0:
                K = int(min(K0, slide_patch_num / 3))
            else:
                K = int(min(math.ceil((round_id + 1) * K0 * math.log10(slide_patch_num)),
                            slide_patch_num / 3))

            h5py_path = os.path.join(coord_dir, slide_id + '.h5')
            file = h5py.File(h5py_path, 'r')
            coord_dset = file['coords']
            coords = np.array(coord_dset[:])
            file.close()

            # positive Top-K tumor patches
            _, ptopk_id = torch.topk(score, k=K, dim=0)
            ptopk_coords = coords[ptopk_id.numpy()].tolist()
            select_coords = ptopk_coords
            label = [slide_label] * K
            idx = ptopk_id.tolist()

            # negative Top-K tumor patches
            _, ntopk_id = torch.topk(-score, k=K, dim=0)
            ntopk_coords = coords[ntopk_id.numpy()].tolist()
            select_coords = select_coords + ntopk_coords
            label = label + [0] * K
            idx = idx + ntopk_id.tolist()
            if args.save_topK:
                save_ptopk_id = int(min(args.save_ptopK_rate * K, slide_patch_num))
                _, save_ptopk_id = torch.topk(score, k=save_ptopk_id, dim=0)
                save_ptopk_coords = coords[save_ptopk_id.numpy()].tolist()
                save_coords = save_ptopk_coords
                save_ntopK_num = int(min(args.save_ntopK_num, slide_patch_num))
                _, save_ntopk_id = torch.topk(-score, k=save_ntopK_num, dim=0)
                save_ntopk_coords = coords[save_ntopk_id.numpy()].tolist()
                save_coords = save_coords + save_ntopk_coords
                topk_coord_path = os.path.join(topk_coord_dir, slide_id + '.h5')
                # save_img_name = '{}.{}'.format(score, 'png')
                # save_img_path = os.path.join(topk_coord_dir, slide_id)
                # if not os.path.exists(save_img_path):
                #     os.mkdir(save_img_path)
                # for coord in ptopk_coords:
                #     img_name = '{}_{}.{}'.format(int(coord[0]), int(coord[1]), 'png')
                #     img_dir = os.path.join(args.patch_dir, slide_id)
                #     img_path = os.path.join(img_dir, img_name)
                    # shutil.copy(img_path,save_img_path)

                if not os.path.exists(topk_coord_path):
                    f = h5py.File(topk_coord_path, 'w')
                    f["coords"] = save_coords
                    f.close()
            dset_patch[slide_id] = {'coords': select_coords, 'labels': label, 'idx': idx}
            pbar.set_description(item)
            pbar.update(1)
        dset_name = item + '_dset_patch'
        new_set = {dset_name: dset_patch}
        new_obj = {**new_obj, **new_set}

    end = time.time()
    print('E use time: ', end - start)
    return new_obj

def C_step(args, obj):
    dataset = args.dataset
    round_id = args.round_id
    coord_dir = args.coord_dir
    feature_dir = args.pretrained_feat_dir if round_id == 0 else os.path.join(args.feat_dir, 'round_{}'.format(round_id))
    K0 = args.K0
    topk_coord_dir = os.path.join(args.topk_coord_dir, 'round_{}'.format(round_id))
    vis_dir = args.vis_dir
    os.makedirs(topk_coord_dir, exist_ok=True)
    metric = 'bcl'

    new_obj = {}
    print('------------------E stage starts-----------------')
    start = time.time()

    dset_patch = {}
    attns, probs = {}, {}
    attns.update(obj['train_attns'])
    attns.update(obj['val_attns'])
    probs.update(obj['train_probs'])
    probs.update(obj['val_probs'])

    topk_features = None
    slide_to_label = {}
    slide_to_label.update(dataset['train'])
    slide_to_label.update(dataset['val'])
    with tqdm(total=len(attns)) as pbar:
        for slide_id, attn in attns.items():
            slide_label = slide_to_label[slide_id]
            feat = np.array(torch.load(os.path.join(feature_dir, slide_id + '.pt')))
            if metric == 'bcl':
                attn = torch.from_numpy(attn).squeeze(0)
                prob = torch.from_numpy(probs[slide_id])
                prob = torch.transpose(prob, 1, 0)
                prob = prob[slide_label]
                attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
                score = prob * attn
            elif metric == 'shap':
                attn_index = np.argsort(-attn)
                search_num = int(min(100, len(score)))
                search_indices = attn_index[:search_num]
                shap = shapley_value(search_indices, feat, slide_label, model, device, model_suffix)
                ptopk_id = search_indices[np.argsort(-shap)]

            slide_patch_num = len(score)
            K = int(min(K0, slide_patch_num / 3))
            # K = int(min(math.ceil((round_id + 1) * K0 * math.log10(slide_patch_num)), slide_patch_num / 3))

            h5py_path = os.path.join(coord_dir, slide_id + '.h5')
            file = h5py.File(h5py_path, 'r')
            coord_dset = file['coords']
            coords = np.array(coord_dset[:])
            file.close()

            features = np.array(torch.load(os.path.join(feature_dir, slide_id + '.pt')))

            # positive Top-K tumor patches
            _, ptopk_id = torch.topk(score, k=K, dim=0)
            ptopk_coords = coords[ptopk_id.numpy()].tolist()
            ptopk_features = features[ptopk_id.numpy()].tolist()
            idx = ptopk_id.tolist()

            _, ntopk_id = torch.topk(-score, k=K, dim=0)
            ntopk_coords = coords[ntopk_id.numpy()].tolist()
            ntopk_features = features[ntopk_id.numpy()].tolist()
            select_coords = ptopk_coords + ntopk_coords
            select_features = ptopk_features + ntopk_features
            idx = idx + ntopk_id.tolist()

            if slide_label == 0:
                search_features = [item for i,item in enumerate(features) if i != idx]
                search_coords = [item for i,item in enumerate(coords) if i != idx]

            if topk_features is None:
                topk_features = select_features
            else:
                topk_features = topk_features + select_features
            if args.save_topK:
                save_ptopk_id = int(min(args.save_ptopK_rate * K, slide_patch_num))
                _, save_ptopk_id = torch.topk(score, k=save_ptopk_id, dim=0)
                save_ptopk_coords = coords[save_ptopk_id.numpy()].tolist()
                save_coords = save_ptopk_coords
                save_ntopK_num = int(min(args.save_ntopK_num, slide_patch_num))
                _, save_ntopk_id = torch.topk(-score, k=save_ntopK_num, dim=0)
                save_ntopk_coords = coords[save_ntopk_id.numpy()].tolist()
                save_coords = save_coords + save_ntopk_coords
                topk_coord_path = os.path.join(topk_coord_dir, slide_id + '.h5')
                if not os.path.exists(topk_coord_path):
                    f = h5py.File(topk_coord_path, 'w')
                    f["coords"] = save_coords
                    f.close()
            dset_patch[slide_id] = {'coords': select_coords,'search_coords': search_coords, 'features': select_features, 'search_features': search_features, 'idx': idx, 'k': K}
            pbar.update(1)

        cluster_features = np.array(topk_features).astype(np.float32)

        c_num = 5
        max_label = c_num
        # I, D, C = run_kmeans(cluster_features, nmb_clusters=c_num, soft_centroids=1)
        n_data, d = cluster_features.shape
        # print(n_data, d)

        # faiss implementation of k-means
        clus = faiss.Clustering(d, c_num)

        # Change faiss seed at each k-means so that the randomly picked
        # initialization centroids do not correspond to the same feature ids
        # from an epoch to another.
        clus.seed = np.random.randint(1234)

        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        clus.train(cluster_features, index)
        _, I = index.search(cluster_features, 1)

        cluster_set = {str(i): [] for i in range(max_label)}
        p = 0
        while p < len(I):
            for slide_id, slide in dset_patch.items():
                k = slide['k']
                coords = slide['coords']
                slide_label = slide_to_label[slide_id]

                cluster_labels = I[p:p + 2*k].reshape(-1).tolist()
                for idx, label in enumerate(cluster_labels):
                    if idx < k:
                        cluster_set[str(label)] += [slide_label]
                    else:
                        cluster_set[str(label)] += [0]
                labels = cluster_labels
                p = p + 2*k

                slide['labels'] = labels
                del slide['features']
                del slide['k']
                # for idx, coord in enumerate(coords):
                #     cluster_id = cluster_labels[idx]
                #     img_name = '{}_{}.{}'.format(int(coord[0]), int(coord[1]), 'png')
                #     src_img_path = os.path.join(args.patch_dir, slide_id, img_name)
                #     target_img_path = os.path.join(vis_dir, str(cluster_id))
                #     os.makedirs(target_img_path,exist_ok=True)
                #     target_name = '{}_{}_{}_label{}.{}'.format(slide_id,int(coord[0]), int(coord[1]), str(slide_label), 'png')
                #     shutil.copy(src_img_path, os.path.join(target_img_path,target_name))


        condition = {"0": [], "1": []}
        for key, value in cluster_set.items():
            length = len(value)
            counter = Counter(value)
            info = counter.most_common(1)[0]
            idem, t = info[0], info[1]
            condition[str(idem)] += [key]
            print("在第{}个簇中label{}出现的次数最多占比为{}".format(key, idem, t / length))

        # high, low = 0.8, 0.5
        # high_clu = [[], []]
        # mid_clu = [[], []]
        # for key, value in condition.items():
        #     for v in value:
        #         if v[1] > high:
        #             high_clu[int(key)] += [v[0]]
        #         elif v[1] < high and v[1] > low:
        #             mid_clu[int(key)] += [v[0]]
        #
        # for id, tem in enumerate(high_clu):
        #     if tem != []:
        #         condition[str(id)] = tem
        #     else:
        #         condition[str(id)] = mid_clu[id]

        for slide_id, slide in dset_patch.items():
            coords = slide['coords']
            idx = slide['idx']
            labels = slide['labels']
            slide_label = slide_to_label[slide_id]
            keep = condition[str(slide_label)]
            high_indexes = [i for i, x in enumerate(labels) if str(x) in keep]
            new_coords = [x for i, x in enumerate(coords) if i in high_indexes]
            # new_idx = [x for i, x in enumerate(idx) if i in high_indexes]
            new_labels = [x for i, x in enumerate(labels) if i in high_indexes]


            slide['coords'] = new_coords
            slide['labels'] = new_labels
            print(new_coords)
            del slide['idx']

            if slide_label == 0:

                slide_label = slide_to_label[slide_id]
                search_features = slide['search_features']
                search_coords = np.array(slide['search_coords'])
                D_search, I_search = index.search(np.array(search_features).astype(np.float32), 1)
                D_search, I_search = D_search.reshape(-1), I_search.reshape(-1)
                key_cluster = int(condition["1"][0])
                need_idx = np.argsort(D_search[np.argwhere(I_search == key_cluster)],axis=0)[:8]
                need_coords = search_coords[need_idx].reshape(-1,2)
                slide['coords'] += need_coords
                slide['labels'] += [0] * len(new_coords)

                for idx, coord in enumerate(need_coords):
                    img_name = '{}_{}.{}'.format(int(coord[0]), int(coord[1]), 'png')
                    src_img_path = os.path.join(args.patch_dir, slide_id, img_name)
                    target_img_path = os.path.join(vis_dir, str(c_num))
                    os.makedirs(target_img_path,exist_ok=True)
                    target_name = '{}_{}_{}_label{}.{}'.format(slide_id,int(coord[0]), int(coord[1]), str(slide_label), 'png')
                    shutil.copy(src_img_path, os.path.join(target_img_path,target_name))


        dset_name = 'dset_patch'
        new_set = {dset_name: dset_patch}
        new_obj = {**new_obj, **new_set}

    end = time.time()
    print('E use time: ', end - start)
    return new_obj

def shapley_value(search_indices, data, label, model, device, MIL_model='ABMIL', shuffle=True, shuffle_time=2,
                  subset_num=3):
    model.eval()
    with torch.no_grad():
        left_indices = [i for i in range(data.shape[1]) if i not in search_indices]
        random.shuffle(left_indices)
        left_data = data[:, left_indices, :]
        left_logits = []
        subset_data = [left_data[:,i::subset_num,:] for i in range(subset_num)]
        for _subset_data in subset_data:
            if 'ABMIL' in MIL_model:
                left_logit, _ = model(_subset_data.to(device))
            elif 'CLAM' in MIL_model:
                left_logit, _, results_dict = model(_subset_data.to(device), return_features=True)
            elif 'TransMIL' in MIL_model:
                left_logit = model(_subset_data.to(device))
            else:
                raise NotImplementedError
            left_logits.append(left_logit.cpu())
        cont = torch.zeros((data.shape[1], left_logit.shape[-1]))
        for i in search_indices:
            for j, _subset_data in enumerate(subset_data):
                x = torch.cat((data[:, i, :].unsqueeze(0), _subset_data), axis=1)
                for _ in range(shuffle_time):
                    if shuffle:
                        idx = torch.randperm(x.shape[1])
                        x = x[:, idx, :]
                    if 'ABMIL' in MIL_model:
                        logit, _ = model(x.to(device))
                    elif 'CLAM' in MIL_model:
                        logit, _, _ = model(x.to(device))
                    elif 'TransMIL' in MIL_model:
                        logit = model(x.to(device))
                    else:
                        raise NotImplementedError
                    cont[i] = cont[i] + logit.cpu() - left_logits[j]
        cont = cont / shuffle_time
        score = cont[search_indices, int(label)]
        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
    return score

def D_step(args, obj):
    dataset = args.dataset
    round_id = args.round_id
    coord_dir = args.coord_dir
    feature_dir = args.pretrained_feat_dir if round_id == 0 else os.path.join(args.feat_dir,'round_{}'.format(round_id))
    K0 = args.K0
    topk_coord_dir = os.path.join(args.topk_coord_dir, 'round_{}'.format(round_id))
    os.makedirs(topk_coord_dir, exist_ok=True)
    metric = 'bcl'

    new_obj = {}
    print('------------------E stage starts-----------------')
    start = time.time()

    dset_patch = {}
    attns,probs = {}, {}
    attns.update(obj['train_attns'])
    attns.update(obj['val_attns'])
    probs.update(obj['train_probs'])
    probs.update(obj['val_probs'])

    topk_features = None
    slide_to_label = {}
    slide_to_label.update(dataset['train'])
    slide_to_label.update(dataset['val'])
    with tqdm(total=len(attns)) as pbar:
        for slide_id, attn in attns.items():
            slide_label = slide_to_label[slide_id]
            feat = np.array(torch.load(os.path.join(feature_dir, slide_id + '.pt')))
            if metric == 'bcl':
                attn = torch.from_numpy(attn).squeeze(0)
                prob = torch.from_numpy(probs[slide_id])
                prob = torch.transpose(prob, 1, 0)
                prob = prob[slide_label]
                attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
                score = prob * attn
            elif metric == 'shap':
                attn_index = np.argsort(-attn)
                search_num = int(min(100, len(score)))
                search_indices = attn_index[:search_num]
                shap = shapley_value(search_indices, feat, slide_label, model, device, model_suffix)
                ptopk_id = search_indices[np.argsort(-shap)]


            slide_patch_num = len(score)
            if round_id == 0:
                K = int(min(K0, slide_patch_num / 3))
            else:
                K = int(min(math.ceil((round_id + 1) * K0 * math.log10(slide_patch_num)),
                            slide_patch_num / 3))

            h5py_path = os.path.join(coord_dir, slide_id + '.h5')
            file = h5py.File(h5py_path, 'r')
            coord_dset = file['coords']
            coords = np.array(coord_dset[:])
            file.close()

            features = np.array(torch.load(os.path.join(feature_dir, slide_id + '.pt')))

            # positive Top-K tumor patches
            _, ptopk_id = torch.topk(score, k=K, dim=0)
            ptopk_coords = coords[ptopk_id.numpy()].tolist()
            ptopk_features = features[ptopk_id.numpy()].tolist()
            # select_coords = ptopk_coords
            # select_features = ptopk_features
            # label = [slide_label] * K
            idx = ptopk_id.tolist()

            # negative Top-K tumor patches
            # _, ntopk_id = torch.topk(-score, k=K, dim=0)
            # ntopk_coords = coords[ntopk_id.numpy()].tolist()
            # ntopk_features = features[ntopk_id.numpy()].tolist()
            # select_coords = select_coords + ntopk_coords
            # select_features = select_features + ntopk_features
            # label = label + [0] * K
            # idx = idx + ntopk_id.tolist()
            if slide_label != 0:
                if topk_features is None:
                    topk_features = ptopk_features
                else:
                    topk_features = topk_features + ptopk_features
            if args.save_topK:
                save_ptopk_id = int(min(args.save_ptopK_rate * K, slide_patch_num))
                _, save_ptopk_id = torch.topk(score, k=save_ptopk_id, dim=0)
                save_ptopk_coords = coords[save_ptopk_id.numpy()].tolist()
                save_coords = save_ptopk_coords
                save_ntopK_num = int(min(args.save_ntopK_num, slide_patch_num))
                _, save_ntopk_id = torch.topk(-score, k=save_ntopK_num, dim=0)
                save_ntopk_coords = coords[save_ntopk_id.numpy()].tolist()
                save_coords = save_coords + save_ntopk_coords
                topk_coord_path = os.path.join(topk_coord_dir, slide_id + '.h5')
                if not os.path.exists(topk_coord_path):
                    f = h5py.File(topk_coord_path, 'w')
                    f["coords"] = save_coords
                    f.close()
            dset_patch[slide_id] = {'coords': ptopk_coords, 'features': ptopk_features, 'idx': idx, 'k': K}
            # pbar.set_description(item)
            pbar.update(1)

        cluster_features = np.array(topk_features).astype(np.float32)

        c_num = 10
        max_label = c_num
        I, D = run_kmeans(cluster_features, nmb_clusters=c_num, soft_centroids=1)

        cluster_set ={str(i):[] for i in range(max_label)}
        p = 0
        while p < len(I):
            for slide_id,slide in dset_patch.items():
                k = slide['k']
                slide_label = slide_to_label[slide_id]
                if slide_label == 0:
                    labels = [0] * k
                else:
                    cluster_labels = I[p:p+k].reshape(-1).tolist()
                    for label in cluster_labels:
                        cluster_set[str(label)] += [slide_label]
                    labels = cluster_labels
                    p = p + k

                slide['labels'] = labels
                del slide['features']
                del slide['k']

        condition = {"1": [], "2": []}
        for key,value in cluster_set.items():
            length = len(value)
            counter = Counter(value)
            info = counter.most_common(1)[0]
            idem,t = info[0],info[1]
            condition[str(idem)] += [(key,t/length)]
            print("在第{}个簇中label{}出现的次数最多占比为{}".format(key, idem, t/length))

        high,low = 0.8,0.5
        high_clu = [[],[]]
        mid_clu = [[],[]]
        low_clu = []
        for key,value in condition.items():
            for v in value:
                if v[1] > high:
                    high_clu[int(key)-1] += [v[0]]
                elif v[1] < high and v[1] > low:
                    mid_clu[int(key)-1] += [v[0]]
                elif v[1] < low:
                    low_clu.append(v[0])

        for id,tem in enumerate(high_clu):
            if tem != []:
                condition[str(id+1)] = tem
            else:
                condition[str(id+1)] = mid_clu[id]

        condition["3"] = low_clu

        for slide_id, slide in dset_patch.items():
            coords = slide['coords']
            idx = slide['idx']
            labels = slide['labels']
            slide_label = slide_to_label[slide_id]
            if slide_label != 0:
                new_labels,new_coords,new_idx = [],[],[]
                keep = condition[str(slide_label)]
                high_indexes = [i for i, x in enumerate(labels) if str(x) in keep]
                low_indexes = [i for i, x in enumerate(labels) if str(x) in condition['3']]
                new_coords = [x for i, x in enumerate(coords) if i in high_indexes or i in low_indexes]
                new_idx = [x for i, x in enumerate(idx) if i in high_indexes or i in low_indexes]
                for i, x in enumerate(labels):
                    if i in high_indexes:
                        new_labels.append(slide_label)
                    elif i in low_indexes:
                        new_labels.append(3)

                slide['coords'] = new_coords
                slide['idx'] = new_idx
                slide['labels'] = new_labels

        # tsne_vis(features_x,labels_y,item)
        # dset_name = item + '_dset_patch'
        dset_name = 'dset_patch'
        new_set = {dset_name: dset_patch}
        new_obj = {**new_obj, **new_set}

    end = time.time()
    print('E use time: ', end - start)
    return new_obj


def M1_updating(args, new_obj):
    print('----------------M1_updating starts---------------')
    start = time.time()
    device = args.device
    ts_writer = args.writer
    round_id = args.round_id
    patch_dir = args.patch_dir
    num_classes = args.C_numbers
    dataset = args.dataset

    batch_size = args.M1_batch_size
    dset_patch = new_obj['dset_patch']
    train_dset_patch = {}
    val_dset_patch = {}
    train_dataset, val_dataset = dataset['train'], dataset['val']
    for key,value in dset_patch.items():
        # coords = value['coords']
        # labels = value['labels']
        if key in train_dataset:
            train_dset_patch[key] = value
        else:
            val_dset_patch[key] = value
        # train_dset_patch[key] = {'coords':coords[:int(len(coords)*0.8)], 'labels':labels[:int(len(labels)*0.8)]}
        # val_dset_patch[key] = {'coords':coords[int(len(coords)*0.8):], 'labels':labels[int(len(labels)*0.8):]}
    # train_dset_patch = new_obj['train_dset_patch']
    # val_dset_patch = new_obj['val_dset_patch']
    train_dset = M1Dataset(split=train_dset_patch, patch_dir=patch_dir, transform=set_transforms(True))
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    train_dset1 = M1Dataset(split=train_dset_patch, patch_dir=patch_dir, transform=set_transforms(False))
    train_loader1 = DataLoader(train_dset1, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    val_dset = M1Dataset(split=val_dset_patch, patch_dir=patch_dir, transform=set_transforms(False))
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.M1_model_dir, exist_ok=True)
    if args.joint:
        M1_model_dir = os.path.join(args.M1_model_dir, 'joint_encoder_{}.pth'.format(round_id))
        pre_M1_model_dir = os.path.join(args.M1_model_dir, 'joint_encoder_{}.pth'.format(round_id - 1))
    else:
        M1_model_dir = os.path.join(args.M1_model_dir, 'encoder_{}.pth'.format(round_id))
        pre_M1_model_dir = os.path.join(args.M1_model_dir, 'encoder_{}.pth'.format(round_id - 1))
    # num_classes = (round_id + 1)*3 + 1

    model = Aux_Model(num_classes)
    model = model.to(device)
    if not os.path.exists(M1_model_dir):
        if os.path.exists(pre_M1_model_dir):
            pretrained_dict = torch.load(pre_M1_model_dir, map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if(k in model_dict and 'fc' not in k)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict,strict=False)
            # else:
            # model.load_state_dict(torch.load(pre_M1_model_dir, map_location='cpu'))
        # optimization
        optimizer = torch.optim.Adam(model.parameters(), lr=args.M1_lr, weight_decay=5e-4)
        early_stopping = EarlyStopping(model_path=M1_model_dir, patience=args.M1_patience, verbose=True)

        for m1_epoch in range(1, args.M1_epochs + 1):
            m1_train_epoch(round_id, m1_epoch, model, optimizer, train_loader, criterion, device, num_classes)
            val_loss, val_acc, _, _ = m1_pred(round_id, model, val_loader, criterion, device, num_classes, status='Val')
            early_stopping(m1_epoch, val_loss, model, val_acc)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
    model.load_state_dict(torch.load(M1_model_dir, map_location='cpu'))
    loss, acc, auc, f1 = m1_pred(round_id, model, train_loader1, criterion, device, num_classes, status='Train')
    draw_metrics(ts_writer, 'Train_M1', num_classes, loss, acc, auc, None, f1, round_id)
    loss, acc, auc, f1 = m1_pred(round_id, model, val_loader, criterion, device, num_classes, status='Val')
    draw_metrics(ts_writer, 'Val_M1', num_classes, loss, acc, auc, None, f1, round_id)
    # if args.test:
    #     test_dset_patch = new_obj['test_dset_patch']
    #     test_dset = M1Dataset(split=test_dset_patch, patch_dir=patch_dir, transform=set_transforms(False))
    #     test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    #     loss, acc, auc, f1 = m1_pred(round_id, model, test_loader, criterion, device, num_classes, status='Test')
    #     draw_metrics(ts_writer, 'Test', num_classes, loss, acc, auc, None, f1, round_id)

    end = time.time()
    print('M1 use time: ', end - start)


def extract_feature(args):
    print('-------------feature extracting starts------------')
    start = time.time()
    device = args.device
    data_eset = args.data_eset
    round_id = args.round_id
    num_classes = args.C_numbers
    # if round_id == 0:
    #     M1_model = Aux_Model(args.num_classes)
    # else:
    M1_model = Aux_Model(args.num_classes)
    # M1_model = timm.create_model('densenet121', pretrained=False, num_classes=4)
    if args.joint:
        M1_model_dir = os.path.join(args.M1_model_dir, 'joint_encoder_{}.pth'.format(round_id))
    else:
        M1_model_dir = os.path.join(args.M1_model_dir, 'encoder_{}.pth'.format(round_id))
    if os.path.exists(M1_model_dir):
        M1_model.load_state_dict(torch.load(M1_model_dir, map_location='cpu'), strict=False)
        print('loading checkpoints from ', M1_model_dir)
    # else:
    #     if args.pretrained_ckpt == 'random':
    #         print('using checkpoints from ImageNet')
    #     elif args.pretrained_ckpt == 'bt':
    #         M1_model = resnet50()
    #         M1_model.load_state_dict(torch.load('/data_sdc/wxn/HOMIL/bt.pth', map_location='cpu'), strict=True)
    #         print('using checkpoints from bt')
    #     elif args.pretrained_ckpt == 'mocov2':
    #         M1_model = resnet50()
    #         M1_model.load_state_dict(torch.load('/data_sdc/wxn/HOMIL/mocov2.pth', map_location='cpu'),
    #                                  strict=True)
    #         print('using checkpoints from mocov2')
    #     elif args.pretrained_ckpt == 'swav':
    #         M1_model = resnet50()
    #         M1_model.load_state_dict(torch.load('/data_sdc/wxn/HOMIL/swav.pth', map_location='cpu'), strict=True)
    #         print('using checkpoints from swav')
    #     elif args.pretrained_ckpt == 'cpath':
    #         M1_model = ctranspath()
    #         M1_model.head = nn.Identity()
    #         M1_model.load_state_dict(
    #             torch.load('/data_sdb/yra/HKUST/HOMIL/ctranspath.pth', map_location='cpu')['model'], strict=True)
    #         print('using checkpoints from cpath')
    #     elif args.pretrained_ckpt == 'WSIFT':
    #         M1_model = resnet50()
    #         M1_model.load_state_dict(torch.load('/data_sdc/wxn/WSI-finetuning-main/results/num_512_s1/sflod_3_checkpoint_backbone.pt', map_location='cpu'), strict=False)
    #         print('using checkpoints from WSI-FT')
    #     else:
    #         raise NotImplementedError


    model = M1_model.to(device)

    feat_dir = args.pretrained_feat_dir if round_id == 0 else os.path.join(args.feat_dir, 'round_{}'.format(round_id))
    os.makedirs(feat_dir, exist_ok=True)
    patch_dir = args.patch_dir
    test_patch_dir = args.patch_dir

    slide_dict = {}
    for data_name in data_eset:
        paths = sorted(data_eset[data_name])
        for path in paths:
            if 'test' in data_name:
                slide_dict.update({os.path.join(test_patch_dir, path): 'test'})
            elif 'val' in data_name:
                slide_dict.update({os.path.join(patch_dir, path): 'val'})
            else:
                slide_dict.update({os.path.join(patch_dir, path): 'train'})
    slide_paths = sorted(slide_dict.keys())
    with tqdm(total=len(slide_paths)) as pbar:
        for i, slide_path in enumerate(slide_paths):
            slide_name = os.path.basename(slide_path).split('.')[0]
            coord_dir = os.path.join(args.coord_dir, slide_name + '.h5')
            feat_path = os.path.join(feat_dir, '{}.pt'.format(slide_name))
            if os.path.exists(feat_path):
                pbar.update(1)
                continue

            if slide_dict.get(slide_path) == 'test':
                transform = set_transforms(is_train=False)
            else:
                transform = set_transforms(is_train=True)
            dset = Extract_Feat_Dataset(slide_path, transform=transform, img_format=args.img_format)
            if len(dset) == 0:
                pbar.update(1)
                continue
            loader = DataLoader(dset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False)
            features, coords = feat_extraction(model, loader, device)
            if not os.path.exists(coord_dir):
                f = h5py.File(coord_dir, 'w')
                f["coords"] = coords
                f.close()
            torch.save(features, feat_path)

            pbar.set_description('Round: {}, WSI: {}, with {} patches'.format(round_id, slide_name, len(dset)))
            pbar.update(1)

    end = time.time()
    print('feature extracting use time: ', end - start)
