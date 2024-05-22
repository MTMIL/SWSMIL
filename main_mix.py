import os
import glob
import argparse
import timm
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.MIL_models import ABMIL, CLAM_MB, CLAM_SB, AttentionGated, TransMIL, MeanMIL, MaxMIL, DSMIL
from utils import *
from torch.autograd import Variable

def draw_metrics(ts_writer, name, num_class, loss, acc, auc, mat, f1, step):
    ts_writer.add_scalar("{}/loss".format(name), loss, step)
    ts_writer.add_scalar("{}/acc".format(name), acc, step)
    ts_writer.add_scalar("{}/auc".format(name), auc, step)
    ts_writer.add_scalar("{}/f1".format(name), f1, step)
    if mat is not None:
        ts_writer.add_figure("{}/confusion mat".format(name),
                             plot_confusion_matrix(cmtx=mat, num_classes=num_class), step)

def draw_metrics2(ts_writer, name, loss, round, step):
    ts_writer.add_scalar("{}/loss_round{}".format(name, round), loss, step)


split_times = 0

def get_args():
    parser = argparse.ArgumentParser(description='MIL main parameters')

    # general params.
    parser.add_argument('--experiment_name', type=str, default='testfast-10-2', help='experiment name')
    parser.add_argument('--MIL_model', type=str, default='GatedABMIL',
                        choices=['ABMIL', 'CLAM_SB', 'CLAM_MB', 'TransMIL', 'GatedABMIL', 'MeanMIL', 'MaxMIL', 'DSMIL', 'Stage_CLAM'],
                        help='MIL model to use')
    parser.add_argument('--metric2save', type=str,
                        # default='acc_auc',  
                        default='f1_auc', 
                        choices=['acc', 'f1', 'auc', 'acc_auc', 'f1_auc', 'loss'],
                        help='metrics to save best model')
    parser.add_argument('--device_ids', type=str, default=6, help='gpu devices for training')
    parser.add_argument('--seed', type=int, default=3701, help='random seed')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--num_classes', type=int,
                        default=2,
                        help='classification number')

    parser.add_argument('--rounds', type=int, default=5, help='rounds to train')

    # MIL params.
    parser.add_argument('--split_data', action='store_false', help='use data split')
    parser.add_argument('--mean_teacher', action='store_false', help='use mean teacher')
    parser.add_argument('--ada_ssl_th', action='store_false', help='use adaptive ssl threshold')
    parser.add_argument('--consistency_type', type=str, default='mse', choices=['mse', 'kl'], help='criterion type')
    parser.add_argument('--pseudo_label_num', type=int, default=4, help='max pseudo bag number')
    parser.add_argument('--max_pseudo_num', type=int, default=6, help='max pseudo bag number')
    parser.add_argument('--pseudo_step', type=int, default=2, help='pseudo bag number step')
    parser.add_argument('--max_merge_num', type=int, default=1, help='max bag number to merge')
    parser.add_argument('--Mix', type=int, default=1, help='Use Mixup')
    parser.add_argument('--Noise', type=int, default=0, help='Use Noise')
    parser.add_argument('--Merge', type=int, default=0, help='Use Merge')
    parser.add_argument('--metrics', type=str, default='attn', choices=['random', 'attn', 'shap', 'cont'],
                        help='metrcics to sort')
    parser.add_argument('--epochs', type=int, default=200, help='MIL epochs to train')
    parser.add_argument('--patience', type=int, 
                        default=10, 
                        help='MIL epochs to early stop')
    parser.add_argument('--lr_patience', type=int, 
                        default=7, 
                        help='MIL epochs to adjust lr')
    parser.add_argument('--max_lr', type=float, 
                        default=3e-4, 
                        help='MIL max learning rate')
    parser.add_argument('--min_lr', type=float,
                        default=1e-4,
                        help='MIL min learning rate')

    # dir params.
    parser.add_argument('--ckpt_dir', type=str,
                        default='./ckpt/debug3',
                        help='dir to save models')
    parser.add_argument('--instance_label_dir', type=str,
                        default='.data/MILs/c16/label',
                        help='dir to save models')
    parser.add_argument('--csv_dir', type=str,
                        default='./csv/c16', #  ./csv/bracs/3cls
                        help='csv dir to load data')
    parser.add_argument('--feat_dir', type=str,
                        # default='/data_sdb/yra/HKUST/BRACS/data/5x/feat0',
                        default='./data/MILs/c16/feat0',
                        help='train/val/test dir for feat/coord')
    parser.add_argument('--test', action='store_false', help='use test dataset')
    parser.add_argument('--ts_dir', type=str,
                        default='./logger/work/c16',
                        help='tensorboard dir')
    parser.add_argument('--ada_pse', type=int,
                        default=1,
                        help='tensorboard dir')
    parser.add_argument('--testmode', type=int,
                        default=1,
                        help='tensorboard dir')
    
    args = parser.parse_args()
    return args



class MILDataset(Dataset):
    def __init__(self, split, feat_dir, ins_label_dir=None):
        self.slide_ids = list(split.keys())
        self.labels = list(split.values())
        self.feat_dir = feat_dir
        self.feat_files = self.get_feat()
        self.ins_label_dir = ins_label_dir

    def get_labels(self):
        return self.labels

    def get_feat(self):
        feat_files = {}
        for slide_id in self.slide_ids:
            feat_paths = glob.glob(os.path.join(self.feat_dir, slide_id + '.pt'))
            slide_feats = []
            for feat_path in feat_paths:
                slide_feats.append(feat_path)
            feat_files[slide_id] = slide_feats
        return feat_files

    def __getitem__(self, idx):
        slide_name = self.slide_ids[idx]
        target = self.labels[idx]
        feat_files = self.feat_files[slide_name]
        feats = torch.Tensor()
        ins_labels = []
        for feat_file in feat_files:
            feat = torch.load(feat_file, map_location='cpu')
            feats = torch.cat((feats, feat), dim=0)
            if self.ins_label_dir is not None:
                ins_label_path = os.path.join(self.ins_label_dir, slide_name + '.npy')
                if os.path.exists(ins_label_path):
                    ins_label = np.load(ins_label_path)
                else:
                    ins_label = np.zeros(feat.shape[0])
                ins_label = torch.Tensor(ins_label)
                ins_labels.append(ins_label)

        if self.ins_label_dir is not None:
            ins_labels = torch.stack(ins_labels)
            sample = {'slide_id': slide_name, 'feat': feats, 'target': target, 'ins_label': ins_labels[0]}
        else:
            sample = {'slide_id': slide_name, 'feat': feats, 'target': target}
        return sample

    def __len__(self):
        return len(self.slide_ids)



class MT_Dataset(Dataset):
    def __init__(self, slide_ids, feat_dir, split, MT=False, merge_num=1, ins_label_dir=None, mix=False, num_class=2):
        self.slide_ids = slide_ids 
        self.feat_dir = feat_dir
        self.split = split
        self.MT = MT
        self.merge_num = merge_num
        self.ins_label_dir = ins_label_dir
        self.pse_slide_ids, self.pse_labels, self.labeled_feat_files = self.get_labeled_feat()
        self.mix = mix
        self.num_class = num_class

    def __len__(self):
        return len(self.pse_slide_ids)  
    
    def get_labeled_feat(self):
        pse_slide_ids = []
        pse_labels = []
        pse_truth_labels = []
        labeled_feat_files = {-1: []}
        for i in np.unique(list(self.slide_ids.values())):
            labeled_feat_files.update({i: []})
        for i in list(self.slide_ids.keys()): 
            pseudo_num = len(self.split.get(i)["data"]) if self.split is not None else 1 
            for j in range(pseudo_num):
                pse_slide_id = '{}-pse{}'.format(i, j)
                pse_slide_ids.append(pse_slide_id)
                if self.split is None:
                    pse_labels.append(self.slide_ids.get(i))
                    if self.ins_label_dir is not None:
                        pse_truth_labels.append(self.slide_ids.get(i))
                    labeled_feat_files[self.slide_ids.get(i)] += [pse_slide_id]
                else:
                    try:    
                        if self.ins_label_dir is not None:
                            pse_truth_labels.append(self.split.get(i)["label"][j])
                        pse_labels.append(self.split.get(i)["label"][j])
                        labeled_feat_files[self.split.get(i)["label"][j]] += [pse_slide_id]
                    except:
                        print(len(self.split.get(i)["data"]))
                        print(self.split.get(i)["label"])
        return pse_slide_ids, pse_labels, labeled_feat_files
        
    def __getitem__(self, idx):
        slide_name, pse_name = self.pse_slide_ids[idx].split('-pse')
        target = self.pse_labels[idx]
        feat_path = os.path.join(self.feat_dir, '{}.pt'.format(slide_name))
        feats = torch.load(feat_path, map_location='cpu')
        if self.MT:
            if self.split is not None:
                pse_index = self.split.get(slide_name)["data"][int(pse_name)]  
                feats = feats[pse_index, :]
            ema_feats = feats

            if self.split is not None:
                for _ in range(self.merge_num):
                    # feature augmentation
                    if target > 0:
                        random_label = random.randint(0, target - 1) 
                    else:
                        random_label = 0
                    merge_pse_slide_id = random.sample(self.labeled_feat_files.get(random_label), 1)[0]  
                    merge_slide_name, merge_pse_name = merge_pse_slide_id.split('-pse')

                    merge_pse_index = self.split.get(merge_slide_name)["data"][int(merge_pse_name)]  
                    merge_feat_path = os.path.join(self.feat_dir, '{}.pt'.format(merge_slide_name))
                    merge_feats = torch.load(merge_feat_path, map_location='cpu')
                    bag_feat = merge_feats[merge_pse_index, :]
                    feats = torch.cat((feats, bag_feat), dim=0) 
            sample = {'slide_id': slide_name, 'feat': feats, 'ema_feat': ema_feats, 'target': target}  
        elif self.mix:
            new_target = [0] * self.num_class
            if target >= 0:
                new_target[target] = 1
            if self.split is not None:
                pse_index = self.split.get(slide_name)["data"][int(pse_name)]  
                feats = feats[pse_index, :]
            ema_feats = feats
            if self.split is not None:
                for _ in range(self.merge_num):
                    # feature augmentation
                    if target > 0:
                        random_label = random.randint(0, target)
                    else:
                        random_label = 0
                    merge_pse_slide_id = random.sample(self.labeled_feat_files.get(random_label), 1)[0]  
                    merge_slide_name, merge_pse_name = merge_pse_slide_id.split('-pse')

                    merge_pse_index = self.split.get(merge_slide_name)["data"][int(merge_pse_name)]  
                    merge_feat_path = os.path.join(self.feat_dir, '{}.pt'.format(merge_slide_name))
                    merge_feats = torch.load(merge_feat_path, map_location='cpu')
                    bag_feat = merge_feats[merge_pse_index, :]
                    feats = torch.cat((feats, bag_feat), dim=0) 
                    new_target[random_label] += 1
                new_target = [x / (self.merge_num + 1) for x in new_target]
                # new_target = torch.Tensor(new_target)
            new_target = torch.Tensor(new_target)
            sample = {'slide_id': slide_name, 'feat': feats, 'ema_feat': ema_feats, 'target': new_target}  #需要返回的值
        else:
            if self.split is not None:
                pse_index = self.split.get(slide_name)["data"][int(pse_name)]  
                feats = feats[pse_index, :]
            sample = {'slide_id': slide_name, 'feat': feats, 'target': target}  
        return sample



def model_infer(model_suffix, model, feat, label, criterion, ema=False, mix=False):
    if 'ABMIL' in model_suffix:
        logit, _ = model(feat)
        if ema:
            logit = Variable(logit.detach().data, requires_grad=False)
        if mix:
            try:
                loss = criterion(logit, label.float())
            except:
                c2 = nn.CrossEntropyLoss(ignore_index=-1)
                loss = c2(logit, label.long())
        else:
            if int(label) < 0:
                loss = None
            else:
                loss = criterion(logit, label.long())
    elif 'DSMIL' in model_suffix:
        bag_weight = 0.5
        patch_pred, logit, _, _ = model(feat)
        if ema:
            logit = Variable(logit.detach().data, requires_grad=False)
        if int(label) < 0:
            loss = None
        else:
            patch_pred, _ = torch.max(patch_pred, 0)
            loss = bag_weight * criterion(patch_pred.view(1, -1), label.long()) \
                    + (1 - bag_weight) * criterion(logit, label.long())
    elif 'CLAM' in model_suffix:
        bag_weight = 0.7
        logit, _, instance_dict = model(feat, label, instance_eval=True)
        if ema:
            logit = Variable(logit.detach().data, requires_grad=False)
        if int(label) < 0:
            loss = None
        else:
            instance_loss = instance_dict['instance_loss']
            loss = bag_weight * criterion(logit, label.long()) + (1 - bag_weight) * instance_loss
    else:
        logit = model(feat)
        if ema:
            logit = Variable(logit.detach().data, requires_grad=False)
        if int(label) < 0:
            loss = None
        else:
            loss = criterion(logit, label.long())
    return loss, logit


def MIL_train_epoch(round_id, epoch, model, optimizer, loader, criterion, device, num_classes, model_suffix='ABMIL',
                    ema_model=None, cst_criterion=None, global_step=0, mix=False, use_noise=False, strength=0.05):
    global class_th
    _sample_th = np.array([0.] * num_cls)
    _sample_num = np.array([0] * num_cls)
    model.train()
    if ema_model is not None:
        ema_model.train()
    loss_all = 0.
    consistency_loss_all = 0.
    ema_loss_all = 0.
    logits = torch.Tensor()
    labels = torch.Tensor()
    with tqdm(total=len(loader)) as pbar:
        for i, sample in enumerate(loader):

            optimizer.zero_grad()
            if ema_model is None:
                slide_ids, feat, target = sample['slide_id'], sample['feat'], sample['target']
            else:
                slide_ids, feat, ema_feat, target = sample['slide_id'], sample['feat'], sample['ema_feat'], sample['target']
                ema_feat = ema_feat.to(device)
            feat = feat.to(device)
            label = target.to(device)
            loss, logit = model_infer(model_suffix, model, feat, label, criterion, mix=mix)
            if ema_model is not None:
                if use_noise:
                    noise = torch.randn_like(feat).to(device)
                    noise = (1 + strength) * noise / noise.abs().max()
                    feat = torch.mul(noise, feat.clone().detach())
                ema_loss, ema_logit = model_infer(model_suffix, ema_model, ema_feat, label, criterion, mix=mix)
                if not mix:
                    if class_th is not None and label != -1:
                        ema_probs = F.softmax(ema_logit, dim=1)
                        ema_probs = ema_probs.cpu().numpy()
                        
                        _sample_th[int(label)] = _sample_th[int(label)] + ema_probs[0, int(label)]
                        _sample_num[int(label)] = _sample_num[int(label)] + 1
                    if ema_loss is not None:
                        ema_loss_all += ema_loss * len(label)
                if cst_criterion is not None:
                    consistency_weight = get_current_consistency_weight(epoch)
                    consistency_loss = 0.3 * consistency_weight * cst_criterion(logit, ema_logit)
                    if loss is None:
                        loss = consistency_loss
                    else:
                        loss = loss + consistency_loss
                    consistency_loss_all += consistency_loss.detach().item() * len(label)

            # calculate metrics
            logits = torch.cat((logits, logit[torch.where(label>=0)[0]].detach().cpu()), dim=0)
            labels = torch.cat((labels, label[torch.where(label>=0)[0]].cpu()), dim=0)
            if loss is not None:
                loss_all += loss.detach().item() * len(label)
                # loss backward
                loss.backward()
                optimizer.step()
            if ema_model is not None:
                global_step += 1
                update_ema_variables(model, ema_model, 0.999, global_step)

            lr = optimizer.param_groups[0]['lr']
            # if ema_model is not None:
            #     if len(labels) > 0:
            #         acc, f1, roc_auc = calculate_metrics(logits, labels, num_classes)
            #         pbar.set_description(
            #             '[Round:{}, Epoch:{}] lr:{:.5f}, loss:{:.4f}, consist loss:{:.4f}, ema loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
            #                 .format(round_id, epoch, lr, loss_all / len(labels), consistency_loss_all / len(labels), ema_loss_all / len(labels), acc, roc_auc, f1))
            # else:
            #     acc, f1, roc_auc = calculate_metrics(logits, labels, num_classes)
            #     pbar.set_description(
            #         '[Round:{}, Epoch:{}] lr:{:.5f}, loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
            #             .format(round_id, epoch, lr, loss_all / len(labels), acc, roc_auc, f1))
            try:
                pbar.set_description(
                        '[Round:{}, Epoch:{}] lr:{:.5f}, loss:{:.4f}'
                            .format(round_id, epoch, lr, loss_all / len(labels)))
            except:
                pbar.set_description(
                        '[Round:{}, Epoch:{}] lr:{:.5f}, loss:{:.4f}'
                            .format(round_id, epoch, lr, loss_all))
            pbar.update(1)
    
    _class_th = _sample_th / _sample_num
    class_th = update_ema_threshold(_class_th, class_th, 0.999, global_step)
    if mix:
        return loss_all / len(labels)
    acc, f1, roc_auc, mat = calculate_metrics(logits, labels, num_classes, confusion_mat=True)
    print('acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'.format(acc, roc_auc, f1))
    print(mat)
    return loss_all / len(labels)


def MIL_pred(round_id, model, loader, criterion, device, num_classes, model_suffix='ABMIL', status='Val', mix=False):
    model.eval()
    loss_all = 0.
    logits = torch.Tensor()
    targets = torch.Tensor()
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, sample in enumerate(loader):
                slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
                if len(feat[0]) == 0:
                    pbar.update(1)
                    continue
                feat = feat.to(device)
                target = target.to(device)
                if 'ABMIL' in model_suffix:
                    logit, _ = model(feat)
                elif 'DSMIL' in model_suffix:
                    _, logit, _, _ = model(feat)
                elif 'CLAM' in model_suffix:
                    logit, _, _ = model(feat, target)
                else:
                    logit = model(feat)

                # calculate metrics
                loss = criterion(logit, target.long())
                logits = torch.cat((logits, logit.detach().cpu()), dim=0)
                targets = torch.cat((targets, target.cpu()), dim=0)
                loss_all += loss.item() * len(target)
                acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)

                pbar.set_description('[{} Round:{}] loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                     .format(status, round_id, loss_all / len(targets), acc, roc_auc, f1))
                pbar.update(1)
    acc, f1, roc_auc, mat = calculate_metrics(logits, targets, num_classes, confusion_mat=True)
    print(mat)
    return loss_all / len(targets), acc, roc_auc, f1, mat



def split_slide_data(split_num, model, model_path, loader, device, model_suffix='ABMIL', prev_split=None,
                     metrics='attn', ada_pse=False, ada_threshold=0.9, hard_target_num=2, ada_ssl_th=1.0,
                     instance=False, num_classes=2, writer=None, cri=None, round_id=0):
    global class_th
    global split_times
    if class_th is not None:
        print(class_th)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        metrics='random'
    model.eval()
    split = {}
    ori_total_num = 0
    total_num = 0
    logits = torch.Tensor()
    targets = torch.Tensor()
    targets_list = []
    pse_label = []
    loss_all = 0.
    num = 0
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, sample in enumerate(loader):
                slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
                if instance:
                    inc_label = sample['ins_label']
                    inc_label = inc_label.to(device)
                feat = feat.to(device)
                target = target.to(device)
                if split_num > feat.shape[1] / 2:
                    _split_num = int(feat.shape[1] / 2)
                else:
                    _split_num = split_num
                if metrics == 'random':
                    attn_index = np.arange(feat.shape[1])
                    np.random.shuffle(attn_index)
                else:
                    if prev_split is None:
                        if 'ABMIL' in model_suffix:
                            _, attn = model(feat)
                        elif 'CLAM' in model_suffix:
                            _, attn, _ = model(feat, target)
                        else:
                            attn = np.zeros((1, feat.shape[1]))
                    else:
                        attn = np.zeros((1, feat.shape[1]))
                        pre_split_index = prev_split[slide_id[0]]["data"]
                        for i in range(len(pre_split_index)):
                            idx = pre_split_index[i]
                            np.random.shuffle(idx)
                            bag_feat = feat[:, idx, :]
                            if 'ABMIL' in model_suffix:
                                _, bag_attn = model(bag_feat)
                            elif 'CLAM' in model_suffix:
                                _, bag_attn, _ = model(bag_feat, target)
                            else:
                                bag_attn = np.zeros((1, feat.shape[1]))
                            attn[:, idx] = bag_attn[0]
                    score = attn[0]
                    attn_index = np.argsort(-score)
                    search_num = int(min(_split_num * 10, len(score) / 2))
                    if metrics == 'shap':
                        search_indices = attn_index[:search_num]
                        shap = shapley_value(search_indices, feat, target, model, device, model_suffix)
                        ptopk_indices = search_indices[np.argsort(-shap)]
                        left_indices = attn_index[search_num:]
                        attn_index = ptopk_indices.tolist() + left_indices.tolist()
                split_index = [attn_index[i::_split_num] for i in range(_split_num)] # 每隔_split_num个去一个
                if ada_pse:
                    ada_split_index = []
                    remove_split_list = []
                    pseudo_label_list = []
                    for i in range(len(split_index)):
                        idx = split_index[i]
                        np.random.shuffle(idx)
                        bag_feat = feat[:, idx, :]

                        if 'ABMIL' in model_suffix:
                            bag_logit, _ = model(bag_feat)
                        elif 'DSMIL' in model_suffix:
                            _, bag_logit, _, _ = model(feat)
                        elif 'CLAM' in model_suffix:
                            bag_logit, _, _ = model(bag_feat, target)
                        else:
                            bag_logit = model(feat)
                        _, bag_pred = torch.max(bag_logit, dim=1)
                        bag_prob = F.softmax(bag_logit, dim=1)
                        if instance:
                            bag_label_truth = inc_label[:, idx]
                            _, bag_label_pred = torch.max(bag_label_truth, dim=1)
                            logits = torch.cat((logits, bag_logit.detach().cpu()), dim=0)
                            targets = torch.cat((targets, torch.tensor([0 if bag_label_pred.long().cpu() < 0.5 else 1])), dim=0)
                            targets_list.append(0 if bag_label_pred.long().cpu() < 0.5 else 1)
                            # loss = cri(bag_logit, torch.tensor([0. if bag_label_pred.long().cpu() < 0.5 else 1.]))
                            # try:
                            # loss_all += loss.item() * len(feat)
                            # num += len(feat)
                            # except:
                            #     pass
                                    
                        if bag_pred != target and torch.max(bag_prob) >= ada_threshold:
                            remove_split_list.extend(idx)
                        else:
                            ada_split_index.append(idx)
                            if i < hard_target_num:
                                pseudo_label_list.append(int(target))
                            else:
                                if ada_ssl_th >= 1.0 and bag_pred == target:
                                    pseudo_label_list.append(int(target))
                                if bag_pred == target and torch.max(bag_prob) >= ada_ssl_th:
                                    pseudo_label_list.append(int(target))
                                else:
                                    pseudo_label_list.append(-1)
                                    
                                if bag_prob[0, int(target)] >= class_th[int(bag_pred)]:
                                    pseudo_label_list.append(int(target))
                                else:
                                    pseudo_label_list.append(-1)
                    if len(remove_split_list) > 0:
                        ada_num = len(ada_split_index)
                        if ada_num > 0:
                            np.random.shuffle(remove_split_list)
                            remove_split_index = [remove_split_list[i::ada_num] for i in range(ada_num)]
                            for i in range(len(ada_split_index)):
                                ada_split_index[i] = np.concatenate((ada_split_index[i], remove_split_index[i]))
                        else:
                            ada_split_index = [remove_split_list]
                            pseudo_label_list.append(int(target))
                    split[slide_id[0]] = {"data": [x for x in ada_split_index if len(x) > 0], "label": pseudo_label_list}
                    pse_label.extend(pseudo_label_list)
                else:
                    split[slide_id[0]] = {"data": split_index, "label": [int(target)] * _split_num}
                    for idx in split_index:
        
                        bag_label_truth = inc_label[:, idx]
                        _, bag_label_pred = torch.max(bag_label_truth, dim=1)
                        targets_list.append(0 if bag_label_pred.long().cpu() < 0.5 else 1)
                        pse_label.append(int(target.long().cpu()))
                
                
                
                total_num = total_num + len(split[slide_id[0]]["data"])
                ori_total_num = ori_total_num + _split_num
                pbar.set_description(
                    '[Pseudo Bag Splitting] total bag num:{}/{}, current bag num:{}/{}'
                        .format(total_num, ori_total_num, len(split[slide_id[0]]["data"]), _split_num))
                pbar.update(1)
    try:
        if instance:
            acc = 0
            num = 0
            pse_bag_num = 0
            pse_bag_num2 = 0 
            for label, target in zip(targets_list, pse_label):
                if target != -1:
                    pse_bag_num += 1
                    num += 1
                    if label == target:
                        acc += 1
                else:
                    pse_bag_num2 += 1    
            writer.add_scalar("Train_pse_truth_acc", acc/num, split_times)   
            writer.add_scalar("Train_pse_truth_w_label", pse_bag_num, split_times)      
            writer.add_scalar("Train_pse_truth_wo_label", pse_bag_num2, split_times)           
            # acc, f1, roc_auc, mat = calculate_metrics(logits, targets, num_classes, confusion_mat=True)
            # print(mat)
            # draw_metrics(writer, 'Train_pse', num_classes, 0, acc, roc_auc, mat, f1, split_times)
            writer.add_scalar("round", round_id, split_times)
            split_times += 1
    except:
        pass
    return split

if __name__ == '__main__':
    args = get_args()
    args.store_false = False
    # set device
    device = torch.device('cuda:{}'.format(args.device_ids))
    print('Using GPU ID: {}'.format(args.device_ids))

    # set random seed
    set_seed(args.seed)
    print('Using Random Seed: {}'.format(str(args.seed)))

    # set tensorboard
    args.ts_dir = os.path.join(args.ts_dir, args.experiment_name)
    os.makedirs(args.ts_dir, exist_ok=True)
    writer = SummaryWriter(args.ts_dir)
    print('Set Tensorboard: {}'.format(args.ts_dir))

    csv_path = os.path.join(args.csv_dir, 'Fold_{}.csv'.format(args.fold))  # dir to save label
    if args.test:
        train_dataset, val_dataset, test_dataset = return_splits(csv_path=csv_path, test=True)
        args.dataset = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    else:
        train_dataset, val_dataset = return_splits(csv_path=csv_path, test=False)
        args.dataset = {'train': train_dataset, 'val': val_dataset}

    feat_dir = args.feat_dir
    ins_label_dir = args.instance_label_dir   
    split = None
    split_num = 0
    # train_dset = MILDataset(train_dataset, feat_dir, split, args.max_pseudo_num)
    train_dset = MILDataset(train_dataset, feat_dir, ins_label_dir=ins_label_dir)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
    val_dset = MILDataset(val_dataset, feat_dir, ins_label_dir=ins_label_dir)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)
    split_dset = MILDataset(train_dataset, feat_dir, ins_label_dir=ins_label_dir)
    split_loader = DataLoader(split_dset, batch_size=1, shuffle=False, num_workers=0)
    if args.test:
        test_dset = MILDataset(test_dataset, feat_dir, ins_label_dir=ins_label_dir)
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if args.Mix:
        mix_criterion = nn.BCEWithLogitsLoss()
    else:
        mix_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else: 
        raise NotImplementedError


    if args.pseudo_step >= args.max_pseudo_num:  
        split_list = [args.max_pseudo_num]
    else:
        split_list = list(range(0, args.max_pseudo_num + 1, args.pseudo_step))
        # split_list = list(range(args.pseudo_step, args.max_pseudo_num + 1, args.pseudo_step)) 
        if split_list[-1] != args.max_pseudo_num:
            split_list.append(args.max_pseudo_num)

    model_dir = os.path.join(args.ckpt_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    mode = args.MIL_model
    num_cls = args.num_classes
    ada_pse = True
    ada_ssl_th = 0.5 if args.ada_ssl_th else 1.0
    global class_th
    class_th = np.array([ada_ssl_th] * num_cls)
    for round_id in range(args.rounds):  
        def build_model(mode, num_cls, criterion=nn.CrossEntropyLoss()):
            if 'GatedABMIL' == mode:
                model = AttentionGated(n_classes=num_cls, L=512)
            elif 'ABMIL' == mode:
                model = ABMIL(n_classes=num_cls)
            elif 'DSMIL' == mode:
                model = DSMIL(n_classes=num_cls)
            elif 'CLAM_SB' == mode:
                model = CLAM_SB(size_arg="small", k_sample=8, n_classes=num_cls, instance_loss_fn=criterion)
            elif 'CLAM_MB' == mode:
                model = CLAM_MB(size_arg="small", k_sample=8, n_classes=num_cls, instance_loss_fn=criterion)
            elif 'MeanMIL' == mode:
                model = MeanMIL(n_classes=num_cls)
            elif 'MaxMIL' == mode:
                model = MaxMIL(n_classes=num_cls)
            else:
                model = TransMIL(n_classes=num_cls)
            return model
        model = build_model(mode, num_cls).to(device)
        model_path = os.path.join(model_dir, '{}_model_{}.pth'.format(mode, round_id))
        pre_model_path = os.path.join(model_dir, '{}_model_{}.pth'.format(mode, round_id - 1))
        if args.mean_teacher:
            ema_model = build_model(mode, num_cls).to(device)
            ema_model_path = os.path.join(model_dir, 'ema_{}_model_{}.pth'.format(mode, round_id))
            pre_ema_model_path = os.path.join(model_dir, 'ema_{}_model_{}.pth'.format(mode, round_id - 1))
            for param in ema_model.parameters():
                param.detach_()
        else:
            ema_model = None
        lr = args.max_lr
        if 'TransMIL' == mode:
            base_optim = timm.optim.RAdam(model.parameters(), lr=lr)
            optimizer = timm.optim.Lookahead(base_optim, k=5, alpha=0.5)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        if args.split_data:
            pse_list_idx = 0 if round_id == 0 else len(split_list) - 1
            # pse_list_idx = 0
            split_num = split_list[pse_list_idx]
        if not os.path.exists(model_path):
            if args.split_data and split_num > 1:
                if args.mean_teacher:
                    split = split_slide_data(split_num, ema_model, pre_ema_model_path, split_loader, device, mode,
                                            split, args.metrics, ada_pse, 0.9, args.pseudo_label_num, ada_ssl_th, instance=True,
                                            num_classes=num_cls, writer=writer, cri=criterion, round_id=round_id)
                else:
                    split = split_slide_data(split_num, model, pre_model_path, split_loader, device, mode,
                                            split, args.metrics, ada_pse, 0.9, args.pseudo_label_num, ada_ssl_th, instance=True,
                                            num_classes=num_cls, writer=writer, cri=criterion, round_id=round_id)
                    
            if args.split_data:
                model.reset()
                if args.mean_teacher:
                    ema_model.reset()
            
            train_dset = MT_Dataset(train_dataset, feat_dir, split, MT=args.Merge, merge_num=args.max_merge_num, mix=args.Mix, num_class=num_cls)
            train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
            
            if args.mean_teacher:
                early_stopping = EarlyStopping(model_path=ema_model_path, patience=args.patience, verbose=True, student_model_path=model_path)
            else:
                early_stopping = EarlyStopping(model_path=model_path, patience=args.patience, verbose=True)
            global_step = 0
            for epoch in range(args.epochs):
                train_loss = MIL_train_epoch(round_id, epoch, model,
                                            optimizer, train_loader,
                                            mix_criterion, device,
                                            num_cls, mode, 
                                            ema_model=ema_model, 
                                            cst_criterion=consistency_criterion, 
                                            global_step=global_step,
                                            mix=args.Mix,
                                            use_noise=False)
                if args.mean_teacher:
                    val_loss, val_acc, val_auc, val_f1, val_mat = MIL_pred(round_id, ema_model, val_loader, criterion,
                                                                           device, num_cls, mode, 'Val_Teacher')
                    if epoch % 10 == 0:
                        MIL_pred(round_id, model, val_loader, criterion, device, num_cls, mode, 'Val_Student')
                else:
                    val_loss, val_acc, val_auc, val_f1, val_mat = MIL_pred(round_id, model, val_loader, criterion,
                                                                           device, num_cls, mode)

                draw_metrics2(writer, 'Train_loss', train_loss, round_id, epoch)
                draw_metrics(writer, 'Val_process', num_cls, val_loss, val_acc, val_auc, val_mat, val_f1, epoch)
                def count_earlystop(epoch, early_stop, metric, val_loss, val_acc, val_auc, val_f1, model, model1=None):
                    if metric == 'acc':
                        counter = early_stop(epoch, val_loss, model, val_acc, model1)
                    elif metric == 'f1':
                        counter = early_stop(epoch, val_loss, model, val_f1, model1)
                    elif metric == 'auc':
                        counter = early_stop(epoch, val_loss, model, val_auc, model1)
                    elif metric == 'acc_auc':
                        counter = early_stop(epoch, val_loss, model, (val_acc + val_auc) / 2, model1)
                    elif metric == 'f1_auc':
                        counter = early_stop(epoch, val_loss, model, (val_f1 + val_auc) / 2, model1)
                    elif metric == 'loss':
                        counter = early_stop(epoch, val_loss, model, model1)
                    else:
                        raise NotImplementedError
                    return counter
                if args.mean_teacher:
                    counter = count_earlystop(epoch, early_stopping, args.metric2save, val_loss, val_acc, val_auc, val_f1, ema_model, model)
                else:
                    counter = count_earlystop(epoch, early_stopping, args.metric2save, val_loss, val_acc, val_auc, val_f1, model)
                if early_stopping.early_stop:
                    if split_num == split_list[-1]:
                        print('Early Stopping')
                        break
                # adjust learning rate
                if counter > 0 and counter % args.lr_patience == 0:
                    if lr == args.min_lr and args.split_data:
                        if split_num != split_list[-1]:
                            early_stopping.reset()
                            pse_list_idx = pse_list_idx + 1 if pse_list_idx < len(split_list) - 1 else pse_list_idx
                            split_num = split_list[pse_list_idx]

                        if split_num > 1:
                            if args.mean_teacher:
                                split = split_slide_data(split_num, ema_model, ema_model_path, split_loader, device, mode,
                                                         split, args.metrics, ada_pse, 0.9, args.pseudo_label_num, ada_ssl_th, instance=True,
                                            num_classes=num_cls, writer=writer, cri=criterion, round_id=round_id)
                            else:
                                split = split_slide_data(split_num, model, model_path, split_loader, device, mode,
                                                         split, args.metrics, ada_pse, 0.9, args.pseudo_label_num, ada_ssl_th, instance=True,
                                            num_classes=num_cls, writer=writer, cri=criterion, round_id=round_id)
                            train_dset = MT_Dataset(train_dataset, feat_dir, split, MT=args.mean_teacher, merge_num=args.max_merge_num)
                            train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
                    if lr > args.min_lr:
                        early_stopping.reset()
                        lr = lr / 10 if lr / 10 >= args.min_lr else args.min_lr
                        for params in optimizer.param_groups:
                            params['lr'] = lr
                    if ada_ssl_th < 0.9:
                        ada_ssl_th = ada_ssl_th + 0.1


        if args.mean_teacher:
            ema_model.load_state_dict(torch.load(ema_model_path, map_location='cpu'))
            val_sign = 'Val_Teacher'
            test_sign = 'Test_Teacher'
            val_loss, val_acc, val_auc, val_f1, val_mat = MIL_pred(round_id, ema_model, val_loader, criterion, device,
                                                                   num_cls, mode, val_sign)
            draw_metrics(writer, 'Val_Teacher', num_cls, val_loss, val_acc, val_auc, val_mat, val_f1, round_id)
            
            if args.test:
                test_loss, test_acc, test_auc, test_f1, test_mat = MIL_pred(round_id, ema_model, test_loader, criterion, device,
                                                                            num_cls, mode, test_sign)
                draw_metrics(writer, 'Test_Teacher', num_cls, test_loss, test_acc, test_auc, test_mat, test_f1, round_id)
            val_sign = 'Val_Student'
            test_sign = 'Test_Student'
        else:
            val_sign = 'Val'
            test_sign = 'Test'

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        val_loss, val_acc, val_auc, val_f1, val_mat = MIL_pred(round_id, model, val_loader, criterion, device,
                                                               num_cls, mode, val_sign)
        draw_metrics(writer, val_sign, num_cls, val_loss, val_acc, val_auc, val_mat, val_f1, round_id)
        
        if args.test:
            test_loss, test_acc, test_auc, test_f1, test_mat = MIL_pred(round_id, model, test_loader, criterion, device,
                                                                        num_cls, mode, test_sign)
            draw_metrics(writer, test_sign, num_cls, test_loss, test_acc, test_auc, test_mat, test_f1, round_id)
