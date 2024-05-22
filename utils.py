import numpy as np
import sys
import random
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn 
import torch.nn.functional as F 


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    consistency = 100
    consistency_rampup = 5
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
    

def update_ema_variables(model, ema_model, alpha=0.999, global_step=0):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_ema_threshold(x, ema_x, alpha=0.999, global_step=0):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    return ema_x * alpha + (1 - alpha) * x


def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark= False
    sys.setrecursionlimit(10000)


def set_transforms(is_train=True, is_Gray=False):
    if is_train:
        t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
        ])
        if is_Gray:
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            t.transforms.append(transforms.RandomGrayscale(p=0.3))
        else:
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        t.transforms.append(transforms.ToTensor())
        t.transforms.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    return t


class EarlyStopping:
    def __init__(self, model_path, patience=7, warmup_epoch=0, verbose=False, student_model_path=None):
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
        self.student_model_path = student_model_path
        
    def reset(self):
        self.counter = 0

    def __call__(self, epoch, val_loss, model, val_acc=None, student_model=None):
        flag = False
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, 'loss', student_model)
            self.counter = 0
            flag = True
        if val_acc is not None:
            if self.best_acc is None or val_acc >= self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, model, 'acc', student_model)
                self.counter = 0
                flag = True
        if flag:
            return self.counter
        self.counter += 1
        print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
        if self.counter >= self.patience and epoch > self.warmup_epoch:
            self.early_stop = True
        return self.counter

    def save_checkpoint(self, score, model, status='loss', student_model=None):
        """Saves model when validation loss or validation acc decrease."""
        if status == 'loss':
            pre_score = self.val_loss_min
            self.val_loss_min = score
        else:
            pre_score = self.val_acc_max
            self.val_acc_max = score
        torch.save(model.state_dict(), self.model_path)
        if student_model is not None:
            torch.save(student_model.state_dict(), self.student_model_path)
        if self.verbose:
            print('Valid {} ({} --> {}).  Saving model ...{}'.format(status, pre_score, score, self.model_path))

def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes, confusion_mat=False):
    targets = targets.numpy()
    _, pred = torch.max(logits, dim=1)
    pred = pred.numpy()
    acc = accuracy_score(targets, pred)
    f1 = f1_score(targets, pred, average='macro')

    probs = F.softmax(logits, dim=1)
    probs = probs.numpy()
    if len(np.unique(targets)) != num_classes:
        roc_auc = 0
    else:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            binary_labels = label_binarize(targets, classes=[i for i in range(num_classes)])
            valid_classes = np.where(np.any(binary_labels, axis=0))[0]
            binary_labels = binary_labels[:, valid_classes]
            valid_cls_probs = probs[:, valid_classes]
            fpr, tpr, _ = roc_curve(y_true=binary_labels.ravel(), y_score=valid_cls_probs.ravel())
            roc_auc = auc(fpr, tpr)
    if confusion_mat:
        mat = confusion_matrix(targets, pred)
        return acc, f1, roc_auc, mat
    return acc, f1, roc_auc


def plot_confusion_matrix(cmtx, num_classes, class_names=None, title='Confusion matrix', normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cmtx = cmtx.astype('float') / cmtx.sum(axis=1)[:, np.newaxis]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure()
    plt.imshow(cmtx, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    fmt = '.2f' if normalize else 'd'
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        plt.text(j, i, format(cmtx[i, j], fmt), horizontalalignment="center",
                 color="white" if cmtx[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure



def prepare_data(df, case_id, label_dict=None):
    df_case_id = df['case_id'].tolist()
    df_slide_id = df['slide_id'].tolist()
    df_label = df['label'].tolist()

    slide_id = []
    label = []
    for case_id_ in case_id:
        idx = df_case_id.index(case_id_)
        slide_id.append(df_slide_id[idx])
        label_ = df_label[idx]
        if label_dict is None:
            label.append(int(label_))
        else:
            label.append(label_dict[label_])
    return slide_id, label


def return_splits(csv_path, label_dict=None, label_csv=None, test=False):
    split_df = pd.read_csv(csv_path)
    train_id = split_df['train'].dropna().tolist()
    val_id = split_df['val'].dropna().tolist()
    if test:
        test_id = split_df['test'].dropna().tolist()
    if label_csv is None:
        train_label = split_df['train_label'].dropna().tolist()
        train_label = list(map(int, train_label))
        val_label = split_df['val_label'].dropna().tolist()
        val_label = list(map(int, val_label))
        if test:
            test_label = split_df['test_label'].dropna().tolist()
            test_label = list(map(int, test_label))
    else:
        df = pd.read_csv(label_csv)
        train_id, train_label = prepare_data(df, train_id, label_dict)
        val_id, val_label = prepare_data(df, val_id, label_dict)
        if test:
            test_id, test_label = prepare_data(df, test_id, label_dict)

    train_split = dict(zip(train_id, train_label))
    val_split = dict(zip(val_id, val_label))
    if test:
        test_split = dict(zip(test_id, test_label))
        return train_split, val_split, test_split
    return train_split, val_split


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



class LabelStagingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, low_smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelStagingCrossEntropy, self).__init__()
        assert low_smoothing < 1.0
        self.low_smoothing = low_smoothing
        self.confidence = 1.
        
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        loss = self.confidence * nll_loss

        if int(target) != 0: # low staging
            low_smooth_loss = -logprobs[:, :target].mean(dim=-1)
            loss = loss + self.low_smoothing * low_smooth_loss
        return loss.mean()