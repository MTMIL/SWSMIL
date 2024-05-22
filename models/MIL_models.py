import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from models.backbone import resnet50_baseline


class MeanMIL(nn.Module):
    def __init__(self, n_classes=1, dropout=True, act='relu'):
        super(MeanMIL, self).__init__()
        head = [nn.Linear(1024, 512)]
        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]
        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512, n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def reset(self):
        initialize_weights(self)

    def forward(self, x):
        x = self.head(x).mean(axis=1)
        return x


class MaxMIL(nn.Module):
    def __init__(self, n_classes=1, dropout=True, act='relu'):
        super(MaxMIL, self).__init__()
        head = [nn.Linear(1024, 512)]
        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]
        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512, n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def reset(self):
        initialize_weights(self)

    def forward(self, x):
        x, _ = self.head(x).max(axis=1)
        return x


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, n_classes=2, num_feats=1024):
        super(DSMIL, self).__init__()
        self.i_classifier = FCLayer(num_feats, n_classes)
        self.b_classifier = BClassifier(input_size=num_feats, output_class=n_classes)
        initialize_weights(self)

    def forward(self, x):
        x = x.squeeze(0)
        feats, classes = self.i_classifier(x)
        logit, A, B = self.b_classifier(feats, classes)

        return classes, logit, A, B


class AttentionGated(nn.Module):
    def __init__(self, n_classes=2, act='relu', bias=False, dropout=False, L=512, D=128):
        super(AttentionGated, self).__init__()
        self.L = L
        self.D = D  # 128
        self.K = 1

        self.feature = [nn.Linear(1024, self.L)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D, bias=bias),
        ]
        if act == 'gelu':
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D, bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

        self.apply(initialize_weights)

    def reset(self):
        initialize_weights(self)

    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        Y_prob = self.classifier(x)

        return Y_prob, A.detach().cpu().numpy()


class Stage_CLAM(nn.Module):
    def __init__(self, n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(ignore_index=-1), k_sample=4):
        nn.Module.__init__(self)
        size = [1024, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        fc.append(Attn_Net_Gated(L=size[1], D=size[2], n_classes=n_classes))
        self.attention_net = nn.Sequential(*fc)
        
        bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], n_classes)]
        self.instance_classifiers = nn.Sequential(*instance_classifiers)
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.k_sample = k_sample
        initialize_weights(self)

    def reset(self):
        initialize_weights(self)

    @staticmethod
    def create_positive_targets(length, label, device):
        return torch.full((length,), int(label), device=device).long()

    def inst_eval(self, A, h, classifier, target):
        device = h.device
        if len(A) == 1:
            A = A.squeeze(0)

        inst_logits = classifier(h)
        inst_max_idx = torch.argwhere(torch.argmax(inst_logits, dim=-1)>int(target))

        left_idx = [i for i in range(A.shape[-1]) if i not in inst_max_idx.squeeze(-1)]

        sample = self.k_sample if int(A.shape[-1] / 3) >= self.k_sample else int(A.shape[-1] / 3)
        if sample > len(left_idx):
            sample = len(left_idx)
        _, top_p_ids = torch.topk(A[int(target), left_idx], sample, dim=-1)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_positive_targets(sample, target, device)

        if len(inst_max_idx) != 0:
            inst_p = torch.index_select(h, dim=0, index=inst_max_idx.squeeze(-1))
            inst_targets = self.create_positive_targets(len(inst_p), target, device)
            all_targets = torch.cat([p_targets, inst_targets], dim=0)
            all_instances = torch.cat([top_p, inst_p], dim=0)
        else:
            all_targets = p_targets
            all_instances = top_p
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets) / len(all_targets)
        return instance_loss, all_preds, all_targets
        

    def forward(self, h, label, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h = torch.squeeze(h)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        if attention_only:
            return A

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []

            instance_loss, preds, targets = self.inst_eval(A, h, self.instance_classifiers, label)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_inst_loss += instance_loss
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return logits, A.detach().cpu().numpy(), results_dict


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# def initialize_weights(module):
#     for m in module.modules():
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight)
#             m.bias.data.zero_()
#
#         elif isinstance(m, nn.BatchNorm1d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout > 0:
            self.module.append(nn.Dropout(dropout))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, only_A=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        if only_A:
            return A
        return A, x


class ABMIL(nn.Module):
    def __init__(self, n_classes=2, dropout=0.25):
        super(ABMIL, self).__init__()
        fc_size = [1024, 256]
        self.n_classes = n_classes
        self.path_attn_head = Attn_Net_Gated(L=fc_size[0], D=fc_size[1], dropout=dropout, n_classes=1)
        self.classifiers = nn.Linear(fc_size[0], n_classes)
        initialize_weights(self)

    def reset(self):
        initialize_weights(self)

    def forward(self, wsi_h):
        wsi_trans = wsi_h.squeeze(0)
        # Attention Pooling
        path = self.path_attn_head(wsi_trans, only_A=True)
        ori_path = path.view(1, -1)
        path = F.softmax(ori_path, dim=1)
        M = torch.mm(path, wsi_trans)  # all instance
        attn = path.detach().cpu().numpy()
        # ---->predict (cox head)
        logits = self.classifiers(M)

        return logits, attn


class Joint_ABMIL(nn.Module):
    def __init__(self, n_classes=2, dropout=0.25):
        super(Joint_ABMIL, self).__init__()
        fc_size = [1024, 256]
        self.n_classes = n_classes
        self.path_attn_head = Attn_Net_Gated(L=fc_size[0], D=fc_size[1], dropout=dropout, n_classes=1)
        self.path_attn_head_fixed = Attn_Net_Gated(L=fc_size[0], D=fc_size[1], dropout=dropout, n_classes=1)
        self.classifiers = nn.Linear(2 * fc_size[0], n_classes)

    def forward(self, wsi_h, fixed_wsi_h):
        wsi_trans = wsi_h.squeeze(0)
        path = self.path_attn_head(wsi_trans, only_A=True)
        ori_path = path.view(1, -1)
        path = F.softmax(ori_path, dim=1)
        M = torch.mm(path, wsi_trans)
        attn = path.detach().cpu().numpy()

        fixed_wsi_trans = fixed_wsi_h.squeeze(0)
        fixed_path = self.path_attn_head_fixed(fixed_wsi_trans, only_A=True)
        fixed_ori_path = fixed_path.view(1, -1)
        fixed_path = F.softmax(fixed_ori_path, dim=1)
        fixed_M = torch.mm(fixed_path, fixed_wsi_trans)
        fixed_attn = fixed_path.detach().cpu().numpy()

        # ---->predict (cox head)
        logits = self.classifiers(torch.cat(M, fixed_M))

        return logits, attn, fixed_attn


class Joint_Feat_Classifier(nn.Module):
    def __init__(self, n_classes=2):
        super(Joint_Feat_Classifier, self).__init__()
        fc_size = 2 * 1024
        self.classifiers = nn.Linear(fc_size, n_classes)

    def forward(self, patch_h):
        patch_h = patch_h.squeeze(0)
        fixed_patch_h = torch.zeros_like(patch_h)
        logits = self.classifiers(torch.cat((patch_h, fixed_patch_h), dim=1))
        return logits


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=2, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(ignore_index=-1), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def reset(self):
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A) == 1:
            A = A.squeeze(0)
        sample = max(self.k_sample, 1) if len(A) / 2 >= self.k_sample else max(int(len(A) / 2), 1)
        _, top_p_ids = torch.topk(A, sample, dim=0)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        _, top_n_ids = torch.topk(-A, sample, dim=0)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(sample, device)
        n_targets = self.create_negative_targets(sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A) == 1:
            A = torch.squeeze(A)
        sample = max(self.k_sample, 1) if len(A) / 2 >= self.k_sample else max(int(len(A) / 2), 1)
        _, top_p_ids = torch.topk(A, sample, dim=0)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = torch.squeeze(h)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = A.view(1, -1)
        if attention_only:
            return A.detach().cpu().numpy()
        A_raw = A.detach().cpu().numpy()
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval and label != -1:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        if instance_eval and label != -1:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': h.detach()})
        return logits, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=4, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(ignore_index=-1), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h = torch.squeeze(h)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A.detach().cpu().numpy()
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, A_raw, results_dict


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, h):
        device = h.device
        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        return logits


class Aux_Model(nn.Module):
    def __init__(self, n_classes):
        super(Aux_Model, self).__init__()
        fc_size = 1024
        self.backbone = resnet50_baseline(True)
        self.fc = nn.Linear(fc_size, n_classes)

    def forward(self, x, feat_only=False):
        feat = self.backbone(x)
        if feat_only:
            return feat
        logits = self.fc(feat)
        return logits, feat


class FT_ABModel(nn.Module):
    def __init__(self, n_classes, dropout=0.25):
        super(FT_ABModel, self).__init__()
        fc_size = [1024, 256]
        self.backbone = resnet50_baseline(True)
        self.path_attn_head = Attn_Net_Gated(L=fc_size[0], D=fc_size[1], dropout=dropout, n_classes=1)
        self.classifiers = nn.Linear(fc_size[0], n_classes)

    def forward(self, x):
        feat = self.backbone(x)
        wsi_trans = feat.squeeze(0)
        # Attention Pooling
        path = self.path_attn_head(wsi_trans, only_A=True)
        ori_path = path.view(1, -1)
        path = F.softmax(ori_path, dim=1)
        M = torch.mm(path, wsi_trans)  # all instance
        attn = path.detach().cpu().numpy()
        # ---->predict (cox head)
        logits = self.classifiers(M)

        return logits, attn
