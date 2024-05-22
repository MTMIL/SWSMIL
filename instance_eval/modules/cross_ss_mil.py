import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .mamba.mamba_ssm.modules.cross_mamba import Cross_Mamba
from .mamba.mamba_ssm.modules.mamba_simple import Mamba
import math
def rotational_positional_encoding(length, dim):
    # 初始化位置编码
    positional_encoding = torch.zeros((1, length, dim))

    # 计算每个位置的编码
    for pos in range(length):
        for i in range(0, dim, 2):
            positional_encoding[0, pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim)))
            if i + 1 < dim:
                positional_encoding[0, pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / dim)))

    return positional_encoding
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.model = list(models.resnet50(pretrained = True).children())[:-1]
        self.features = nn.Sequential(*self.model)

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.classifier = nn.Linear(512,1)
        initialize_weights(self.feature_extractor_part2)
        initialize_weights(self.classifier)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x=self.feature_extractor_part2(x)
        # feat = torch.mean(x,dim=0)
        x1 = self.classifier(x)
        # x2 = torch.mean(x1, dim=0).view(1,-1)
        x2,_ = torch.max(x1, dim=0)
        x2=x2.view(1,-1)
        return x2,x
class AttentionGated(nn.Module):
    def __init__(self,n_classes,input_dim=512,act='relu',bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(1024, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        Y_prob = self.classifier(x)

        return Y_prob

class CROSS_SS_MIL(nn.Module):
    def __init__(self,n_classes,dropout,act):
        super(CROSS_SS_MIL, self).__init__()
        self.feature = [nn.Linear(1024, 512)]
        self.POS_dim = 128
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.1)]

        self.feature = nn.Sequential(*self.feature)
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, 1024))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, 1024))
        self.simple_ssm = Mamba(d_model=512,d_state=128,d_conv=4,expand=1)
        self.simple_ssm_i = Mamba(d_model=512,d_state=128,d_conv=4,expand=1)
        self.cross_ssm = Cross_Mamba(d_model=512,d_state=128,d_conv=4,expand=1)
        self.cross_ssm_i = Cross_Mamba(d_model=512,d_state=128,d_conv=4,expand=1)
        self.fi_linear = nn.Sequential(
            nn.Linear(512, 1),
            nn.ReLU())
            # nn.Dropout(0.1)
        
        self.if_linear = nn.Sequential(
            nn.Linear(512,1),
            nn.Sigmoid())
            # nn.Dropout(0.1)))
        
        self.classifier = nn.Linear(1024,n_classes)

    def forward(self, x, return_attn=False,no_norm=False):
        # rot_pos = rotational_positional_encoding(x.shape[1], self.POS_dim).to(x.device)
        feature_f = self.feature(x)
        feature_i = torch.flip(feature_f, [1])
        feature_ssm_f = self.simple_ssm(feature_f)
        feature_ssm_i = self.simple_ssm(feature_i)
        feature_ssm_f = F.softmax(feature_ssm_f,dim=-1)
        feature_ssm_i = F.softmax(feature_ssm_i,dim=-1)
        feature_ssm_i = torch.flip(feature_ssm_i, [1])
        feature_ssm_fi = torch.cat((feature_ssm_f,feature_ssm_i),dim=2)
        feature_ssm_fi = torch.cat((feature_ssm_fi,self.cls_token1),dim=1)
        # hidden_pos = self.get_hidden_pos(feature_ssm_fi)
        feature_ssm_if = torch.cat((feature_ssm_i,feature_ssm_f),dim=2)
        feature_ssm_if = torch.cat((feature_ssm_if,self.cls_token2),dim=1)
        feature_ssm_fi = self.cross_ssm(feature_ssm_fi)
        feature_ssm_if = self.cross_ssm(feature_ssm_if)
        cls_token1 = feature_ssm_fi[:,-1,:]
        cls_token2 = feature_ssm_if[:,-1,:]
        cls_token = torch.cat((cls_token1,cls_token2),dim=-1).squeeze()
        # fi_score = self.fi_linear(feature_ssm_fi)
        # if_score = self.if_linear(feature_ssm_if)
        # ssm_score = fi_score.mul(if_score).squeeze(-1)
        # ssm_score = F.softmax(ssm_score,dim=-1)
        # feature_score = torch.mm(ssm_score,feature_f.squeeze(0))
        logits = self.classifier(cls_token).unsqueeze(0)
        # print(logits.shape)
        return logits



if __name__ == "__main__":
    mil_model = CROSS_SS_MIL(2,dropout=True,act='relu').cuda()
    x = torch.randn(1,1000,1024).cuda()
    y= mil_model(x)
    print(y.shape)
    print(y)


