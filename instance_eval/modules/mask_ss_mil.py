import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .mamba.mamba_ssm.modules.mamba_mask import Mamba_MASK
def generate_2d_rotary_positional_encodings(x, y, dim):
    """
    根据x和y坐标生成二维旋转位置编码，并对x和y进行归一化。

    参数:
    x: x坐标的张量，形状为(N,)。
    y: y坐标的张量，形状为(N,)。
    dim: 每个位置编码的维度大小。

    返回:
    形状为(N, dim)的位置编码张量。
    """
    assert dim % 4 == 0, "维度必须是4的倍数以确保旋转编码的完整性"
    
    # 对x和y进行归一化
    x_min, x_max = 100, 10000
    y_min, y_max = 5000, 10000
    
    x_normalized = (x - x_min) / (x_max - x_min)
    y_normalized = (y - y_min) / (y_max - y_min)
    
    N = x.size(0)
    encodings = torch.zeros(N, dim)
    
    # 使用归一化后的x和y坐标生成旋转编码
    for i in range(dim // 4):
        div_term = 10000 ** (2 * (i // 2) / dim)
        encodings[:, 4*i] = torch.sin(x_normalized / div_term)
        encodings[:, 4*i + 1] = torch.cos(x_normalized / div_term)
        encodings[:, 4*i + 2] = torch.sin(y_normalized / div_term)
        encodings[:, 4*i + 3] = torch.cos(y_normalized / div_term)
    
    return encodings
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

class MASK_SS_MIL(nn.Module):
    def __init__(self,n_classes):
        super(MASK_SS_MIL, self).__init__()

        self.map = nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Dropout(0.1))
        self.feature = [nn.Linear(1024, 512)]
        self.mamba_block1 = Mamba_MASK(d_model=512, d_state=128, d_conv=4, expand=2)
        self.mamba_block2 = Mamba_MASK(d_model=512, d_state=128, d_conv=4, expand=2)
        # self.cls_token1 = nn.Parameter(torch.randn(1, 1, 512))
        # self.cls_token2 = nn.Parameter(torch.randn(1, 1, 512))
        self.classifier = nn.Linear(512*2, n_classes)
        self.map2 = nn.Sequential(nn.Linear(1024,1),nn.GELU(),nn.Dropout(0.1))
        self.norm = nn.LayerNorm(512*2)
        self.classifier = nn.Linear(512, n_classes)
    def forward(self, x,h5_path,return_attn=False,no_norm=False):
        x = self.map(x)
        B,L,D = x.shape
        # x_p = torch.randn([L,]).to(x.device)
        # y_p = torch.randn([L,]).to(x.device)
        # rp = generate_2d_rotary_positional_encodings(x_p, y_p, D).to(x.device)
        # x = x + rp
        x_reverse = torch.flip(x, [1])
        # x = torch.cat([x,],dim=1)
        # x_reverse = torch.cat([x_reverse,self.cls_token2],dim=1)

        # x_mamba = self.mamba_block(x)
        # x_reverse_mamba = self.mamba_block(x_reverse)
        y = self.mamba_block1(x,h5_path).squeeze(0)
        y_reverse = self.mamba_block2(x_reverse,h5_path).squeeze(0)
        # x_reverse_mamba_reverse = torch.flip(x_reverse_mamba, [1])
        # x_total = torch.cat([x_mamba,x_reverse_mamba_reverse],dim=2)
        # cls_tokens = torch.cat([y.squeeze(0)[-1],y_reverse.squeeze(0)[-1]],dim=0)
        # logits_1 = cls_tokens.mean(dim=-1)
        # logits_1 = torch.sigmoid(logits_1)
        y_reverse_reverse = torch.flip(y_reverse, [1])
        y = torch.cat([y,y_reverse_reverse],dim=-1).unsqueeze(0)
        y = self.norm(y)
        y = self.map2(y)
        y = torch.transpose(y, -1, -2) 
        y = F.softmax(y, dim=-1)
        y = torch.matmul(y,x.squeeze(0))
        logits_2 = self.classifier(y).squeeze(0)
        logits = logits_2
        return logits



