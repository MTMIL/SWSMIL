import torch
import torch.nn as nn
import numpy as np
from .mask_agent_attention import Mask_Agent_Attention

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, agent_num=512, tem=0, pool=False, thresh=None, thresh_tem='classical', kaiming_init=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Mask_Agent_Attention(
            dim = dim,
            tem=tem,
            thresh=thresh,
            pool=pool,
            agent_num=agent_num,
            dim_head = dim//8,
            heads = 8,  
            residual = True,        
            dropout=0.1,
            thresh_tem=thresh_tem,
            kaiming_init=kaiming_init
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class PPEG(nn.Module):
    def __init__(self, dim=512,cls_num=1):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        self.cls_num = cls_num

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, :self.cls_num], x[:, self.cls_num:]
    
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, x), dim=1)
        return x

class Mask_Agent_MIL(nn.Module):
    def __init__(self, n_classes,dropout,act,agent_num=512,cls_num=1,cls_agg='mean', tem=0, pool=False, thresh=None, thresh_tem='classical', kaiming_init=False):
        super(Mask_Agent_MIL, self).__init__()
        self.pos_layer = PPEG(dim=512, cls_num=cls_num)
        #self.pos_layer = nn.Identity()
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self._fc1 = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(0.25)]

        #self._fc1 += [SwinEncoder(attn='swin',pool='none',n_heads=2,trans_conv=False)]
        
        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, cls_num, 512))
        self.cls_num = cls_num
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512, agent_num=agent_num, tem=tem, pool=pool, thresh=thresh, thresh_tem=thresh_tem, kaiming_init=kaiming_init)
        self.layer2 = TransLayer(dim=512, agent_num=agent_num, tem=tem, pool=pool, thresh=thresh, thresh_tem=thresh_tem, kaiming_init=kaiming_init)
        self.norm = nn.LayerNorm(512)        
        if cls_agg == 'concat':
            self._fc2 = nn.Linear(512*self.cls_num, n_classes)
        else:
            self._fc2 = nn.Linear(512, n_classes)
        if kaiming_init:
            initialize_weights(self._fc2)
            initialize_weights(self._fc1)
        
        self.cls_agg = cls_agg  # 'mean' or 'max'
        self.apply(initialize_weights)

    def forward(self, x):

        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,:self.cls_num] #  [B, cls_num, 512]
        #---->predict
        
        if self.cls_agg == 'mean':
            h = h.mean(1)
        elif self.cls_agg == 'max':
            h = h.max(1)[0]
        elif self.cls_agg == 'sum':
            h = h.sum(1)
        elif self.cls_agg == 'concat':   
            h = h.view(h.size(0), -1)
        else:
            raise "nmd"     
        logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits

if __name__ == "__main__":
    model = Mask_Agent_MIL(n_classes=2,dropout=False,act='relu',cls_num=10,cls_agg='sum').cuda()
    x = torch.randn(1, 1000, 1024).cuda()
    y = model(x)
    print(y.shape)


