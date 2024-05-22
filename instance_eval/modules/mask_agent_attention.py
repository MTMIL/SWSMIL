# agent attention+mask+noise
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
class Mask_Agent_Attention(nn.Module):
    def __init__(
        self,
        dim,
        pool=False,
        thresh=None,
        thresh_tem='classical',
        tem=0,
        agent_num=256,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        kaiming_init = False
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.agent_num = agent_num
        self.noise = nn.Linear(dim_head,dim_head)
        self.mask = nn.Linear(dim_head,dim_head)
        self.get_thresh = nn.Linear(dim,1)
        self.heads = heads
        # self.noise_stddev = nn.Parameter(torch.randn(1))
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.agent = nn.Parameter(torch.randn(heads, agent_num, dim//heads))
        self.residual = residual
        self.tem = tem
        self.pool = pool
        self.thresh = thresh
        self.thresh_tem = thresh_tem
        
        if kaiming_init == True:
            nn.init.kaiming_normal_(self.get_thresh.weight, mode='fan_out')
            nn.init.kaiming_normal_(self.to_qkv.weight, mode='fan_out')
            nn.init.kaiming_normal_(self.noise.weight, mode='fan_out')
            nn.init.kaiming_normal_(self.mask.weight, mode='fan_out')
        if self.thresh_tem == 'cnn':
            # 分组卷积
            self.get_thresh2 = nn.Conv1d(in_channels=dim, out_channels=4, kernel_size=1, groups=4)
            self.get_thresh3 = nn.Linear(4, 1)
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, return_attn = False):
        b, n, _, h, eps = *x.shape, self.heads, self.eps



        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        agent = self.agent.unsqueeze(0).expand(b,-1,-1,-1)
        if self.pool:
            q = q.transpose(-1,-2).squeeze(0)
            agent = nn.AdaptiveAvgPool1d(self.agent_num)(q).transpose(-1,-2)
            q = q.transpose(-1,-2).unsqueeze(0)
            
        q = torch.matmul(q,agent.transpose(-1,-2))
        k = torch.matmul(agent,k.transpose(-1,-2))
        # Add noise to the queries
        # ratio = torch.sigmoid(self.noise_stddev) 
        # noise = torch.randn_like(q)
        softmax = nn.Softmax(dim=-1)
        # mask = torch.rand_like(noise)
        # mask = (mask>ratio).float()
        # noise = noise * mask
        # noise = torch.sigmoid(noise)

        # noise = softmax(noise)
        q *= self.scale

        q = softmax(q)
        k = softmax(k)
        kv = torch.matmul(k,v) # n d 1,8,256,64
        kv_c = kv.reshape(b,self.agent_num,-1)# 1,256,N
        if self.thresh_tem == 'classical': 
            if self.thresh is not None:
                thresh = self.thresh
            else:
                thresh = self.get_thresh(kv_c).squeeze().mean()
                thresh = F.sigmoid(thresh)
        elif self.thresh_tem == 'maxpooling':
            thresh = kv_c.max(dim=1)[0].mean()
            thresh = F.sigmoid(thresh)
        elif self.thresh_tem == 'meanpooling':
            thresh = kv_c.mean(dim=1)[0].mean()
            thresh = F.sigmoid(thresh)
        elif self.thresh_tem == 'cnn':
            
            # kv_c = self.get_thresh(kv_c).squeeze()
            kv_c = self.get_thresh2(kv_c.transpose(-1,-2)).transpose(-1,-2)
            kv_c = self.get_thresh3(kv_c)
            thresh = kv_c.squeeze().mean()
            thresh = F.sigmoid(thresh)
        
        noise = self.noise(kv)
        noise = torch.sigmoid(noise)
        mask = self.mask(kv)
        mask = torch.sigmoid(mask)
        mask = torch.where(mask > thresh, torch.ones_like(mask), torch.zeros_like(mask))
        # noise = noise * mask
        
        # kv = kv  + noise * mask
        
        # 不同的方法
        if self.tem == 0:
            kv = kv * mask + noise
        elif self.tem == 1:
            kv = kv + noise * mask
        elif self.tem == 2:
            kv = kv + noise
        elif self.tem == 3:
            kv = kv * mask
        elif self.tem == 4:
            kv = kv
        
        kv = softmax(kv)
        out = torch.matmul(q,kv)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return out
    
    
if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Mask_Agent_Attention(dim=512, pool=True, thresh_tem='cnn').to(device)
    x = torch.randn(1, 1000, 512).to(device)
    out = model(x)
    print(out.shape)
    print(out)