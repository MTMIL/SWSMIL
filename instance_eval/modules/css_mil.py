import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba.mamba_ssm.modules.rc_mamba import RC_Mamba
from . import rc_mamba_utils
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=2):
        super(AttentionPooling, self).__init__()
        self.key = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(input_dim, output_dim)  # Add a fully connected layer

    def forward(self, x):
        keys = self.key(x)  # Shape: (batch_size, tokens, hidden_dim)
        attn_weights = torch.bmm(keys, self.query.unsqueeze(0))  # Shape: (batch_size, tokens, 1)
        attn_weights = self.softmax(attn_weights)  # Shape: (batch_size, tokens, 1)
        pooled = torch.bmm(attn_weights.transpose(1, 2), x)  # Shape: (batch_size, 1, features)
        output = self.fc(pooled.squeeze(1))  # Shape: (batch_size, output_dim)
        return output
    
    
    
class CSS_MIL(nn.Module):
    def __init__(self,n_classes,num_cls_token,pt_dim = 512):
        super(CSS_MIL,self).__init__()
        K = 512
        self.map = nn.Linear(1024,pt_dim)
        self.pt_dim = pt_dim
        self.n_classes = n_classes
        self.num_cls_token = num_cls_token
        self.cls_tokens_list = nn.ParameterList([nn.Parameter(torch.randn(1, 1, pt_dim)) for _ in range(self.num_cls_token)])
        self.rc_mamba = RC_Mamba(d_model=pt_dim, d_state=128, d_conv=4, expand=2)
        self.att_pool = AttentionPooling(pt_dim,output_dim=self.n_classes)
        self.classifier1 = nn.Linear(num_cls_token*pt_dim*2,K)
        self.classifier2 = nn.Linear(K,n_classes)



    def forward(self,x,h5_path):
        x = self.map(x)
        x,cls_token_pos_index = rc_mamba_utils.INSERT_cls_token(x,self.cls_tokens_list)
        # print(x.shape)
        # print(cls_token_pos_index)
        y = self.rc_mamba(x,cls_token_pos_index,h5_path)
        # print('------y------')
        # print(y.shape)
        # print('------cls_token_pos_index------')
        # print(cls_token_pos_index)
        cls_tokens = rc_mamba_utils.Get_cls_tokens(y,cls_token_pos_index)
        # logits = self.att_pool(cls_tokens)
        # print('------cls_tokens------')
        # print(cls_tokens.shape)
        logits = self.classifier1(cls_tokens.view(-1,self.num_cls_token*self.pt_dim*2))
        logits = self.classifier2(F.relu(logits))
        # print('------logits------')
        # print(logits.shape)
        # print('------logits------')
        # print(logits)
        print('------logits------')
        print(logits)
        return logits
        



if __name__ == '__main__':
    print('------test_css_mil------')
    css_mil_model = CSS_MIL(2,3).to('cuda:2')
    x = torch.randn(1,7893,1024).to('cuda:2')
    h5_path = '/mnt_ljw/lxt_projects/Camelyon16_data/training/tumor/patches/tumor_001.h5'
    logits = css_mil_model(x,h5_path)
    print('------logits------')
    print(logits.shape)