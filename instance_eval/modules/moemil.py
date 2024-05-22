import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, reduce

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

class AgentAttention(nn.Module):
    def __init__(
        self,
        dim,
        agent_num=256,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.noise_stddev = nn.Parameter(torch.randn(1))
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.agent = nn.Parameter(torch.randn(heads, agent_num, dim//heads))
        self.to_noise = nn.Linear(agent_num, agent_num, bias = False)
        # self.to_noise2 = nn.Linear(round(agent_num/4), agent_num, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # print('---q.shape:       ',q.shape)
        # print('---k.shape:       ',k.shape)
        # print('---v.shape:       ',v.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # print('q,k,v=map..........')
        # print('---q.shape:        ',q.shape)
        # print('---k.shape:        ',k.shape)
        # print('---v.shape:        ',v.shape)
        # set masked positions to 0 in queries, keys, values


        agent = self.agent.unsqueeze(0).expand(b,-1,-1,-1)
        q = torch.matmul(q,agent.transpose(-1,-2))
        # print('---q.shape:        ',q.shape)
        k = torch.matmul(agent,k.transpose(-1,-2))
        # Add noise to the queries
        # ratio = torch.sigmoid(self.noise_stddev) 
        # noise = self.to_noise(q)
        softmax = nn.Softmax(dim=-1)
        # mask = torch.rand_like(noise)
        # mask = (mask>ratio).float()
        # noise = torch.sigmoid(noise)
        # noise = noise/torch.sum(noise)
        # noise = noise * mask
        # noise = softmax(noise)
        # q = q + noise
        q *= self.scale

        q = softmax(q)
        k = softmax(k)
        kv = torch.matmul(k,v)
        kv = softmax(kv)
        out = torch.matmul(q,kv)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return out

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

class Agent_ABMIL(nn.Module):
    def __init__(self,n_classes,dropout,act):
        super(Agent_ABMIL, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(1024, 512)]
        self.agent_attention = AgentAttention(dim=512)
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        self.apply(initialize_weights)
    def forward(self, x, return_attn=False,no_norm=False):
        print('168')
        feature = self.feature(x)
        feature = self.agent_attention(feature)
        # feature = group_shuffle(feature)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        Y_prob = self.classifier(M)
        print('177')
        if return_attn:
            if no_norm:
                return Y_prob,A_ori
            else:
                return Y_prob,A
        else:
            return Y_prob
    

class moe_AgentAttention(nn.Module):
    def __init__(
        self,
        dim,
        shared=True,
        agent_num=256,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.noise_stddev = nn.Parameter(torch.randn(1))
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        if not shared:
            self.agent = nn.Parameter(torch.randn(heads, agent_num, dim//heads))
        else:
            self.agent = None
        self.to_noise = nn.Linear(agent_num, agent_num, bias = False)
        # self.to_noise2 = nn.Linear(round(agent_num/4), agent_num, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, shared_agent=None):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        if shared_agent is None:
            agent = self.agent
        else:            
            agent = shared_agent
        agent = agent.unsqueeze(0).expand(b,-1,-1,-1)
        q = torch.matmul(q,agent.transpose(-1,-2))
        k = torch.matmul(agent,k.transpose(-1,-2))
        softmax = nn.Softmax(dim=-1)
        q *= self.scale

        q = softmax(q)
        k = softmax(k)
        kv = torch.matmul(k,v)
        kv = softmax(kv)
        out = torch.matmul(q,kv)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return out
   
class moe_mil(nn.Module):
    def __init__(self, n_classes, shared=True, dropout=0.25, num_moe=None, hidden=1024):
        super(moe_mil, self).__init__()
        """
        主要用于实现moe总体的框架，包括
        1. agent attention
        2. attention gated
        3. 融合
        """
        self.feature = [nn.Linear(hidden, 512)]
        self.feature += [nn.GELU()]
        if dropout:
            self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)
        
        self.n_classes = n_classes
        if num_moe is not None:
            self.num_moe = num_moe
        else:
            self.num_moe = n_classes * 2
        
        if shared:
            self.agent = nn.Parameter(torch.randn(8, 256, 512//8))
            self.moe = nn.ModuleList([moe_AgentAttention(512, shared=shared) for _ in range(self.num_moe)])
        else:
            self.agent = None
            self.moe = nn.ModuleList([moe_AgentAttention(512, shared=shared) for _ in range(self.num_moe)])  
            
        self.granted = nn.Linear(512, self.num_moe)
        self.softmax = nn.Softmax(dim=-1)
        
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes),
        )
              
    def forward(self, x):    
        feature = self.feature(x)
        outputs = [moe(feature, self.agent) for moe in self.moe]
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.squeeze(1)
        granted = self.softmax(self.granted(feature))
        # 取最大的granted
        granted = torch.max(granted, dim=-1)
        granted = granted.indices
        expanded_indices = granted.unsqueeze(-1).expand(-1, -1, 512)
        outputs = torch.gather(outputs, 0, expanded_indices)
        # outputs = torch.stack([outputs[granted[i], i] for i in range(outputs.size(1))])
        outputs = outputs.squeeze(0)
        
        A = self.attention(outputs)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, outputs)  # KxL
        Y_prob = self.classifier(M)
        return Y_prob
        
        
if __name__ == "__main__":
    N, feature_dim = 10, 1024
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data = torch.randn(1, N, feature_dim).to(device)
    model = moe_mil(n_classes=2, shared=True, dropout=0.25, num_moe=2, hidden=1024).to(device)
    # 输出网络参数量
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    output = model(data)
    print(output.shape)
