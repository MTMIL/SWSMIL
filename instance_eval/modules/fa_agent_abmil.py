import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import modules.agent_attention
import modules.fa_agent_attention
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

class FA_Agent_ABMIL(nn.Module):
    def __init__(self,n_classes,dropout,act):
        super(FA_Agent_ABMIL, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(1024, 512)]
        self.fa_agent_attention1 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=512,v_de_dim=256,dim=512,dim_head=64)
        self.fa_agent_attention2 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=256,v_de_dim=128,dim=256,dim_head=32)
        self.fa_agent_attention3 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=128,v_de_dim=64,dim=128,dim_head=16)
        self.fa_agent_attention4 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=64,v_de_dim=32,dim=64,dim_head=8)
        
        self.fa_agent_attention5 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=32,v_de_dim=64,dim=32,dim_head=4)
        self.fa_agent_attention6 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=64,v_de_dim=128,dim=64,dim_head=8)
        self.fa_agent_attention7 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=128,v_de_dim=256,dim=128,dim_head=16)
        self.fa_agent_attention8 = modules.fa_agent_attention.FA_AgentAttention_Block(v_in_dim=256,v_de_dim=512,dim=256,dim_head=32)
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
        feature1 = self.feature(x)
        feature2 = self.fa_agent_attention1(feature1)
        feature3 = self.fa_agent_attention2(feature2)
        feature4 = self.fa_agent_attention3(feature3)
        feature5 = self.fa_agent_attention4(feature4)
        
        feature6 = self.fa_agent_attention5(feature5)+feature4
        feature7 = self.fa_agent_attention6(feature6)+feature3
        feature8 = self.fa_agent_attention7(feature7)+feature2
        feature9 = self.fa_agent_attention8(feature8)+feature1
        
        # feature = group_shuffle(feature)
        feature9 = feature9.squeeze(0)
        A = self.attention(feature9)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature9)  # KxL
        Y_prob = self.classifier(M)

        if return_attn:
            if no_norm:
                return Y_prob,A_ori
            else:
                return Y_prob,A
        else:
            return Y_prob
if __name__ == "__main__":
    x=torch.rand(1,100,1024).cuda()
    model = FA_Agent_ABMIL(2,dropout=True,act='relu')
    model=model.cuda()
    Y_prob=model(x)
    
    gcnnet=Resnet().cuda()
    Y_prob=gcnnet(x)
    criterion = nn.BCEWithLogitsLoss()
    # loss_max = criterion(Y_prob[1].view(1,-1), label.view(1,-1))
    print(Y_prob)

