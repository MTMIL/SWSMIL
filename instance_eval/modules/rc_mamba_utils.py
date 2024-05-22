import torch
import math
import h5py
import numpy as np
import torch.nn as nn
'''
输入 (1, L, D) M个cls_tokens
嵌入cls_tokens (1, L+M, D)          保存cls_tokens的索引(从0开始)
'''
def insert_values(coords, pos_index):
    coords_list = coords.tolist()
    for index in pos_index:
        if index == 0:
            new_value = np.ceil(np.array(coords_list[index]) // 2)
        elif index == len(coords_list):
            new_value = np.ceil(coords_list[index - 1] / 2)
        else:
            a1 = coords_list[index - 1]
            b1 = coords_list[index]
            new_value = [a + b for a, b in zip(a1, b1)]
            new_value = [math.ceil(x / 2) for x in new_value]
            print(new_value)
        coords_list.insert(index, new_value)
    return torch.tensor(np.array(coords_list))

def INSERT_cls_token(input:torch.Tensor,cls_token_list:nn.ParameterList):
    num_cls_token = len(cls_token_list)
    B, L, D = input.shape
    interval = math.floor(L/num_cls_token)
    positions = []
    for i in range(num_cls_token):
        positions.append(int(i*interval))
    assert len(cls_token_list) == len(positions)
    assert cls_token_list[0].shape[2] == input.shape[2]
    segments = []
    for i in range(len(positions) - 1):
        segments.append(cls_token_list[i])
        segments.append(input[:, positions[i]:positions[i+1], :])
    segments.append(cls_token_list[-1])
    segments.append(input[:, positions[-1]:, :])

    output_tensor = torch.cat(segments, dim=1)
    positions_index = [value + i for i, value in enumerate(positions)]
    return output_tensor,positions_index


def change_direction(input:torch.Tensor,position_index_list):
    '''
    position_index_list:是当前输入tensor的所有cls_token的位置索引
    '''
    L = input.shape[1]
    interval_point = position_index_list[1]
    output_tensor = torch.cat((input[:, interval_point:, :], input[:, :interval_point, :]), dim=1)
    '''
    返回新的pos_index_list
    '''
    new_position_index_list = []
    for i in range(len(position_index_list)):
        if i == 0:
            continue
        else:
            new_position_index_list.append(position_index_list[i]-position_index_list[1])
    new_position_index_list.append(L-position_index_list[-1]+new_position_index_list[-1])
    return output_tensor,new_position_index_list

'''
进行RC_操作 (2*M, L+M, D),
'''
def RC_process(input:torch.Tensor,cls_token_pos_index:list):
    num_cls_token = len(cls_token_pos_index)
    c_output = []
    c_output.append(input)
    c_cls_token_pos_list = [] # 保存c_cls_token的位置索引  是一个M M的列表
    c_cls_token_pos_list.append(cls_token_pos_index)
    for i in range(num_cls_token-1):
        c_input,c_cls_token_pos_index = change_direction(input,cls_token_pos_index)
        c_output.append(c_input)
        c_cls_token_pos_list.append(c_cls_token_pos_index)
        input = c_input
        cls_token_pos_index = c_cls_token_pos_index
    r_output = [] 
    for _,c_tensor in enumerate(c_output):
        r_tensor = torch.flip(c_tensor, [1])
        r_output.append(r_tensor)
    
    rc_output = []
    for c,r in zip(c_output,r_output):
        rc_output.append(c)
        rc_output.append(r)
    
    return torch.stack(rc_output,dim=0).squeeze(1),c_cls_token_pos_list

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

def PCPE_process(B,C,input:torch.Tensor,postion_index,h5_path,expand):
    _,D,L = input.shape
    h5_file = h5py.File(h5_path, 'r')
    coords = np.array(h5_file['coords'])
    coords = insert_values(coords, postion_index)
    x = coords[:, 0].clone().detach()
    y = coords[:, 1].clone().detach()
    assert len(x) == L
    pos_encodings = generate_2d_rotary_positional_encodings(x, y, D*expand//2).to(input.device)
    pos_encodings = pos_encodings.unsqueeze(0)     # 1, L+M, D*expand
    pos_encodings,_ = RC_process(pos_encodings,postion_index)
    pB = torch.matmul(B, pos_encodings)
    pBC = torch.matmul(C.transpose(1,2), pB)
    pBC = torch.softmax(pBC, dim=-1)
    input_with_clstoken_pos = input+pBC.transpose(1,2)
    return input_with_clstoken_pos
    
    
def R_fusion(input:torch.Tensor):
    M = int(input.shape[0] // 2)
    tensor1 = input[::2]
    tensor2 = input[1::2]
    tensor2 = tensor2.flip(1)
    '''
    直接加
    '''
    tensor_R = tensor1 + tensor2
    return tensor_R

def move_to_front(tensor, index):
    front = tensor[:, index:]
    back = tensor[:, :index]
    result = torch.cat((front, back), dim=1)
    return result

def C_fusion(input:torch.Tensor,c_cls_token_pos_list): 
    assert len(c_cls_token_pos_list) == len(input)
    M = len(c_cls_token_pos_list)
    for i in range(M-1):
        i = len(c_cls_token_pos_list) - i - 1
        tensor_i = input[i].unsqueeze(0)
        index = c_cls_token_pos_list[i][-1]
        tensor_i = move_to_front(tensor_i, index)
        input[i-1] = tensor_i + input[i-1]
    
    return input[0]
        
    
def Get_cls_tokens(input,cls_token_pos_index):
    # print('------input------')
    input = input.squeeze(0)
    cls_tokens = torch.stack([input[index] for index in cls_token_pos_index],dim=0)
    cls_tokens = cls_tokens.unsqueeze(0)
    return cls_tokens


if __name__ == '__main__':
    print('------test------')
    cls_token_list = [torch.ones(1, 1,5), torch.ones(1,1, 5)*2, torch.ones(1,1, 5)*3]
    ori_input = torch.zeros(1, 10, 5)
    ori_input_with_cls_token,cls_tokens_position = INSERT_cls_token(ori_input,cls_token_list)
    print('------ori_input_with__cls_token------')
    print(ori_input_with_cls_token)
    print('------cls_tokens_position------')
    print(cls_tokens_position)
    rc_output,c_cls_token_pos_index = RC_process(ori_input_with_cls_token,cls_tokens_position)
    print('------rc_output------')
    print(rc_output.shape)
    print(rc_output)
    print('------c_cls_token_pos_index------')
    print(c_cls_token_pos_index)
    print('------pos_encodings------')
    h5_path = '/mnt_ljw/lxt_projects/Camelyon16_data/training/tumor/patches/tumor_001.h5'
    input = torch.randn(1, 7896, 100)
    pos_index = [20,40,50]
    pos_encoding  = PCPE_process(1,1,input,pos_index,h5_path,4)
    print(pos_encoding.shape)
    # print(pos_encoding)
    
    
    