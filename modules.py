import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class mygru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, input_dim, hidden_dim):
        super().__init__()
        
        this_layer = n_layer
        self.g_ir = funcsgru(this_layer, input_dim, hidden_dim, 0) #输入重置门
        self.g_iz = funcsgru(this_layer, input_dim, hidden_dim, 0)# 输入更新门
        self.g_in = funcsgru(this_layer, input_dim, hidden_dim, 0)# 输入候选状态
        self.g_hr = funcsgru(this_layer, hidden_dim, hidden_dim, 0)# 隐藏重置门
        self.g_hz = funcsgru(this_layer, hidden_dim, hidden_dim, 0)# 隐藏更新门
        self.g_hn = funcsgru(this_layer, hidden_dim, hidden_dim, 0)# 隐藏候选状态
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, h):
        # 计算重置门 (r_t)
        r_t = self.sigmoid(
            self.g_ir(x) + self.g_hr(h)
        )
        # 计算更新门 (z_t)
        z_t = self.sigmoid(
            self.g_iz(x) + self.g_hz(h)
        )
        # 计算候选状态 (n_t)
        n_t = self.tanh(
            self.g_in(x) + self.g_hn(h).mul(r_t)
        )
        # 更新隐藏状态
        h_t = (1 - z_t) * n_t + z_t * h
        return h_t

class funcsgru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class funcs(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))
