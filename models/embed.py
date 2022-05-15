import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(-2)    


class TemporalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super(TemporalEncoding, self).__init__()

        self.dayofweek_emb = nn.Embedding(7, d_model)
        self.timeofday_emb = nn.Embedding(24*12, d_model)
    
    def forward(self, x):
        x = x.long()
        
        dayofweek_x = self.dayofweek_emb(x[:,:,:,0])
        timeofday_x = self.timeofday_emb(x[:,:,:,1])
        
        return dayofweek_x + timeofday_x

class TokenEncoding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(TokenEncoding, self).__init__()

        self.conv = nn.Conv2d(in_channels=c_in, out_channels=d_model, kernel_size=(1,3), padding=(0,2), padding_mode='circular', bias=True)
        # input: [B, C_channel, H_height, W_width]; kernelsize: [h, w]; padding: [h, w]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [B, L, N, D]
        x = self.conv(x.permute(0,3,2,1)) # [B, D, N, L]
        
        return x.transpose(-1,1) # [B, L, N, D]

class SpatialEncoding(nn.Module):
    def __init__(self, c_in, d_model):
        super(SpatialEncoding, self).__init__()
        
        # self.SE = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=(1,1), bias=True)
        self.SE = nn.Linear(c_in, d_model)

    def forward(self, x):
        x = self.SE(x)

        return x