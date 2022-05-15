import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class GCN_diff(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN_diff,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.projection = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # x: [B, D, N, T]
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        
        h = torch.cat(out,dim=1)
        h = self.projection(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h # [B, D, N, T]

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0, nodes=None):
        super(GCN,self).__init__()
        self.Theta1 = nn.Parameter(torch.DoubleTensor(c_in, c_out))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self,x,A_hat):
        # x: [B, N, T, D]
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, x.transpose(1,0)]).contiguous() # MN NBTD-> BMTD
        out = F.relu(torch.matmul(lfs, self.Theta1))
        out = self.dropout(out)
        return out # [B, N, T, D]


class nconv_t2s(nn.Module):
    def __init__(self):
        super(nconv_t2s,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('bdn,bnm->bdm',(x,A))
        return x.contiguous()

class GCN_t2s(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN_t2s,self).__init__()
        self.nconv = nconv_t2s()
        c_in = (order*support_len+1)*c_in
        self.projection = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # x: [B, D, N]
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        
        h = torch.cat(out, dim=1)
        h = self.projection(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h # [B, D, N]