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

class GCN_diffTest(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN_diffTest,self).__init__()
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
        return h, [out, self.projection.weight] # [B, D, N, T]

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

class GCN_cheby(nn.Module):
    def __init__(self, c_in, c_out, Ks, dropout=0.0, nodes=None):
        super(GCN_cheby,self).__init__()
        self.Ks = Ks
        self.Theta1 = nn.Parameter(torch.DoubleTensor(c_in*Ks, c_out))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, x, support):
        # x: [B, N, T, D]
        lfs = []
        for A_hat in support:
            lf = torch.einsum("ij,jklm->kilm", [A_hat, x.transpose(1,0)]).contiguous() # MN NBTD-> BMTD
            lfs.append(lf)
        lfs = torch.cat(lfs, dim=-1)
        out = F.relu(torch.matmul(lfs, self.Theta1))
        out = self.dropout(out)
        return out # [B, N, T, D]

class GCN_com(nn.Module): 
    def __init__(self, c_in, c_out, dropout=0.0, nodes=None, support_len=3, order=2):
        super(GCN_com,self).__init__()

        c_in = (support_len+1)*c_in
        self.projection = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = nn.Dropout(dropout)
   
    def forward(self, x, support):
        # x: [B, N, T, D]
        out = []
        for A_hat in support:
            if len(A_hat.shape)!=4:
                A_hat = A_hat.repeat(x.shape[0], x.shape[-2], 1, 1)
            lfs = torch.einsum("klij,jklm->kilm", [A_hat, x.transpose(1,0)]).contiguous() # BTMN NBTD-> BMTD
            out.append(lfs)
        out = torch.cat(out, dim=-1)
        out = self.projection(out.transpose(1,-1)).transpose(1,-1)
        out = self.dropout(out)
        return out # [B, N, T, D]

class GCN_com2(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0, nodes=None):
        super(GCN_com2,self).__init__()
        self.Theta1 = nn.Parameter(torch.DoubleTensor(c_in, c_out))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self,x,A_hat):
        # x: [B, N, T, D]
        lfs = torch.einsum("klij,jklm->kilm", [A_hat, x.transpose(1,0)]).contiguous() # BTMN NBTD-> BMTD
        out = F.relu(torch.matmul(lfs, self.Theta1))
        out = self.dropout(out)
        return out # [B, N, T, D]