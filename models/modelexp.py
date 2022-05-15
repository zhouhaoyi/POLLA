import torch
import torch.nn as nn
import torch.nn.functional as F

from util.masking import FullMask, LengthMask, TriangularCausalMask
from models.attnexp import STGCNAttentionLayer, STGCNdiffAttentionLayer, STLinearAttention, STFullAttention, elu_feature_map, \
    STAttentionGCNdiffLayer, STGCNchebyAttentionLayer, STGCNcomAttentionLayer, STGCNcom2AttentionLayer, STGCNdiffAttentionLayerTest, STFullAttentionTest
from models.embed import TemporalEncoding, TokenEncoding, PositionalEncoding

class POLLAEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(POLLAEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1)) # nn.Linear(d_model, d_ff)
        self.linear2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,1)) # nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        attn_mask = attn_mask or FullMask(L, device=x.device) # TriangularCausalMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64), device=x.device)
        
        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask = attn_mask,
            query_lengths = length_mask,
            key_lengths = length_mask,
            support = support
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y.transpose(-1,1)))) # [B, D, N, L]
        y = self.dropout(self.linear2(y)).transpose(-1,1) # [B, L, N, D]

        return self.norm2(x+y)

class POLLAEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(POLLAEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        attn_mask = attn_mask or FullMask(L, device=x.device) # TriangularCausalMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64), device=x.device)

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask, support=support)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x

class POLLA_gcn(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_gcn, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STGCNAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        dropout=dropout, nodes=nodes),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)# + self.spatial_embedding(se)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLA_diff(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STGCNdiffAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = self.supports

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)# + self.spatial_embedding(se)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLA_adpadj(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_adpadj, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        if aptinit is None:
            self.nodevec1 = nn.Parameter(torch.randn(nodes, 10).double(), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, nodes).double(), requires_grad=True).to(device)
            support_len +=1
        else:
            m, p, n = torch.svd(aptinit)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self.nodevec1 = nn.Parameter(initemb1.double(), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(initemb2.double(), requires_grad=True).to(device)
            support_len += 1

        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STGCNdiffAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        support = self.supports + [adp]

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)# + self.spatial_embedding(se)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(elf.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLA_adpadj_test(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_adpadj_test, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        if aptinit is None:
            self.nodevec1 = nn.Parameter(torch.randn(nodes, 10).double(), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, nodes).double(), requires_grad=True).to(device)
            support_len +=1
        else:
            m, p, n = torch.svd(aptinit)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self.nodevec1 = nn.Parameter(initemb1.double(), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(initemb2.double(), requires_grad=True).to(device)
            support_len += 1

        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STAttentionGCNdiffLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        support = self.supports + [adp]

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)# + self.spatial_embedding(se)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLA_diff_test(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff_test, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STAttentionGCNdiffLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = self.supports

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)# + self.spatial_embedding(se)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLAEncoder_skip(nn.Module):
    def __init__(self, layers, skip_convs, norm_layer=None):
        super(POLLAEncoder_skip, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.skip_convs = nn.ModuleList(skip_convs)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        attn_mask = attn_mask or FullMask(L, device=x.device) # TriangularCausalMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64), device=x.device)

        # Apply all the transformers
        skip = 0
        for layer, skip_conv in zip(self.layers, self.skip_convs):
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
            s = skip_conv(x.transpose(-1,1)).transpose(-1,1)
            skip = skip + s

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(F.relu(skip))

        return x

class POLLA_adpadj_skip(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_adpadj_skip, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        if aptinit is None:
            self.nodevec1 = nn.Parameter(torch.randn(nodes, 10).double(), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, nodes).double(), requires_grad=True).to(device)
            support_len +=1
        else:
            m, p, n = torch.svd(aptinit)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self.nodevec1 = nn.Parameter(initemb1.double(), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(initemb2.double(), requires_grad=True).to(device)
            support_len += 1

        # encoder
        self.encoder = POLLAEncoder_skip(
            [
                POLLAEncoderLayer(
                    STGCNdiffAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            [
                nn.Conv1d(
                    in_channels=d_model, 
                    out_channels=d_model, 
                    kernel_size=(1, 1)
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        support = self.supports + [adp]

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)# + self.spatial_embedding(se)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]


class POLLAEncoder_com(nn.Module):
    def __init__(self, layers, 
                batch_size=32, seq_len=12, nodes=207, d_model=32,
                device=None, norm_layer=None):
        super(POLLAEncoder_com, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.nfs = [nn.Parameter(torch.randn(batch_size, seq_len, nodes, d_model), requires_grad=False).to(device) for l in layers]
        self.pfs = []

    def forward(self, x, attn_mask=None, length_mask=None, support=None, mode='train', alpha=0.1):
        # x [B, L, N, D]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        attn_mask = attn_mask or FullMask(L, device=x.device) 
        length_mask = length_mask or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64), device=x.device)

        # Apply all the transformers
        # pfs[i]: [B, L, N, D]
        # support: [N, N]
        if mode=='train':
            self.nfs[0].copy_(x.detach())
        
        n_support = support
        for i, layer in enumerate(self.layers):
            if mode=='train' and len(self.pfs)>0:
                theta = self.pfs[i-1].detach()
                support_add = F.softmax(F.relu(torch.einsum('blnd,bldm->blnm', theta, theta.transpose(-1,-2))))
                n_support = support + [support_add.double().to(x.device)]
            else:
                n_support = support + support[0:1]

            x = layer(x, attn_mask=attn_mask, length_mask=length_mask, support=n_support)
            
            if mode=='train' and i+1 < len(self.layers):
                self.nfs[i+1].copy_(x.detach())
        
        if mode=='train':
            self.pfs = self.nfs
        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x

class POLLA_com(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, batch_size=32, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_com, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        # encoder
        self.encoder = POLLAEncoder_com(
            [
                POLLAEncoderLayer(
                    STGCNcomAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            batch_size = batch_size,
            seq_len = seq_len,
            nodes = nodes,
            d_model = d_model,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None, mode='train', alpha=0.1):
        # x [B, L, N, D]
        support = self.supports

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)# + self.spatial_embedding(se)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support, mode=mode) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLAEncoder_com2(nn.Module):
    def __init__(self, layers, 
                batch_size=16, seq_len=12, nodes=207, d_model=32,
                device=None, norm_layer=None):
        super(POLLAEncoder_com2, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.nfs = [nn.Parameter(torch.randn(batch_size, seq_len, nodes, d_model), requires_grad=False).to(device) for l in layers]
        self.pfs = []

    def forward(self, x, attn_mask=None, length_mask=None, support=None, mode='train', alpha=0.1):
        # x [B, L, N, D]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64), device=x.device)

        # Apply all the transformers
        # pfs[i]: [B, L, N, D]
        # support: [N, N]
        if mode=='train':
            self.nfs[0].copy_(x.detach())
        support = support.repeat(x.shape[0],x.shape[1],1,1)
        n_support = support
        for i, layer in enumerate(self.layers):
            if mode=='train' and len(self.pfs)>0:
                theta = self.pfs[i-1].detach()
                support_add = F.softmax(F.relu(torch.einsum('blnd,bldm->blnm', theta, theta.transpose(-1,-2))))
                n_support = support + alpha*support_add
                
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask, support=n_support)
            if mode=='train' and i+1 < len(self.layers):
                self.nfs[i+1].copy_(x.detach())
        if mode=='train':
            self.pfs = self.nfs
        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x

class POLLA_com2(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              batch_size=16, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_com2, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # encoder
        self.encoder = POLLAEncoder_com2(
            [
                POLLAEncoderLayer(
                    STGCNcom2AttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        dropout=dropout, nodes=nodes),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            batch_size=batch_size,
            seq_len = seq_len,
            nodes = nodes,
            d_model = d_model,
            device = device,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None, mode='train', alpha=0.1):
        # x [B, L, N, D]
        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support, mode=mode, alpha=alpha) # [B, L, N, D]
        out = self.end_conv1(out) # [B, OL, N, D]
        out = F.relu(out)
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]


class POLLA_cheby(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              Ks=3, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_cheby, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STGCNchebyAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        Ks=Ks, dropout=dropout, nodes=nodes),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.end_conv1(out) # [B, OL, N, D]
        out = F.relu(out)
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLA_softmax(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_softmax, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STGCNAttentionLayer(STFullAttention(), d_model, n_heads, 
                                        dropout=dropout, nodes=nodes),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        L = x.shape[1]
        attn_mask = attn_mask or TriangularCausalMask(L, device=x.device)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.end_conv1(out) # [B, OL, N, D]
        out = F.relu(out)
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]

class POLLA_diff_softmax(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff_softmax, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STGCNdiffAttentionLayer(STFullAttention(), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        L = x.shape[1]
        attn_mask = attn_mask or TriangularCausalMask(L, device=x.device)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.end_conv1(out) # [B, OL, N, D]
        out = F.relu(out)
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]


class POLLA_diff_embed(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, embed_type='temp', dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff_embed, self).__init__()
        
        self.embed_type = embed_type

        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        # encoder
        self.encoder = POLLAEncoder(
            [
                POLLAEncoderLayer(
                    STGCNdiffAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = self.supports

        if self.embed_type=='temp':
            out = self.value_embedding(x) + self.temporal_embedding(x_mark)
        elif self.embed_type=='pos':
            out = self.value_embedding(x) + self.position_embedding(x)
        elif self.embed_type=='no':
            out = self.value_embedding(x)
        else:
            out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]



class POLLAEncoderLayerTest(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(POLLAEncoderLayerTest, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1))
        self.linear2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        attn_mask = attn_mask or FullMask(L, device=x.device) 
        length_mask = length_mask or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64), device=x.device)
        
        # Run self attention and add it to the input
        val, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask,
            query_lengths = length_mask,
            key_lengths = length_mask,
            support = support
        )
        x = x + self.dropout(val)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y.transpose(-1,1)))) # [B, D, N, L]
        y = self.dropout(self.linear2(y)).transpose(-1,1) # [B, L, N, D]

        return self.norm2(x+y), attn

class POLLAEncoderTest(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(POLLAEncoderTest, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64), device=x.device)

        # Apply all the transformers
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask, length_mask=length_mask, support=support)
            attns.append(attn)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class POLLA_diff_fullTest(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff_fullTest, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        # encoder
        self.encoder = POLLAEncoderTest(
            [
                POLLAEncoderLayerTest(
                    STGCNdiffAttentionLayerTest(STFullAttentionTest(), d_model, n_heads, 
                                        support_len=support_len, order=order, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)

        self.pred_len = out_len
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = self.supports
        L = x.shape[1]
        attn_mask = attn_mask or TriangularCausalMask(L, device=x.device)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out, attns = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.end_conv1(out) # [B, OL, N, D]
        out = F.relu(out)
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out, attns # [B, L, N, D]
