import torch
import torch.nn as nn
import torch.nn.functional as F

from util.masking import FullMask, LengthMask, TriangularCausalMask
from models.attn import STGCNAttentionLayer, STGCNdiffAttentionLayer, STLinearAttention, STFullAttention, elu_feature_map
from models.embed import TemporalEncoding, TokenEncoding, PositionalEncoding
from models.attnexp import STGCNdiffAttentionLayerTest, STFullAttentionTest

class POLLAEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(POLLAEncoderLayer, self).__init__()
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
        attn_mask = attn_mask or FullMask(L, device=x.device)
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
        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
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

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
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

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]


class POLLA_gcn_full_gen(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_gcn_full_gen, self).__init__()
        
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
        
        self.linear = nn.Linear(d_model, c_out)

        self.pred_len = out_len
        
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        L = x.shape[1]
        attn_mask = attn_mask or TriangularCausalMask(L, device=x.device)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.linear(out)
        
        return out[:,-self.pred_len:,:,:] # [B, L, N, D]

class POLLA_gcn_linear_gen(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_gcn_linear_gen, self).__init__()
        
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
        
        self.linear = nn.Linear(d_model, c_out)

        self.pred_len = out_len
        
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.linear(out)
        
        return out[:,-self.pred_len:,:,:] # [B, L, N, D]

class POLLA_diff_linear_gen(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff_linear_gen, self).__init__()
        
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
        self.linear = nn.Linear(d_model, c_out)

        self.pred_len = out_len
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = self.supports

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.linear(out)
        
        return out[:,-self.pred_len:,:,:] # [B, L, N, D]

class POLLA_diff_full_gen(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff_full_gen, self).__init__()
        
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
        self.linear = nn.Linear(d_model, c_out)

        self.pred_len = out_len
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = self.supports
        L = x.shape[1]
        attn_mask = attn_mask or TriangularCausalMask(L, device=x.device)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.linear(out)
        
        return out[:,-self.pred_len:,:,:] # [B, L, N, D]


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

class POLLA_diff_full_genTest(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(POLLA_diff_full_genTest, self).__init__()
        
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
        self.linear = nn.Linear(d_model, c_out)

        self.pred_len = out_len
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = self.supports
        L = x.shape[1]
        attn_mask = attn_mask or TriangularCausalMask(L, device=x.device)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        out, attns = self.encoder(out, attn_mask=attn_mask, length_mask=length_mask, support=support) # [B, L, N, D]
        out = self.linear(out)
        
        return out[:,-self.pred_len:,:,:], attns # [B, L, N, D]
