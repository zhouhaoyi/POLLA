import torch
import torch.nn as nn
import torch.nn.functional as F

from util.masking import FullMask, LengthMask, TriangularCausalMask
from models.attnT2S import STAttentionT2SLayer, STGCNAttentionLayer, STGCNdiffAttentionLayer, STLinearAttention, STFullAttention, elu_feature_map
from models.embed import TemporalEncoding, TokenEncoding, PositionalEncoding
from models.gcnT2S import GCN_diff, GCN, GCN_t2s

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

class POLLADecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, pred_len, nodes=207,
                 d_ff=None, support_len=1, order=2, dropout=0.1, activation="relu"):
        super(POLLADecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear3 = nn.Linear(d_model, nodes)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.gcns = nn.ModuleList(
            [
                GCN_t2s(
                    d_model, d_model, 
                    dropout=dropout, support_len=support_len, order=order
                ) for t in range(pred_len)
            ]
        )

    def forward(self, x, memory, data, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None, support=None):
        # Normalize the masks
        B = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((B,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask
        ))
        x = self.norm1(x)

        # Secondly apply the cross attention and add it to the previous output
        x = x + self.dropout(self.cross_attention(
            memory, x, x,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        ))

        # Finally run the fully connected part of the layer
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        # gcn
        As = self.linear3(y) # [B, L, N, N]
        # GCN input: x [B,D,N] Ahat [B,N,N] 
        dts = []
        for l in range(L):
            A = As[:,l,:,:] # [B,N,N]
            dt = data[:,l,:,:] # [B,N,D]
            n_support = support+[A]
            n_dt = self.gcns[l](dt.permute(0,2,1), n_support) # [B,D,N]
            dts.append(n_dt.transpose(2,1)) #[B,N,D]
        dts = torch.stack(dts, dim=1) # [B,L,N,D]

        return self.norm3(x+y), dts

class T2SLayer(nn.Module):
    def __init__(self, temporal_layers, spatial_layers, skip_convs, norm_layer=None):
        super(T2SLayer, self).__init__()
        self.temporal_layers = nn.ModuleList(temporal_layers)
        self.spatial_layers = nn.ModuleList(spatial_layers)
        self.skip_convs = nn.ModuleList(skip_convs)
        self.norm = norm_layer

    def forward(self, x, As, data, x_mask=None, x_length_mask=None,
                As_mask=None, As_length_mask=None, support=None):
        # Normalize the masks
        B = x.shape[0]
        L = x.shape[1]
        L_prime = As.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64))
        As_mask = As_mask or FullMask(L, L_prime, device=x.device)
        As_length_mask = As_length_mask or \
            LengthMask(x.new_full((B,), L_prime, dtype=torch.int64))

        # Apply all the transformer decoders
        skip = 0.
        for t_layer, s_layer, skip_conv in zip(self.temporal_layers, self.spatial_layers, self.skip_convs):
            # 此处不是一般的encoder-decoder
            # 一般的形式是先完成encoder, 然后把encoder的结构当做memory丢给decoder
            # 此处是encoder和decoder同时进行, 相当于enc是编码temporal, dec是将时序信息作为memory然后生成As
            x = t_layer(x, x_mask, x_length_mask, support)
            As, dt = s_layer(x=As, memory=x, data=data, # data可以为不变的data, 也可以为变化的x
                            x_mask=As_mask, x_length_mask=As_length_mask,
                            memory_mask=x_mask, memory_length_mask=x_length_mask,
                            support=support)
            # dt [B,L,N,D]
            s = skip_conv(dt.transpose(-1,1)).transpose(-1,1) # 这里必须是dt
            skip = skip + s

        # Apply the normalization if needed
        if self.norm is not None:
            v = self.norm(F.relu(skip))

        return v

class T2S_Enc_skip(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              trans_type='trans1', aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(T2S_Enc_skip, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.spatial_embedding = nn.Linear(nodes, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        support_len += 1

        # encoder
        self.encoder = T2SLayer(
            temporal_layers=[
                POLLAEncoderLayer(
                    STAttentionT2SLayer(STLinearAttention(feature_map=elu_feature_map), 
                                        d_model, n_heads, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            spatial_layers=[
                POLLADecoderLayer(
                    STAttentionT2SLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        dropout=dropout),
                    STAttentionT2SLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        dropout=dropout),
                    d_model=d_model,
                    pred_len=out_len,
                    nodes=nodes,
                    d_ff=d_ff, 
                    support_len=support_len, 
                    order=order,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers)
            ],
            skip_convs=[
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
        support = [s.repeat(x.shape[0],1,1) for s in self.supports]
        As = self.supports[0].repeat(x.shape[0],x.shape[1],1,1)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        As = self.spatial_embedding(As) + self.position_embedding(As)
        As = self.embedding_dropout(As)
        out = self.encoder(x=out, As=As, data=out, 
                           x_mask=attn_mask, x_length_mask=length_mask,
                           As_mask=attn_mask, As_length_mask=length_mask,
                           support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]    


class TransLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, pred_len,
                 nodes=207, d_ff=None, support_len=1, order=2,
                 dropout=0.1, activation="relu"):
        super(TransLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear3 = nn.Linear(d_model, nodes)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.gcns = nn.ModuleList(
            [
                GCN_t2s(
                    d_model, d_model, 
                    dropout=dropout, support_len=support_len, order=order
                ) for t in range(pred_len)
            ]
        )

    def forward(self, x, memory, data, Sinfo, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None, support=None):
        # X [B, L, N, D] data [B, L, N, D]
        # memory [B, L, N, D] [N,N]->[B,L,N,D]
        
        # Normalize the masks
        B = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((B,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask,
        ))
        x = self.norm1(x) # [B, L, N, D]
        
        # Secondly apply the cross attention and add it to the previous output
        x = x + self.dropout(self.cross_attention(
            x, memory, memory,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        )) # 此处是memory永远不变 而x在第一层是原始数据 但是到高层就是已经和memory混合过的
        # [B, L, N, D]
        
        # Finally run the fully connected part of the layer
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        
        # gcn
        As = self.linear3(y) # [B, L, N, N]
        # GCN input: x [B,D,N] Ahat [B,N,N] 
        dts = []
        for l in range(L):
            A = As[:,l,:,:] # [B,N,N]
            dt = data[:,l,:,:] # [B,N,D]
            n_support = support+[A]
            n_dt = self.gcns[l](dt.permute(0,2,1), n_support) # [B,D,N]
            dts.append(n_dt.transpose(2,1)) #[B,N,D]
        dts = torch.stack(dts, dim=1) # [B,L,N,D]

        return self.norm3(x+y), memory, dts, None

class TransLayer2(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, pred_len,
                 nodes=207, d_ff=None, support_len=1, order=2,
                 dropout=0.1, activation="relu"):
        super(TransLayer2, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear3 = nn.Linear(d_model, d_ff)
        self.linear4 = nn.Linear(d_ff, d_model)
        self.linearN2 = nn.Linear(d_ff, nodes)


        self.norm_x1 = nn.LayerNorm(d_model)
        self.norm_x2 = nn.LayerNorm(d_model)
        self.norm_m1 = nn.LayerNorm(d_model)
        self.norm_m2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.gcns = nn.ModuleList(
            [
                GCN_t2s(
                    d_model, d_model, 
                    dropout=dropout, support_len=support_len, order=order
                ) for t in range(pred_len)
            ]
        )

    def forward(self, x, memory, data, Sinfo, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None, support=None):
        # X [B, L, N, D] data [B, L, N, D]
        # memory [B, L, N, D] [N,N]->[B,L,N,D]
        
        # Normalize the masks
        B = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((B,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask,
        ))
        x = self.norm_x1(x) # [B, L, N, D]
        
        # Secondly apply the cross attention and add it to the previous output
        memory = memory + self.dropout(self.cross_attention(
            x, memory, memory,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        ))
        memory = self.norm_m1(memory) # 此处是memory也会和x一样不停往上叠加信息
        # [B, L, N, D]
        
        # Finally run the fully connected part of the layer
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        m = memory
        m = self.dropout(self.activation(self.linear3(m)))
        As = self.dropout(self.linearN2(m)) # [B, L, N, N]
        m = self.dropout(self.linear4(m))
        
        # gcn
        # GCN input: x [B,D,N] Ahat [B,N,N] 
        dts = []
        for l in range(L):
            A = As[:,l,:,:] # [B,N,N]
            dt = data[:,l,:,:] # [B,N,D]
            n_support = support+[A]
            n_dt = self.gcns[l](dt.permute(0,2,1), n_support) # [B,D,N]
            dts.append(n_dt.transpose(2,1)) #[B,N,D]
        dts = torch.stack(dts, dim=1) # [B,L,N,D]

        return self.norm_x2(x+y), self.norm_m2(m+memory), dts, None

class TransLayer3(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, pred_len,
                 nodes=207, d_ff=None, support_len=1, order=2,
                 dropout=0.1, activation="relu"):
        super(TransLayer3, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear3 = nn.Linear(d_model, d_ff)
        self.linear4 = nn.Linear(d_ff, d_model)
        self.linearN2 = nn.Linear(d_ff, nodes)
        self.norm_x1 = nn.LayerNorm(d_model)
        self.norm_x2 = nn.LayerNorm(d_model)
        self.norm_s1 = nn.LayerNorm(d_model)
        self.norm_s2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.gcns = nn.ModuleList(
            [
                GCN_t2s(
                    d_model, d_model, 
                    dropout=dropout, support_len=support_len, order=order
                ) for t in range(pred_len)
            ]
        )

    def forward(self, x, memory, data, Sinfo, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None, support=None):
        # X [B, L, N, D] data [B, L, N, D]
        # memory [B, L, N, D] [N,N]->[B,L,N,D]
        
        # Normalize the masks
        B = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((B,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask,
        ))
        x = self.norm_x1(x) # [B, L, N, D]
        
        # Secondly apply the cross attention and add it to the previous output
        Sinfo = Sinfo + self.dropout(self.cross_attention(
            x, memory, memory,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        )) # 此处是memory永远不变 而x是时序数据的不断混合 而Sinfo会不断往上叠加信息
        # [B, L, N, D]
        Sinfo = self.norm_s1(Sinfo)
        
        # Finally run the fully connected part of the layer
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        S = Sinfo
        S = self.dropout(self.activation(self.linear3(S)))
        As = self.dropout(self.linearN2(S))
        S = self.dropout(self.linear4(S))
        
        # gcn
        # As = self.dropout(self.activation(self.linearN1(S))) # [B, L, N, N]
        # As = self.dropout(self.linearN2(As))
        # GCN input: x [B,D,N] Ahat [B,N,N] 
        dts = []
        for l in range(L):
            A = As[:,l,:,:] # [B,N,N]
            dt = data[:,l,:,:] # [B,N,D]
            n_support = support+[A]
            n_dt = self.gcns[l](dt.permute(0,2,1), n_support) # [B,D,N]
            dts.append(n_dt.transpose(2,1)) #[B,N,D]
        dts = torch.stack(dts, dim=1) # [B,L,N,D]

        return self.norm_x1(x+y), memory, dts, self.norm_s2(Sinfo+S)

class Trans(nn.Module):
    def __init__(self, layers, skip_convs, temporal_layer, norm_layer=None):
        super(Trans, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.skip_convs = nn.ModuleList(skip_convs)
        self.temporal_layer = temporal_layer
        self.norm = norm_layer

    def forward(self, x, memory, data, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None, support=None):
        # Normalize the masks
        B = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask  or \
            LengthMask(x.new_full((B,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((B,), L_prime, dtype=torch.int64))

        # Apply all the transformer decoders
        skip = 0.
        Sinfo = memory
        for layer, skip_conv in zip(self.layers, self.skip_convs):
            x, memory, dt, Sinfo = layer(x, memory, data, Sinfo, x_mask=x_mask, x_length_mask=x_length_mask,
                      memory_mask=memory_mask,
                      memory_length_mask=memory_length_mask,
                      support=support)
            # dt [B,L,N,D]
            # s = skip_conv(x.transpose(-1,1)).transpose(-1,1) # 此处原来写的是x, 所以这里应该是写错了
            # 既然原来是写错了, 那么原来的实验其实是只考虑了时间信息, 一点空间信息都没考虑的模型的结果了
            # 即既没有GCN, 算Attn也是只在时间维度进行, 所以trans2和3的结果基本一样
            # 而trans1则多了一个attn, 相当于多了一层, 而且还是和空间信息一起算的, 相当于引入了空间信息
            # 有意思的是trans1的效果要比GCN好, 不过比diff差
            s = skip_conv(dt.transpose(-1,1)).transpose(-1,1)
            skip = skip + s

        # skip = self.temporal_layer(skip, x_mask, x_length_mask, support)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(F.relu(skip))

        return x

class T2S_trans_skip(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=8, nodes=207,
              trans_type='trans1', aptinit=None, supports=None, order=2, dropout=0.0, activation='gelu', device=torch.device('cuda:0')):
        super(T2S_trans_skip, self).__init__()
        
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.spatial_embedding = nn.Linear(nodes, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        trans_map = {'trans1':TransLayer, 'trans2':TransLayer2, 'trans3':TransLayer3,
                    'trans1N':TransLayer, 'trans2N':TransLayer2, 'trans3N':TransLayer3}
        T_layer = trans_map[trans_type]

        # GCN supports
        self.supports = supports
        support_len = 0 if supports is None else len(supports)
        support_len += 1

        # encoder
        self.encoder = Trans(
            [
                T_layer(
                    STAttentionT2SLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        dropout=dropout),
                    STAttentionT2SLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, 
                                        dropout=dropout),
                    d_model=d_model,
                    pred_len=out_len,
                    nodes=nodes,
                    d_ff=d_ff, 
                    support_len=support_len, 
                    order=order,
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
            temporal_layer=None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=d_model, out_channels=c_out, kernel_size=(1,1), bias=True)
    
    def forward(self, x, x_mark, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D]
        support = [s.repeat(x.shape[0],1,1) for s in self.supports]
        memory = self.supports[0].repeat(x.shape[0],x.shape[1],1,1)

        out = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        out = self.embedding_dropout(out)
        memory = self.spatial_embedding(memory) + self.position_embedding(memory)
        memory = self.embedding_dropout(memory)
        out = self.encoder(x=out, memory=memory, data=out, 
                           x_mask=attn_mask, x_length_mask=length_mask,
                           memory_mask=attn_mask, memory_length_mask=length_mask,
                           support=support) # [B, L, N, D]
        out = F.relu(self.end_conv1(out)) # [B, OL, N, D]
        out = self.end_conv2(out.transpose(-1,1)).transpose(-1,1) # [B, OL, N, OD]
        
        return out # [B, L, N, D]        
        

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