import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from util.masking import FullMask, LengthMask
from models.gcn import GCN_diff, GCN

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class STLinearAttention(nn.Module):
    def __init__(self, feature_map=None, eps=1e-6):
        super(STLinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps
        
    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        # Apply the feature map to the queries and keys
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, None, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("bnshd,bnshm->bnhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("bnlhd,bnhd->bnlh", Q, K.sum(dim=2))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("bnlhd,bnhmd,bnlh->bnlhm", Q, KV, Z)
        
        return V.contiguous()

class STFullAttention(nn.Module):
    def __init__(self, softmax_temp=None, attention_dropout=0.1):
        super(STFullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        # Extract some shapes and compute the temperature
        B, N, L, H, E = queries.shape
        _, _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("bnlhd,bnshd->bnhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("bnhls,bnshd->bnlhd", A, values)

        # Make sure that what we return is contiguous
        return V.contiguous()

class STGCNdiffAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, support_len=1, order=2, dropout=0.0):
        super(STGCNdiffAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        self.gcn1 = GCN_diff(d_model, d_model, dropout=dropout, support_len=support_len, order=order)
        self.gcn2 = GCN_diff(d_model, d_model, dropout=dropout, support_len=support_len, order=order)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, support):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        gcn_keys = self.gcn1(keys.transpose(-1, 1), support) # [B, D, N, T]
        keys = self.key_projection(gcn_keys.transpose(-1,1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        gcn_values = self.gcn2(values.transpose(-1, 1), support) # [B, D, N, T]
        values = self.value_projection(gcn_values.transpose(-1,1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        
        queries = queries.transpose(2,1) # [B, N, L, H, d]
        keys = keys.transpose(2,1) # [B, N, S, H, d]
        values = values.transpose(2,1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(B, N1, L, -1)
        
        new_values = new_values.transpose(2,1) # [B, L, N1, D]
        
        # Project the output and return
        return self.out_projection(new_values)


class STGCNAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, dropout=0.0, nodes=None):
        super(STGCNAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        self.gcn1 = GCN(d_model, d_model, dropout=dropout, nodes=nodes)
        self.gcn2 = GCN(d_model, d_model, dropout=dropout, nodes=nodes)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, support):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape # [B, S, N, D]
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        gcn_keys = self.gcn1(keys.transpose(2, 1), support) # [B, N, S, D]
        keys = self.key_projection(gcn_keys.transpose(2,1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        gcn_values = self.gcn2(values.transpose(2, 1), support) # [B, D, N, T]
        values = self.value_projection(gcn_values.transpose(2,1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        
        queries = queries.transpose(2,1) # [B, N, L, H, d]
        keys = keys.transpose(2,1) # [B, N, S, H, d]
        values = values.transpose(2,1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(B, N1, L, -1)
        
        new_values = new_values.transpose(2,1) # [B, L, N1, D]
        
        # Project the output and return
        return self.out_projection(new_values)