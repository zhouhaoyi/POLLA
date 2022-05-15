import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from util.masking import FullMask, LengthMask
from models.gcnexp import GCN_diff, GCN

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, heads=8, layers=1, feature_map=None, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps
        
    def forward(self, queries, keys, values, attn_mask, attn_scores, query_lengths, key_lengths):
        # Apply the feature map to the queries and keys
        # [B, L*N, D]
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("bnhd,bnhm->bhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("bnhd,bhd->bnh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("bnhd,bhmd,bnh->bnhm", Q, KV, Z)
        
        return None, V.contiguous()

class FullAttention(nn.Module):
    def __init__(self, heads=8, layers=1, softmax_temp=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, attn_scores, query_lengths, key_lengths):
        # Extract some shapes and compute the temperature
        B, NL, H, E = queries.shape
        _, NS, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("blhd,bshd->bhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # Make sure that what we return is contiguous
        return None, V.contiguous()

class FullDenseAttention(nn.Module):
    def __init__(self, heads=8, layers=1, softmax_temp=None, attention_dropout=0.1):
        super(FullDenseAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, attn_scores, query_lengths, key_lengths):
        # Extract some shapes and compute the temperature
        B, NL, H, E = queries.shape
        _, NS, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("blhd,bshd->bhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix

        # QK [B, H, L, L]; A [B, H, L, L];
        # attn_scores [B, H, L, L]
        if attn_scores is not None:
            attn_scores = QK+attn_scores
        else:
            attn_scores = QK
        # attn_scores [B, H, L, L]
        
        A = self.dropout(torch.softmax(softmax_temp * attn_scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # Make sure that what we return is contiguous
        return attn_scores, V.contiguous()

class STDenseAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, support_len=1, order=2, dropout=0.0, attn_type='add'):
        super(STDenseAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.attn_type = attn_type

    def forward(self, queries, keys, values, attn_mask, attn_scores,
                query_lengths, key_lengths, support):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L*N1, H, -1) # [B, L*N, H, d]
        keys = self.key_projection(keys).view(B, S*N2, H, -1)
        values = self.value_projection(values).view(B, S*N2, H, -1)
        
        # Compute the attention
        FM, new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            attn_scores,
            query_lengths,
            key_lengths
        )
        
        new_values = new_values.view(B, N1, L, -1)
        new_values = new_values.transpose(2,1) # [B, L, N1, D]

        if self.attn_type=='cat':
            if attn_scores is not None:
                FM = torch.cat([FM, attn_scores], dim=2)
        
        return FM, self.out_projection(new_values)