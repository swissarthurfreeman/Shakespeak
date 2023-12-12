import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class WPE(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        pos = x.unsqueeze(2)
        i = torch.arange(self.d).to(x.device)
        angles = pos * (1 / torch.pow(10000, (2 * i) / self.d))

        torch.sin_(angles[:, :, 0::2])  # Sinus of even elements
        torch.cos_(angles[:, :, 1::2])  # Cosinus of odd elements
        return angles


class Block(nn.Module):
    def __init__(self, h, d, d_ff):
        super().__init__()
        
        self.CausalSelfAttn = CausalSelfAttention(d, d_qk=d//h, d_v=d//h, h=h, p=0.2)
        #self.CausalSelfAttn = QKVAttention(d, dim_qk=d//h, dim_v=d//h, nb_heads=h, causal=True, attention_dropout=0)
        self.W1 = nn.Linear(d, d_ff)
        self.W2 = nn.Linear(d_ff, d)

        self.LayerNorm1 = nn.LayerNorm(d)
        self.LayerNorm2 = nn.LayerNorm(d)

    def forward(self, X):
        X = X + self.CausalSelfAttn(self.LayerNorm1(X))
        out = X + self.W2(F.relu(self.W1(self.LayerNorm2(X))))
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, d, d_qk, d_v, h, p):
        super().__init__()
        self.h = h
        self.d_qk = d_qk
        self.d_v = d_v
        
        self.W_Q = nn.Linear(in_features=d, out_features=h*d_qk)
        self.W_K = nn.Linear(in_features=d, out_features=h*d_qk)
        self.W_V = nn.Linear(in_features=d, out_features=h*d_v)
        self.W_O = nn.Linear(in_features=h*d_v, out_features=d)

    def forward(self, X: Tensor):          # assume input is Z x B x N x d or N x d

        Q: Tensor = self.W_Q(X)            # B x N x h*d_qkv -> B x N x h x d_qkv
        Q = Q.reshape(shape=(X.size(0), X.size(-2), self.h, self.d_qk))
        Q = Q.transpose(dim0=1, dim1=2).contiguous()
        
        K: Tensor = self.W_K(X)
        K = K.reshape(shape=(X.size(0), X.size(-2), self.h, self.d_qk))
        K = K.transpose(dim0=1, dim1=2).contiguous()

        V = self.W_V(X)
        V: Tensor = V.reshape(shape=(X.size(0), X.size(-2), self.h, self.d_v))
        V = V.transpose(dim0=1, dim1=2).contiguous()
        # UPDATE HERE IF CUDA SUPPORTS F.scaled_dot_product_attention
        AV: Tensor = scaled_dot_product_attention(query=Q, key=K, value=V, is_causal=True) # dim B x h x N x d_v
        AV = torch.transpose(AV, dim0=1, dim1=2)    # B x N x h x d_v
        AV = torch.reshape(AV, shape=(-1, X.size(-2), self.h * self.d_v))   # B x N x h*d_v, head concat
        SA_out = self.W_O(AV)
        return SA_out

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None) -> torch.Tensor:
    # see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value