import math
import torch
import torch.nn as nn
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
        
        self.CausalSelfAttn = QKVAttention(dim_in=d, dim_qk=d//h, dim_v=d//h, nb_heads=h, causal=True, attention_dropout=0)
        self.W1 = nn.Linear(d, d_ff)
        self.W2 = nn.Linear(d_ff, d)

        self.LayerNorm1 = nn.LayerNorm(d)
        self.LayerNorm2 = nn.LayerNorm(d)

    def forward(self, X):
        X = X + self.CausalSelfAttn(self.LayerNorm1(X))
        out = X + self.W2(F.relu(self.W1(self.LayerNorm2(X))))
        return out
    

class QKVAttention(nn.Module):
    def __init__(
        self, dim_in, dim_qk, dim_v, nb_heads=1, causal=True, attention_dropout=0.0
    ):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.attention_dropout = attention_dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q

        q = torch.einsum("ntc,hdc->nhtd", x_q, self.w_q)
        k = torch.einsum("ntc,hdc->nhtd", x_kv, self.w_k)
        v = torch.einsum("ntc,hdc->nhtd", x_kv, self.w_v)

        a = torch.einsum("nhtd,nhsd->nhts", q, k) / math.sqrt(q.size(3))

        if self.causal:
            forbidden_attention = (
                torch.arange(a.size(2), device=q.device)[None, None, :, None]
                < torch.arange(a.size(3), device=q.device)[None, None, None, :]
            )
            a = a.masked_fill(forbidden_attention, float("-inf"))

        a = a.softmax(dim=3)
        y = torch.einsum("nhts,nhsd->nthd", a, v).flatten(2)

        y = y @ self.w_o

        return y
