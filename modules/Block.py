import torch.nn as nn
from modules.CausalSelfAttention import CausalSelfAttention
from modules.FFN import FFN
from modules.LayerNorm import LayerNorm


class Block(nn.Module):
    def __init__(self, B, N, h, d, d_k, d_v, d_ff):
        super().__init__()

        self.CausalSelfAttn = CausalSelfAttention(B, N, d, h, d_k, d_v)
        self.LayerNorm_1 = LayerNorm()
        self.FFN = FFN(d, d_ff)
        self.LayerNorm_2 = LayerNorm()

    def forward(self, X):
        """Apply a transformer block to a (B, N, d) batch tensor."""
        X = X + self.CausalSelfAttn(self.LayerNorm_1(X))
        out = X + self.FFN(self.LayerNorm_2(X))
        return out
