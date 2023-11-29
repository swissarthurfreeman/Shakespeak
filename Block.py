from CausalSelfAttention import CausalSelfAttention
from LayerNorm import LayerNorm
from FFN import FFN
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, B, N, h, d, d_k, d_v, d_ff):
        self.CausalSelfAttn = CausalSelfAttention(B, N, d, h, d_k, d_v)
        self.LayerNorm_1 = LayerNorm()
        self.FFN = FFN(d, d_ff)
        self.LayerNorm_2 = LayerNorm()
        
    def foward(self, X):
        """Apply a transformer block to a (B, N, d) batch tensor."""
        X = X + self.CausalSelfAttn(self.LayerNorm_1(X))
        out = X + self.FFN(self.LayerNorm_2(X))
        return out