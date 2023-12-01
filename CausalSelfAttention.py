import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, B, N, d, h, d_k, d_v):
        super(CausalSelfAttention, self).__init__()

        self.B = B
        self.N = N
        self.d = d
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.to_Q = nn.Linear(d, d_k * h)
        self.to_K = nn.Linear(d, d_k * h)
        self.to_V = nn.Linear(d, d_v * h)
        self.W_O = nn.Linear(h*d_v, d)

    def forward(self, X):
        """Applies multi-head self attention mechanism to B x N x d tensor."""
        Q: Tensor = self.to_Q(X)

        Q = Tensor.contiguous(torch.transpose(  # 0       1       2       3
            Q.reshape(shape=(self.B, self.N, self.h, self.d_k)),
            dim0=1,
            dim1=2
        ))   # Q is (B x h x N x d_k)

        K: Tensor = self.to_K(X)
        K = Tensor.contiguous(torch.transpose(
            K.reshape(shape=(self.B, self.N, self.h, self.d_k)),
            dim0=1,
            dim1=2  # 0   1   2   3
        ))   # K is (B x h x N x d_k)

        V: Tensor = self.to_V(X)
        V = Tensor.contiguous(torch.transpose(
            V.reshape(shape=(self.B, self.N, self.h, self.d_v)),
            dim0=1,
            dim1=2
        ))  # V is (B x h x N x d_v)

        # (B x h x N x N)
        QKT = torch.einsum('bhik,bhkj->bhij', Q, torch.transpose(K, 2, 3))
        # A is (B x h x N x N), careful, softmax over last dimension
        A = F.softmax(QKT, dim=3)

        # V is (B x h x N x d_v), AV is (B x h x N x d_v)
        AV = torch.einsum('bhik,bhkj->bhij', A, V)
        AV = torch.transpose(AV, dim0=1, dim1=2)    # AV is (B x N x h x d_v)

        # AV_concat is (B x N x h*d_v)
        AV_concat = torch.reshape(AV, shape=(self.B, self.N, self.h*self.d_v))
        # W_O is (h*d_v, d), SA_out is (B x N x d)
        SA_out = torch.einsum('bni,id->bnd', AV_concat,
                              torch.transpose(self.W_O.weight, dim0=0, dim1=1))
        return SA_out
