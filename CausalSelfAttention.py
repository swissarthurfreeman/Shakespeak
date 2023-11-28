import torch.nn.functional as F
import torch.nn as nn
import torch

class CausalSelfAttention(nn.Module):
    def __init__(self, B, N, d, h, d_k, d_v):
        self.B = B
        self.N = N
        self.d = d
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        
        W_Q = torch.zeros(size=(h, d, d_k))
        nn.init.xavier_uniform(W_Q)
        W_Q = nn.Parameter(W_Q)

        W_K = torch.zeros(size=(h, d, d_k))
        nn.init.xavier_uniform(W_K)
        W_K = nn.Parameter(W_K)

        W_V = torch.zeros(size=(h, d, d_v))
        nn.init.xavier_uniform(W_V)
        W_K = nn.Parameter(W_V)

        W_O = torch.zeros(size=(h*d_v, d))
        nn.init.xavier_uniform(W_O)
        W_K = nn.Parameter(W_O)

    def forward(self, X):
        Q = torch.einsum('bik,hkj->bhij', X, self.W_Q)
        K = torch.einsum('bik,hkj->bhij', X, self.W_K)
        V = torch.einsum('bik,hkj->bhij', X, self.W_V)

        QKT = torch.einsum('bhik,bhkj->bhij', Q, torch.transpose(K, 2, 3))
        A = F.softmax(QKT, dim=2)
        
        AV = torch.einsum('bhik,bhkj->bhij', A, V)
        AV_concat = torch.reshape(AV, shape=(self.B, self.N, self.h*self.d_v))
        SA_out = torch.einsum('bni,id->bnd', AV_concat, self.W_O)
        return SA_out
