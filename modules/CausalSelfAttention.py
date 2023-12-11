import torch
from math import sqrt
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, B, N, d, h, d_k, d_v):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.B = B
        self.N = N
        self.d = d
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        # weights are d_k*h x d
        self.to_Q = nn.Linear(d, d_k * h)   # since they're h heads, standard practice is to
        # weights are d_v*h x d
        self.to_K = nn.Linear(d, d_k * h)   # concatenate all heads Q, K, V weights and reshape
        # weights are d_v*h x d
        self.to_V = nn.Linear(d, d_v * h)   # them to h x N x d_vk later. 
        # weights are d x h*d_v
        self.W_O  = nn.Linear(h*d_v, d)      # W_O.weight will be (d, h*d_v)

    def ValuesMatrix(self, X):
        V: Tensor = self.to_V(X)
        V = Tensor.contiguous(torch.transpose(
            V.reshape(shape=(self.B, self.N, self.h, self.d_v)),
            dim0=1,
            dim1=2
        ))  # V is (B x h x N x d_v)
        return V

    def ScoreMatrix(self, X):
        Q: Tensor = self.to_Q(X)    # Q is (B x N x d_k * h)

        Q = Tensor.contiguous(torch.transpose(  
            #                   0       1       2       3
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

        # (B x h x N x N)
        QKT = torch.div(
            torch.einsum('bhik,bhkj->bhij', Q, torch.transpose(K, 2, 3)), 
            torch.sqrt(torch.tensor(self.d_k)).item() 
        )    
        return QKT
        
    def AttentionMatrix(self, X):
        QKT = self.ScoreMatrix(X)
        
        lower_mask = torch.ones(size=(self.N , self.N)).tril()
        upper_mask = torch.zeros(size=(self.B, self.h, self.N, self.N)).masked_fill_(lower_mask.logical_not(), float("-inf"))
        
        lower_mask = lower_mask.to(self.device)
        upper_mask = upper_mask.to(self.device)
        QKT = QKT.to(self.device)
        QKT = QKT * lower_mask + upper_mask
        
        # A is (B x h x N x einsumN), careful, softmax over last dimension
        A = F.softmax(QKT , dim=3)
        return A

    def forward(self, X):
        """Applies multi-head self attention mechanism to B x N x d tensor.
        Outputs a tensor of logits """
        A = self.AttentionMatrix(X) # (B x h x N x N)
        V = self.ValuesMatrix(X)    # (B x h x N x d_v)

        
        # V is (B x h x N x d_v), AV is (B x h x N x d_v)
        AV = torch.einsum('bhik,bhkj->bhij', A, V)
        AV = torch.transpose(AV, dim0=1, dim1=2)    # AV is (B x N x h x d_v)
        
        # AV_concat is (B x N x h*d_v)
        AV_concat = torch.reshape(AV, shape=(self.B, self.N, self.h*self.d_v))
        
        #print(AV_concat.size(), self.W_O.weight.size())
        # W_O.weight is (h*d_v, d) (stored transpose), SA_out is (B x N x d)
        SA_out = self.W_O(AV_concat) # torch.einsum('bni,id->bnd', AV_concat, self.W_O.weight)
        return SA_out
