import torch
import torch.nn as nn
from torch.nn import functional as F


class FFN(nn.Module):
    def __init__(self, d, d_ff):
        """Initializes a transformer block FFN module"""
        super(FFN, self).__init__()
        self.d = d
        self.d_ff = d_ff
        self.L1 = nn.Linear(in_features=d, out_features=d_ff)
        self.L2 = nn.Linear(in_features=d_ff, out_features=d)

    def forward(self, X):
        """Applies fully connected two layer network to tensor X 
        of size (B x N x d) where B is the batch size, N the number
        of tokens of the sentence and d the embedding dimension."""
        X = self.L1(X)
        X = F.relu(X)
        X = self.L2(X)
        return X
