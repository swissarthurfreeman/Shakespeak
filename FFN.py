from torch.nn import functional as F
import torch.nn as nn
import torch

class FFN(nn.Module):
    def __init__(self, d, d_ff):
        """Initializes a transformer block FFN module"""
        self.d = d
        self.d_ff = d_ff
        self.W1 = nn.Linear(in_features=d, out_features=d_ff)
        self.W2 = nn.Linear(in_features=d_ff, out_features=d)

    def foward(self, X):
        """Applies fully connected two layer network to tensor X 
        of size (B x N x d) where B is the batch size, N the number
        of tokens of the sentence and d the embedding dimension."""
        X = torch.matmul(X, self.W1)
        X = F.relu(X)
        X = torch.matmul(X, self.W2)
        return X