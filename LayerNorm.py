import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self):
        """Initializes a LayerNorm Module"""
        self.gamma = nn.Parameter(torch.zeros((1)))
        self.beta = nn.Parameter(torch.zeros((1)))

    def foward(self, X):
        """Applies Layer normalization to tensor X of size (B x N x d)"""
        mu = torch.mean(X, dim=2).unsqueeze(-1)
        std = torch.mean(X, dim=2).unsqueeze(-1)

        X_hat = (X - mu) / std
        return self.gamma * X_hat + self.beta