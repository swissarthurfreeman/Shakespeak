import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self):
        """Initializes a LayerNorm Module"""
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((1)))
        self.beta = nn.Parameter(torch.zeros((1)))

    def forward(self, X):
        """Applies Layer normalization to tensor X of size (B x N x d)"""
        mu = torch.mean(X, dim=2).unsqueeze(-1)
        std = torch.std(X, dim=2).unsqueeze(-1)

        X_hat = (X - mu) / std
        return self.gamma * X_hat + self.beta
