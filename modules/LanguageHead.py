import torch
import torch.nn as nn
from torch.nn import functional as F


class LanguageHead(nn.Module):
    def __init__(self, d, V):
        """Initialize a language head module."""
        super().__init__()
        self.to_V = nn.Linear(d, V)

    def forward(self, X):
        """Compute tensor of logits from X a B x N x d tensor, return a B x N x V tensor where 
        output[0][i] is the probability distribution over the vocabulary of token w_i+1."""
        logits = self.to_V(X)   # B x N x V
        # logits = F.softmax(logits, dim=2)
        return logits
