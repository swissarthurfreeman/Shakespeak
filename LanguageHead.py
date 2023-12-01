import torch
import torch.nn as nn
from torch.nn import functional as F


class LanguageHead(nn.Module):
    def __init__(self, V, d):
        """Initialize a language head module."""
        super(LanguageHead, self).__init__()
        self.E: nn.Linear = nn.Linear(d, V, bias=False)

    def forward(self, X):
        """Compute tensor of logits from X a B x N x d tensor, return a B x N x V tensor where 
        output[0][i] is the probability distribution over the vocabulary of token w_i+1."""
        logits = torch.matmul(X, torch.transpose(
            self.E.weight(), dim0=0, dim1=1))   # B x N x V
        logits = F.softmax(logits, dim=2)
        return logits
