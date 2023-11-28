from torch.nn import functional as F
import torch.nn as nn
import torch

class LanguageHead(nn.Module):
    def __init__(self, E):
        """Initialize a language head module."""
        self.E: nn.Linear = E

    def foward(self, X):
        """Compute tensor of logits from X a B x N x d tensor, return a B x N x V tensor where 
        output[0][i] is the probability distribution over the vocabulary of token w_i+1."""
        logits = torch.matmul(X, torch.transpose(self.E.weight(), dim0=0, dim1=1))   # B x N x V
        logits = F.softmax(logits, dim=2)
        return logits

