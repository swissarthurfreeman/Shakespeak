from torch.nn import functional as F
import torch.nn as nn
import torch

class Droupout(nn.Module):
    def __init__(self, P):
        """Initializes a transformer block FFN module"""
        self.P = P
        
    def foward(self, X):
        return X