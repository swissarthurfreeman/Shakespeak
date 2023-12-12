import torch
import torch.nn as nn
from modules import WPE
from modules import Block
from torch.nn import Module

class GPT(Module):
    def __init__(self, B, L, d, d_ff, N, h, V):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d = d
        self.d_ff = d_ff
        self.N = N
        self.h = h
        self.B = B

        self.WTE = nn.Embedding(V, d)
        self.WPE = WPE(d)
        self.blocks = nn.ModuleList()

        for _ in range(L):
            self.blocks.append(Block(h, d, d_ff))
        
        self.Dropout = nn.Dropout(0.2)
        self.Final_LayerNorm = nn.LayerNorm(d)
        self.LM_Head = nn.Linear(d, V)
        self.to(self.device)
    
    def forward(self, X):
        X = X.long().to(self.device)
        tok_emb = self.WTE(X)
        pos_emb = self.WPE(X)

        X = self.Dropout(pos_emb + tok_emb)

        for block in self.blocks:
            X = block(X)

        X = self.Final_LayerNorm(X)
        X = self.LM_Head(X)
        return X
