import torch
import torch.nn as nn
from Block import Block
from LayerNorm import LayerNorm
from LanguageHead import LanguageHead
from Tokenizer import CharDataset

class Transformer(nn.Module):
    def __init__(self, d, L, h, N, data):
        self.N = N
        self.d = d
        self.L = L
        self.h = h
        self.data: CharDataset = data
        self.E = nn.Linear(self.data.get_vocab_size(), self.d)   # initial embeddings matrix
        self.blocks = nn.ModuleList()
        
        for _ in range(self.L):         # L blocks stacked on top of each other 
            self.blocks.append(Block(self.N, self.d, self.h))
        
        self.Final_LayerNorm = LayerNorm()     
        self.LM_Head = LanguageHead(self.E)

    def foward(self, X):
        """X is a (B x N x d) matrix of token and position embeddings
        where X_bi = embedding[w_i] + embedding_position[i], B is 
        the batch size and N the number of characters per sentence. 
        w_i is character n°i, returns a B x N x V tensor out where
        out[k][i] is the probability distribution over the vocabulary for
        token w_i+1 of the kth sentence of the batch."""
        for block in self.blocks:
            X = block.foward(X)
        X = self.Final_LayerNorm(X)
        X = self.LM_Head(X)
        return X