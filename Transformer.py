import torch.nn as nn
from Block import Block
from LayerNorm import LayerNorm

class Transformer(nn.Module):
    def __init__(self, 
            in_dim, N_layers, N_sa_heads, N_tokens
        ):
        self.N = N_tokens
        self.d = in_dim
        self.L = N_layers
        self.h = N_sa_heads
        self.blocks = nn.ModuleList()

        # L blocks stacked on top of each other
        for _ in range(self.L): 
            self.blocks.append(Block(self.N, self.d, self.h))
        
        self.Final_LayerNorm = LayerNorm()     

    def foward(self, X):
        """X is a (B x N x d) matrix of token and position embeddings
        where X_bi = embedding[w_i] + embedding_position[i], B is 
        the batch size and N the number of characters per sentence. 
        w_i is character n°i."""
        for block in self.blocks:
            X = block.foward(X)
        X = self.Final_LayerNorm(X)
        return X