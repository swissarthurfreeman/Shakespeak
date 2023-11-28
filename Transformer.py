from LayerNorm import LayerNorm
from Dropout import Droupout
from Block import Block
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, L, B, N, h, d, d_k, d_v, d_ff):
        self.Dropout = Droupout()
        self.Final_LayerNorm = LayerNorm()

        # L blocks stacked on top of each other
        for _ in range(L): 
            self.blocks.append(Block(B, N, h, d, d_k, d_v, d_ff))
        

    def foward(self, X):
        """X is a (B x N x d) matrix of token and position embeddings
        where X_bi = embedding[w_i] + embedding_position[i], B is 
        the batch size and N the number of characters per sentence. 
        w_i is character n°i."""

        X = self.Dropout(X)
        for block in self.blocks:
            X = block.foward(X)
        X = self.Final_LayerNorm(X) 

        # X is B x N x d transformed output
        # logits is a B x 
        logits = LM_Head(X)
        return X