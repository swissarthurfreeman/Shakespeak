import torch.nn as nn

from Block import Block
from LanguageHead import LanguageHead
from LayerNorm import LayerNorm
from Tokenizer import CharDataset


class Transformer(nn.Module):
    def __init__(self, L, B, N, h, d, d_k, d_v, d_ff, V):
        super(Transformer, self).__init__()

        self.Dropout = nn.Dropout(p=0.1)

        # L blocks stacked on top of each other
        self.blocks = [Block(B, N, h, d, d_k, d_v, d_ff) for _ in range(L)]

        self.Final_LayerNorm = LayerNorm()
        self.LM_Head = LanguageHead(V, d)

    def forward(self, X):
        """X is a (B x N x d) matrix of token and position embeddings
        where X_bi = embedding[w_i] + embedding_position[i], B is 
        the batch size and N the number of characters per sentence. 
        w_i is character n°i, returns a B x N x V tensor out where
        out[k][i] is the probability distribution over the vocabulary for
        token w_i+1 of the k-th sentence of the batch."""
        X = self.Dropout(X)

        for block in self.blocks:
            X = block(X)

        X = self.Final_LayerNorm(X)
        X = self.LM_Head(X)
        return X
