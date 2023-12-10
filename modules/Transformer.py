import torch
import torch.nn as nn
from modules.Block import Block
from modules.LanguageHead import LanguageHead
from modules.LayerNorm import LayerNorm


class Transformer(nn.Module):
    def __init__(self, L, B, N, h, d, d_k, d_v, d_ff, V):
        super().__init__()

        self.Dropout = nn.Dropout(p=0.2)

        # L blocks stacked on top of each other, use module list as 
        # python lists are not seen by Pytorch. 
        self.blocks = nn.ModuleList()
        for _ in range(L): 
            self.blocks.append(Block(B, N, h, d, d_k, d_v, d_ff))

        self.Final_LayerNorm = nn.LayerNorm(normalized_shape=d)
        self.LM_Head = LanguageHead(d, V)

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

    def forward(self, X):
        """
        Parameters
        ----------
        - X is a (B x N x d) matrix of token and position embeddings where \\
          X[b][i] = embedding[w_i] + pos_embedding[i], B is the batch size.
        - N the number of characters per batch, e.g. the sliding window size. 
        - w_i is character n°i in the window. 
        
        Returns
        -------
        - B x N x V tensor out where out[b][i] is the probability distribution \\
          over the vocabulary for token w_i+1 of the b-th sequence in the batch."""
        X = self.Dropout(X)

        for block in self.blocks:
            X = block(X)

        X = self.Final_LayerNorm(X)
        X = self.LM_Head(X)
        return X
