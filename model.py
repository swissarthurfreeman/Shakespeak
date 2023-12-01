
import torch
from torch import nn

from PositionalEncoding import WPE
from Transformer import Transformer


class ShakespearModel(nn.Module):
    def __init__(self, n_layers, n_heads, d, d_ff, d_k, d_v, batch_size, N_tokens, vocabulary_size):
        super(ShakespearModel, self).__init__()
        self.blocks = []
        self.transformer: Transformer = Transformer(L=n_layers,
                                                    B=batch_size,
                                                    N=N_tokens,
                                                    h=n_heads,
                                                    d=d,
                                                    d_k=d_k,
                                                    d_v=d_v,
                                                    d_ff=d_ff,
                                                    V=vocabulary_size)
        self.d = d
        self.WPE = WPE(self.d)
        self.WTE = nn.Embedding(vocabulary_size, d)

    def forward(self, idx, target=None):
        batch_size, N_tokens = idx.size()
        positions = torch.arange(0, N_tokens).expand(
            batch_size, N_tokens).to(idx.device)
        position_embedding = self.WPE(positions)
        token_embedding = self.WTE(idx.long())
        return self.transformer(token_embedding + position_embedding)

    def generate(self, idx, n_new_tokens: int):

        # loop up to the number of desired new tokens.
        for _ in range(n_new_tokens):
            # Get logits
            logits = self(idx)

            # TODO - Get probabilities from logits
            probabilities = None
            # Take element with highest probability
            # TODO - Implement other way to choose the element
            _, new_char_idx = torch.topk(probabilities, k=1, dim=-1)
            # Update text with new element
            idx = torch.cat((idx, new_char_idx), dim=1)
        return idx
