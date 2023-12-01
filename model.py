
import torch
from torch import nn

from PositionalEncoding import WPE
from Transformer import Transformer


class ShakespearModel(nn.Module):
    """
    ShakespearModel class defines a Transformer's decoder-only language model for generating text.

    Args:
    - n_layers (int): Number of layers in the Transformer.
    - n_heads (int): Number of attention heads in the Transformer.
    - d (int): Dimension of model embeddings.
    - d_ff (int): Dimension of the feedforward layer in the Transformer.
    - d_k (int): Dimension of the key vectors in attention heads.
    - d_v (int): Dimension of the value vectors in attention heads.
    - batch_size (int): Size of the input batches.
    - N_tokens (int): Number of tokens in the input sequence.
    - vocabulary_size (int): Size of the vocabulary.

    Attributes:
    - d (int): Dimension of model embeddings.
    - WPE (WPE): Positional encoding layer.
    - WTE (nn.Embedding): Token embedding layer.
    - transformer (Transformer): Transformer model.
    """

    def __init__(self, n_layers: int, n_heads: int, d: int, d_ff: int, d_k: int, d_v: int, batch_size: int, N_tokens: int, vocabulary_size: int) -> None:
        super(ShakespearModel, self).__init__()
        self.d = d
        self.WPE = WPE(self.d)
        self.WTE = nn.Embedding(vocabulary_size, d)
        self.transformer: Transformer = Transformer(L=n_layers,
                                                    B=batch_size,
                                                    N=N_tokens,
                                                    h=n_heads,
                                                    d=d,
                                                    d_k=d_k,
                                                    d_v=d_v,
                                                    d_ff=d_ff,
                                                    V=vocabulary_size,
                                                    E=self.WTE)

    def forward(self, idx: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - idx (torch.Tensor): Input sequence indices.
        - target (torch.Tensor): Target sequence indices for training.

        Returns:
        - torch.Tensor: Model output.
        """
        batch_size, N_tokens = idx.size()
        positions = torch.arange(0, N_tokens).expand(
            batch_size, N_tokens).to(idx.device)
        position_embedding = self.WPE(positions)
        token_embedding = self.WTE(idx.long())
        return self.transformer(token_embedding + position_embedding)

    def generate(self, idx: torch.Tensor, n_new_tokens: int) -> torch.Tensor:
        """
        Generate new text based on the input sequence.

        Args:
        - idx (torch.Tensor): Input sequence indices.
        - n_new_tokens (int): Number of new tokens to generate.

        Returns:
        - torch.Tensor: Generated sequence indices.
        """
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
