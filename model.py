import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modules.PositionalEncoding import WPE
from modules.Transformer import Transformer


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
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.d = d
        self.N_tokens = N_tokens
        self.B = batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.WPE = WPE(self.d)
        self.WTE = nn.Embedding(vocabulary_size, d)
        self.transformer: Transformer = Transformer(L=n_layers, B=batch_size, N=N_tokens,
                                                    h=n_heads, d=d, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                                    V=vocabulary_size)

    def forward(self, idx: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
        - idx Tensor, B x N: Input sequence indices.

        Returns:
        - out : Tensor, a tensor of logits of size B x N x V
          where out[b][i] is the probability distribution over sorted vocabulary
          of character i+1 in sequence n°b of the batch.
        """
        idx = idx.to(self.device)
        batch_size, N_tokens = idx.size()

        # positions is B x N, every line is a range[0, N_tokens]
        positions = torch.arange(0, N_tokens).expand(
            batch_size, N_tokens).to(idx.device)

        position_embedding = self.WPE(positions)
        token_embedding = self.WTE(idx.long())

        return self.transformer(token_embedding + position_embedding)

    def generate(self, idx: Tensor, n_new_tokens: int, sampling: str = "max") -> Tensor:
        """
        Generate new text based on the input sequence.

        Args:
        - idx (torch.Tensor): Input sequence indices of arbitrary length.
        - n_new_tokens (int): Number of new tokens to generate.

        Returns:
        - Tensor: Generated new sequence indices, a 1D vector of size (n_new_tokens,).
        """
        with torch.no_grad():
            if idx.size(1) + n_new_tokens > self.B * self.N_tokens:
                print(
                    "Too many tokens to fit in attention window, n_new_tokens + len(seed) > B * N_tokens")
                return

            N_pad = self.B * self.N_tokens - idx.size(1)
            input = torch.concat([idx, torch.zeros(size=(1, N_pad))], dim=1)
            input = torch.reshape(input, (self.B, self.N_tokens))

            idx_char = idx.size(1)
            for _ in range(n_new_tokens):
                p_input = self.forward(input)
                p_input = F.softmax(p_input, dim=-1)    # B x N x V
                if sampling == "max":
                    # print(p_input[0][idx_char].argmax())
                    # print(p_input[0][idx_char])
                    # print(p_input[0][idx_char].max())
                    # B x N, [b][i] contains idx of w_i+1
                    p_input = torch.argmax(p_input, dim=-1)

                input.flatten()[idx_char] = p_input.flatten()[idx_char]
                idx_char += 1

            return input.flatten()[0:idx.size(1)+n_new_tokens]

    def param_norm(self):
        with torch.no_grad():
            vec = torch.zeros(size=(0,))
            for p in self.parameters():
                vec = torch.concat([vec, p.flatten()])
            return torch.norm(vec)
