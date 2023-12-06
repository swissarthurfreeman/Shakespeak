import torch
from torch import nn
from torch import Tensor
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
        self.d = d
        self.N_tokens = N_tokens
        self.B = batch_size
        self.WPE = WPE(self.d)
        self.WTE = nn.Embedding(vocabulary_size, d)
        self.transformer: Transformer = Transformer(L=n_layers, B=batch_size, N=N_tokens,
                                                    h=n_heads, d=d, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                                    V=vocabulary_size, E=self.WTE)

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
        batch_size, N_tokens = idx.size()   
        positions = torch.arange(0, N_tokens).expand(batch_size, N_tokens).to(idx.device)
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
        prompt_idx = idx.size(1)                                                       # pointer to start of new tokens buffer
        input = torch.concat([idx, torch.zeros( size=(1, n_new_tokens) )], dim=-1)     # size is (1, N_prompt + n_new_tokens), just pad. 

        if prompt_idx > self.N_tokens:  # blablab(la .... blabla 0) 0 0 0 ... 0        window () of N_tokens must be sliced out to have 
            input = input[prompt_idx-self.N_tokens:]                                   # maximum context characters to extrapolate from
        
        N_input = input.size(-1)

        if N_input < (self.N_tokens*self.B) and N_input % (self.N_tokens*self.B) != 0:               # pad to length multiple of N_tokens*B
            pad = torch.zeros(size=(1, self.N_tokens*self.B - (N_input % (self.N_tokens*self.B) )))
            input = torch.concat([input, pad], dim=-1)                                               # input (l * N_tokens * B)
        else:
            print("Too many tokens to generate, max len(prompt) + n_new_tokens > %s" % (self.N_tokens*self.B) )
            print("Please reduce the n_new_tokens or the prompt size.")
            return

        input = torch.reshape(input, shape=(self.B, self.N_tokens))  # input is tensor of size N_batches x B x N_tokens
        print(input.size())
        
        for _ in range(n_new_tokens):
            print(input.size())
            # Get logits
            logits = self(input)                                    # output is B x N x V

            if sampling == "max":                                   # TODO : try other sampling strategies
                new_char_idx = torch.argmax(logits, dim=-1)         # B x N

            shape = input.size()
            input = torch.flatten(input)                            # tensor of size k*N_input
            print(input.size(), prompt_idx)
            input[prompt_idx] = new_char_idx.flatten()[prompt_idx]
            prompt_idx += 1

            input = torch.reshape(input, shape=shape)

        return input.flatten()[idx.size(0):idx.size(0)+n_new_tokens]


