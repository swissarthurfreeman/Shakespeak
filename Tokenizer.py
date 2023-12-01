import torch
from torch import Tensor
from torch.utils.data import Dataset


class Encoder:
    def __init__(self, vocabulary, N_tokens: int):
        sorted_vocabulary = sorted(vocabulary)
        self.encoder = {ch: i for i, ch in enumerate(sorted_vocabulary)}
        self.decoder = {i: ch for i, ch in enumerate(sorted_vocabulary)}
        self.N_tokens = N_tokens

    def encode(self, text: str) -> Tensor:
        """Map string of characters to vector of indexes."""
        idx = torch.zeros(size=(1, self.N_tokens))
        for i, char in enumerate(text):
            idx[0, i] = self.encoder[char]
        return idx

    def decode(self, idx: list) -> str:
        """Decode list of character token indexes as string."""
        chars = [self.decoder[i] for i in idx if i != 0]
        return ''.join(chars)


class CharDataset(Dataset):
    """Emits batches of characters."""

    def __init__(self, N_tokens, data):
        self.N_tokens = N_tokens
        self.raw_data: str = data
        vocabulary = list(set(data))
        self.encoder: Encoder = Encoder(vocabulary, N_tokens)
        self.vocabulary_size = len(vocabulary)

    def get_vocab_size(self):
        return self.vocabulary_size

    def __len__(self):
        return len(self.raw_data) - self.N_tokens

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """Grabs sliding window chunk n°i of N_token characters from 
        the data, encodes every character to an integer and returns 
        the chunk  and the shifted version as tensors.
        """
        text_chunk = self.raw_data[idx:idx+self.N_tokens]
        shifted_text_chunk = self.raw_data[idx+1:idx+1+self.N_tokens]

        chunk_idx = self.encode(text_chunk).squeeze()
        shifted_idx = self.encode(shifted_text_chunk).squeeze()
        return chunk_idx, shifted_idx   # 1 x N_token, 1 x N_token tuple.

    def decode(self, idx) -> str:
        """Decode list of character token indexes as string."""
        return self.encoder.decode(idx.tolist())

    def encode(self, text) -> Tensor:
        """Map string of characters to vector of indexes."""
        return self.encoder.encode(text)
