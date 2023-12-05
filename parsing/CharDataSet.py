import torch
from torch import Tensor
from torch.utils.data import Dataset        


class CharDataSet(Dataset):
    """
    Helper class to emits batches of characters.
    Implements an a __getitem__() method, which 
    allows subscripting by i, where CharDataset[i]
    yields a tuple with the i-th sliding window 
    and the i+1-th window on the data.   
    """
    def __init__(self, N_tokens, data):
        self.N_tokens = N_tokens        # CAREFUL ! Top bound is not included ! 
        self.raw_data: str = data
        self.vocabulary: list[str] = sorted(list(set(data)))
        self.vocabulary_size: int = len(self.vocabulary)
        self.encoder: dict[str, int] = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.decoder: dict[int, str] = {i: ch for i, ch in enumerate(self.vocabulary)}

    def get_vocab_size(self):
        return self.vocabulary_size

    def __len__(self):
        return len(self.raw_data) - self.N_tokens

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """Grabs sliding window chunk n°idx of N_token characters
        from the data starting at character n°idx, encodes every 
        character to an integer and returns the chunk and the 
        shifted version as tensors.
        """
        text_chunk = self.raw_data[idx:idx+self.N_tokens]
        shifted_text_chunk = self.raw_data[idx+1:idx+1+self.N_tokens]

        chunk_idx = self.encode(text_chunk).squeeze()
        shifted_idx = self.encode(shifted_text_chunk).squeeze()
        return chunk_idx, shifted_idx   # 1 x N_token, 1 x N_token tuple.

    def encode(self, text: str) -> Tensor:
        """Map string of characters to vector of indexes."""
        idx = torch.zeros(size=(1, self.N_tokens))
        for i, char in enumerate(text):
            idx[0, i] = self.encoder[char]
        return idx

    def decode(self, idx: Tensor) -> str:
        """Decode list of character token indexes as string."""
        chars = [self.decoder[i] for i in idx.tolist()]         # why was there an if i != 0 in the list ?  
        return ''.join(chars)                                   # this made decoding of spaces impossible 
