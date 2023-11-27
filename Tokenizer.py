from torch.utils.data import Dataset
import torch


class CharDataset(Dataset):
    """Emits batches of characters."""
    def __init__(self, N_tokens, data):
        self.N_tokens = N_tokens
        self.raw_data: str = data
        chars = list(set(data)) # get characters from the input data
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices

    def get_vocab_size(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        """Grabs a chunk of (block_size + 1) characters from the data,
        encodes every character to an integer and returns the chunk 
        and the shifted version as tensors.
        """
        text_chunk = self.raw_data[idx:idx+self.N_tokens]
        shifted_text_chunk = self.raw_data[idx+1:idx+1+self.N_tokens]

        chunk_idx = torch.zeros(size=(1, self.N_tokens))
        shifted_idx = torch.zeros(size=(1, self.N_tokens))

        for i, char in enumerate(text_chunk):
            chunk_idx[i] = self.stoi(char)
        
        for i, char in enumerate(shifted_text_chunk):
            shifted_idx[i] = self.stoi(char)

        return chunk_idx, shifted_idx

        