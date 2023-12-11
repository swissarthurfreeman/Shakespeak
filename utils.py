import torch
import random
from torch import Tensor
from torch.utils.data import Dataset        
from torch.utils.data.dataloader import DataLoader


class CharDataSet(Dataset):
    """
    Helper class to emits batches of characters.
    Implements an a __getitem__() method, which 
    allows subscripting by i, where CharDataSet[i]
    yields a tuple with the i-th sliding window 
    and the i+1-th window on the data.   
    """
    def load_data(self, path):
        with open(path, 'r') as file:
            data = file.read()
        return data

    def __init__(self, N_tokens, path, N_samples): # N_samples is the number of (chunk, shifted) pairs we're preloading. 
        self.N_tokens = N_tokens         
        self.raw_data: str = self.load_data(path)
        self.vocabulary: list[str] = sorted(list(set(self.raw_data)))
        self.vocabulary_size: int = len(self.vocabulary)
        self.encoder: dict[str, int] = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.decoder: dict[int, str] = {i: ch for i, ch in enumerate(self.vocabulary)}
        
        self.chunks: Tensor = None  # N_samples x N
        self.shifts: Tensor = None  # N_samples x N

        for idx in random.sample( range(len(self.raw_data) - self.N_tokens ), N_samples ):
            text_chunk: Tensor = self.raw_data[idx:idx+self.N_tokens]
            shifted_text_chunk: Tensor = self.raw_data[idx+1:idx+1+self.N_tokens]
            
            chunk_idx = self.encode(text_chunk).squeeze().reshape((1, -1))
            shifted_idx = self.encode(shifted_text_chunk).squeeze().reshape((1, -1))
            
            if self.chunks == None:
                self.chunks = chunk_idx
                self.shifts = shifted_idx
            else:
                self.chunks = torch.concat([self.chunks, chunk_idx], dim=0)     # do not move these to device, results in bug
                self.shifts = torch.concat([self.shifts, shifted_idx], dim=0)   # data loader does this if pin_memory = True

    def get_vocab_size(self):
        return self.vocabulary_size

    def __len__(self):
        return self.chunks.size(0)  # number of encoded sentences loaded

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """Grabs sliding window chunk n°idx of N_token characters
        from the data starting at character n°idx, encodes every 
        character to an integer and returns the chunk and the 
        shifted version as tensors.
        """
        return self.chunks[idx, :], self.shifts[idx, :]   # (N_token,), (N_token,) tuple.

    def encode(self, text: str) -> Tensor:
        """Map string of characters to vector of indexes."""
        idx = torch.zeros(size=(1, len(text)), dtype=torch.float32)      # BUG : before we were using self.N_tokens, but this is not expected behavior.
        for i, char in enumerate(text):
            idx[0, i] = self.encoder[char]
        return idx

    def decode(self, idx: Tensor) -> str:
        """Decode list of character token indexes as string."""
        chars = [self.decoder[i] for i in idx.tolist()]         # why was there an if i != 0 in the list ?  
        return ''.join(chars)                                   # this made decoding of spaces impossible 


def getLoaderDataset(N, B, path, N_samples):
    tokenized_data = CharDataSet(N, path, N_samples)
    
    data_loader = DataLoader(
        tokenized_data,
        shuffle=True,
        pin_memory=True,
        batch_size=B,
        num_workers=2,
        drop_last=True
    )

    return data_loader, tokenized_data

@torch.no_grad()
def generate(model, idx, max_new_tokens):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = idx.to(device)
    for k in range(max_new_tokens):
        logits = model(input) 
        
        #char_id = torch.multinomial(logits[0, -1, :], num_samples=1).item()
        
        char_id = torch.distributions.categorical.Categorical(logits=logits[0, -1, :]).sample()

        new_c = torch.tensor(char_id).reshape(shape=(1, 1))
        input = torch.concat( (input, new_c), dim=-1 )
    return input.flatten()

