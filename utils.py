import torch
import numpy as np
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
    def __init__(self, N_tokens, path): 
        self.N_tokens = N_tokens         
        self.raw_data: str = load_data(path)
        self.vocabulary: list[str] = sorted(list(set(self.raw_data)))
        self.vocabulary_size: int = len(self.vocabulary)
        self.encoder: dict[str, int] = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.decoder: dict[int, str] = {i: ch for i, ch in enumerate(self.vocabulary)}
        
        data_indices = np.load(path+'.npy').flatten()
        
        data_indices = data_indices[: ( len(data_indices) - (len(data_indices) % N_tokens)) + 1]   # drop last characters to have multiple of N_tokens
        self.chunks = torch.from_numpy(data_indices)

    def get_vocab_size(self):
        return self.vocabulary_size

    def __len__(self):
        return self.chunks.size(0) - self.N_tokens  # number of encoded sentences loaded

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """Grabs sliding window chunk n°idx of N_token characters
        from the data starting at character n°idx, encodes every 
        character to an integer and returns the chunk and the 
        shifted version as tensors.
        """
        return self.chunks[idx:idx+self.N_tokens], self.chunks[idx+1:idx+1+self.N_tokens]   # (N_token,), (N_token,) tuple.

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

def load_data(path):
    with open(path, 'r') as file:
        data = file.read()
    return data

def encodeDataset(path):
    """Encode character dataset path to single numpy vector of indices."""
    text = load_data(path)
    voc = sorted(list(set(text)))
    encoder: dict[str, int] = {ch: i for i, ch in enumerate(voc)}

    idx = np.zeros(shape=(1, len(text)), dtype=np.uint8)      # TODO : Generalize to larger vocabulary sizes
    for i, char in enumerate(text):
        idx[0, i] = encoder[char]
    np.save(path, arr=idx)


def getLoaderDataset(N, B, path):
    tokenized_data = CharDataSet(N, path)
    
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
        char_id = torch.distributions.categorical.Categorical(logits=logits[0, -1, :]).sample()

        new_c = torch.tensor(char_id).reshape(shape=(1, 1))
        input = torch.concat( (input, new_c), dim=-1 )
    return input.flatten()

import time

if __name__ == '__main__':
    path = './datasets/shakespear_corpus.txt'
    loader, dataset = getLoaderDataset(N=256, B=10, path=path)

    for batch_idx, (inputs, targets) in enumerate(loader):
        print(dataset.decode(inputs[0][:10]), "|", dataset.decode(targets[0][:10]))
        break
        #print(batch_idx, dataset.decode(inputs[0]))