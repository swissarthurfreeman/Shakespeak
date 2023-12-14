import time
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class CharDataSet(Dataset):
    """
    Helper class to emits batches of characters. Implements an a __getitem__() method, which 
    allows subscripting by i, where CharDataSet[i] yields a tuple with the i-th sliding window 
    and the i+1-th window on the data.
    """
    def __init__(self, N_tokens: int, path: str = None, is_training: bool = True, raw_data: str = None, p_train: int=0.9):
        """Instantiate a char dataset, if raw_data is provided, the fulltext is assumed to be raw_data, if not
        the file pointed to by path is opened and it's lines read. If is_training is True, __getitem__() will
        yield training dataset tuples, otherwise they will be validation dataset tuples. p_train is the percentage
        of training data."""
        self.is_training: bool = is_training
        self.N_tokens: int = N_tokens
        
        data_indices: Tensor = None
        if raw_data == None:                                        # if no raw_data, load from path.
            self.raw_data: str = load_data(path)
            if os.path.isfile(path + '.npy'):
                data_indices = torch.from_numpy(np.load(path + '.npy').flatten())
        else:
            self.raw_data: str = raw_data
            
        self.vocabulary: list[str] = sorted(list(set(self.raw_data)))
        self.vocabulary_size: int = len(self.vocabulary)
        
        self.encoder: dict[str, int] = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.decoder: dict[int, str] = {i: ch for i, ch in enumerate(self.vocabulary)}

        if data_indices == None:                                    # raw_data was not provided
            data_indices = self.encode(self.raw_data).flatten()     # always flatten
            if path != None:
                np.save(path, arr=data_indices)                     # e.g. path provided .txt file exists but no .npy file exists 

        # drop last characters to have multiple of N_tokens
        data_indices = data_indices[:(len(data_indices)-(len(data_indices) % N_tokens))+1]

        n = len(data_indices)
        self.train_chunks = data_indices[:int(n*p_train)]         # if p_val=0.9, 90% train
        self.validation_chunks = data_indices[int(n*p_train):]    # 10% validation
                                                  
    def get_vocab_size(self):
        return self.vocabulary_size

    def __len__(self):
        # number of encoded sentences loaded
        chunks = self.train_chunks if self.is_training else self.validation_chunks
        print(chunks.size(0), self.N_tokens)
        return chunks.size(0) - self.N_tokens
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        '''Grabs sliding window chunk n°idx of N_token characters
        from the data starting at character n°idx, encodes every 
        character to an integer and returns the chunk and the 
        shifted version as tensors.
        '''
        chunks = self.train_chunks if self.is_training else self.validation_chunks
        # (N_token,), (N_token,) tuple, chunks must be flattened
        return chunks[idx:idx+self.N_tokens], chunks[idx+1:idx+1+self.N_tokens]
    
    def encode(self, text: str) -> Tensor:
        """Map string of characters to vector of indexes.
        Returns a (1, len(text)) tensor."""
        idx = torch.zeros(size=(1, len(
            text),), dtype=torch.float32)      
        for i, char in enumerate(text):
            idx[0, i] = self.encoder[char]     # BUG : why is this a 1 x len(text) vector ? Why not len(text) vector ?  
        return idx

    def decode(self, idx: Tensor) -> str:
        """Decode list of character token indexes as string."""
        chars = [self.decoder[i] for i in idx.tolist(
        )]         # why was there an if i != 0 in the list ?
        # this made decoding of spaces impossible
        return ''.join(chars)

def load_data(path):
    with open(path, 'r') as file:
        data = "".join(file.readlines())
    return data


def getLoaderDataset(N, B, path, is_training=True, shuffle=True):
    tokenized_data = CharDataSet(N_tokens=N, path=path, is_training=is_training)
    data_loader = DataLoader(
        tokenized_data,
        shuffle=shuffle,
        pin_memory=True,
        batch_size=B,
        num_workers=2,
        drop_last=True
    )

    return data_loader, tokenized_data


@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = idx.to(device)
    for k in range(max_new_tokens):
        logits = model(input)
        char_id = torch.distributions.categorical.Categorical(
            logits=logits[0, -1, :]).sample()

        new_c = torch.tensor(char_id).reshape(shape=(1, 1))
        input = torch.concat((input, new_c), dim=-1)
    return input.flatten()


if __name__ == '__main__':
    path = './datasets/shakespear_corpus.txt'
    loader, dataset = getLoaderDataset(N=256, B=10, path=path)

    for batch_idx, (inputs, targets) in enumerate(loader):
        print(dataset.decode(inputs[0][:10]), "|",
              dataset.decode(targets[0][:10]))
        break
