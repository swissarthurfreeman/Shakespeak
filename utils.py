import time
import numpy as np
import torch
from torch import nn
import argparse
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class Args(argparse.Namespace):
    def __init__(self, batch_size, n_tokens, n_layers, n_heads, d_model, use_lr_decay, learning_rate, dataset, max_iterations, out_dir):
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.use_lr_decay = use_lr_decay
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.max_iterations = max_iterations
        self.out_dir = out_dir
        
class LoadedModel(nn.Module):
    def __init__(self, checkpoint_path, model):
        super(LoadedModel, self).__init__()
        
        # Load the model weights from the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        # Set the model attribute
        self.model = model

    def forward(self, x):
        pass
        # Define the forward pass of your model
        # ...
        
    def get_model(self):
        return self.model


class CharDataSet(Dataset):
    """
    Helper class to emits batches of characters.
    Implements an a __getitem__() method, which 
    allows subscripting by i, where CharDataSet[i]
    yields a tuple with the i-th sliding window 
    and the i+1-th window on the data.   
    """

    def __init__(self, N_tokens: int, path: str, fold: int, k_fold: int, is_training: bool):
        self.is_training: bool = is_training
        self.fold: int = fold
        self.N_tokens: int = N_tokens
        self.raw_data: str = load_data(path)
        self.vocabulary: list[str] = sorted(list(set(self.raw_data)))
        
        self.encoder: dict[str, int] = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.decoder: dict[int, str] = {i: ch for i, ch in enumerate(self.vocabulary)}

        data_indices = np.load(path+'.npy').flatten()

        # drop last characters to have multiple of N_tokens
        data_indices = data_indices[:(len(data_indices) - (len(data_indices) % N_tokens)) + 1]
        
        chunks = torch.from_numpy(data_indices)       # cross validation will segment |train0|train1|...|test|train_i|...|train_n, n = len(chunks)/k_fold
        fraction = 1/k_fold                           # test chunk is somewhere in the middle of the train data
        length_test = int(len(chunks) * fraction)     # it is from character [val_start, val_end] of length seg.

        val_start = fold * length_test                # val_start is index of start of validation chunk, NOTE : this is not test data per say, as model
        val_end = val_start + length_test             # selection will be done on it. Test performance is done on data the model has NEVER encountered
                                                      # and hence has NEVER influenced the model.
        print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" % (0, val_start, val_end, len(chunks), val_start, val_end))   
        
        train_left_indices = list(range(val_start))
        train_right_indices = list(range(val_end, len(chunks)))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(val_start, val_end))
        
        self.train_chunks = chunks[train_indices]
        self.validation_chunks = chunks[val_indices]
        

    def get_vocab_size(self):
        return len(self.vocabulary)

    def __len__(self):
        # number of encoded sentences loaded
        chunks = self.train_chunks if self.is_training else self.validation_chunks
        return chunks.size(0) - self.N_tokens

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """Grabs sliding window chunk n°idx of N_token characters
        from the data starting at character n°idx, encodes every 
        character to an integer and returns the chunk and the 
        shifted version as tensors.
        """
        chunks = self.train_chunks if self.is_training else self.validation_chunks
        # (N_token,), (N_token,) tuple.
        return chunks[idx:idx+self.N_tokens], chunks[idx+1:idx+1+self.N_tokens]

    def encode(self, text: str) -> Tensor:
        """Map string of characters to vector of indexes."""
        idx = torch.zeros(size=(1, len(
            text)), dtype=torch.float32)      # BUG : before we were using self.N_tokens, but this is not expected behavior.
        for i, char in enumerate(text):
            idx[0, i] = self.encoder[char]
        return idx

    def decode(self, idx: Tensor) -> str:
        """Decode list of character token indexes as string."""
        chars = [self.decoder[i] for i in idx.tolist(
        )]         # why was there an if i != 0 in the list ?
        # this made decoding of spaces impossible
        return ''.join(chars)


def load_data(path):
    with open(path, 'r') as file:
        data = file.read()
    return data


def encodeDataset(path):
    """Encode character dataset path to single numpy vector of indices."""
    text = load_data(path)
    voc = sorted(list(set(text)))
    encoder: dict[str, int] = {ch: i for i, ch in enumerate(voc)}

    # TODO : Generalize to larger vocabulary sizes
    idx = np.zeros(shape=(1, len(text)), dtype=np.uint8)
    for i, char in enumerate(text):
        idx[0, i] = encoder[char]
    np.save(path, arr=idx)


def getLoaderDataset(N, B, path, fold, k_fold, is_training=True, shuffle=True):
    tokenized_data = CharDataSet(N, path, fold, k_fold, is_training)
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
