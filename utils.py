import os
import torch
import argparse
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class Args(argparse.Namespace): # TODO : delete this class.
    """
    Helper bundle of parameters class. To be provided to a Training object.
    """
    def __init__(self, 
                 batch_size=10, n_tokens=64, n_layers=4, n_heads=4, 
                 d_model=128, use_lr_decay=True, lr=1e-3, dataset_path="./datasets/shakespear_corpus.txt", 
                 max_iter=100, out_dir=None, n_warm_iters=100, 
                 lr_decay_iter=5000, min_lr=1e-4, 
                 n_validation_batch=200, betas=(0.9, 0.99), 
                 n_epochs=10, val_int = 100
        ):
        
        self.batch_size = batch_size
        """Number of (sentence, shifted) pairs per gradient step."""
        self.n_tokens = n_tokens
        """Size of context/attention window"""
        self.n_layers = n_layers
        """Number of stacked blocks in the Transformer"""
        self.n_heads = n_heads
        """Number of attention heads"""
        self.d_model = d_model
        """Embeddings dimension"""
        self.use_lr_decay = use_lr_decay
        """Wether to use Attention is All you need lr schedule."""
        self.lr = lr
        """Adam learning rate"""
        self.dataset_path: str  = dataset_path
        """Path towards .txt file."""
        self.max_iter = max_iter
        """Maximum number of gradient updates."""
        self.out_dir = out_dir
        """Where to save metric results, plots."""
        self.n_warm_iters = n_warm_iters
        self.lr_decay_iter = lr_decay_iter
        """No of iter during which lr will decay, lr=min_lr"""
        self.min_lr = min_lr
        """Minimum value of lr"""
        self.n_validation_batch = n_validation_batch   
        """No of samples of val set to compute val loss on."""
        self.betas = betas
        """Adam gradient gradient averaging parameters"""
        self.n_epochs = n_epochs
        """Number of times to iterate on dataset."""
        self.val_int = val_int
        """Interval of iters to pass before computing val loss."""

    @staticmethod
    def parse_args() -> argparse.Namespace: 
        default_args = Args()
        parser = argparse.ArgumentParser()
        # TODO : refactor to for loop
        parser.add_argument("--batch_size", "-b", 
                            help=f"Batch size (default: {default_args.batch_size}).", 
                            type=int, default=default_args.batch_size,)
        
        parser.add_argument("--n_tokens", "-n", 
                            help=f"Number of tokens (default: {default_args.n_tokens}).", 
                            type=int, default=default_args.n_tokens,)
        
        parser.add_argument("--n_layers", "-l", help=f"Number of layers (default: {default_args.n_layers}).", 
                            type=int, default=default_args.n_layers,)
        
        parser.add_argument("--n_heads", help=f"Number of heads (default: {default_args.n_heads}).", 
                            type=int, default=default_args.n_heads,)
        
        parser.add_argument("--d_model", "-d", help=f"Dimension of model (default: {default_args.d_model}).", 
                            type=int, default=default_args.d_model,)
        
        parser.add_argument("--lr", "-lr", help=f"Learning Rate (default: {default_args.lr}).", 
                            type=float, default=default_args.lr,)
        
        parser.add_argument("--use_lr_decay", help=f'''Use learning rate decay strategy 
                            (default: {default_args.use_lr_decay}).''', type=bool, 
                            default=default_args.use_lr_decay,)
        
        parser.add_argument("--lr_decay_iter", help=f'''No of iter during which lr will decay 
                            (default: {default_args.lr_decay_iter}).''', type=bool, 
                            default=default_args.lr_decay_iter,)

        parser.add_argument("--dataset_path", help=f'''Dataset file to use for 
                            training (default: {default_args.dataset_path}).''', type=str, 
                            default=default_args.dataset_path,)
        
        parser.add_argument("--max_iter", help=f'''Maximum Number of iterations for 
                            training (default: {default_args.max_iter}).''', 
                            type=int, default=default_args.max_iter,)
        
        parser.add_argument("--betas", nargs='?', help=f'''Adam moving average beta1, beta2 
                            (default: {default_args.betas}).''', 
                            type=tuple[float], default=default_args.betas,)

        parser.add_argument("--n_epochs", nargs='?', help=f'''Number of times to iterate on dataset
                    (default: {default_args.n_epochs}).''', 
                    type=int, default=default_args.n_epochs,)
        
        parser.add_argument("--n_warm_iters", nargs='?', help=f'''Number of warmup iterations of lr schedule 
                    (default: {default_args.n_warm_iters}).''', 
                    type=int, default=default_args.n_warm_iters,)


        parser.add_argument("--min_lr", nargs='?', help=f'''Minimum lr value 
                    (default: {default_args.min_lr}).''', 
                    type=int, default=default_args.min_lr,)

        parser.add_argument("--n_validation_batch", nargs='?', help=f'''Batch size of 
                    validation loss computation (default: {default_args.n_validation_batch}).''', 
                    type=int, default=default_args.n_validation_batch,)

        parser.add_argument("--val_int", nargs='?', help=f'''Interval of iters to pass 
                    before computing val loss (default: {default_args.val_int}).''', 
                    type=int, default=default_args.val_int,)

        parser.add_argument("--out_dir", nargs='?', help=f'''Directory containing the saved 
                            models (default: {default_args.out_dir}).''', type=str, 
                            default=default_args.out_dir,)

        return parser.parse_args()     # this will contain all Args type values


class CharDataSet(Dataset):
    """
    Helper class to emits batches of characters. Implements an a __getitem__() method, which 
    allows subscripting by i, where CharDataSet[i] yields a tuple with the i-th sliding window 
    and the i+1-th window on the data.
    """
    def __init__(self, N_tokens: int, fold: int = None, k_fold: int = None, dataset_path: str = None, is_training: bool = True, raw_data: str = None, p_train: int=0.9):
        """Instantiate a char dataset, if raw_data is provided, the fulltext is assumed to be raw_data, if not
        the file pointed to by path is opened and it's lines read. If is_training is True, __getitem__() will
        yield training dataset tuples, otherwise they will be validation dataset tuples. p_train is the percentage
        of training data."""

        self.is_training: bool = is_training
        self.fold: int = fold
        self.N_tokens: int = N_tokens
        
        data_indices: Tensor = None
        if raw_data == None:                                        # if no raw_data, load from path.
            self.raw_data: str = load_data(dataset_path)
            if os.path.isfile(dataset_path + '.npy'):
                data_indices = torch.from_numpy(np.load(dataset_path + '.npy').flatten())
        else:
            self.raw_data: str = raw_data
            
        self.vocabulary: list[str] = sorted(list(set(self.raw_data)))
        self.vocabulary_size: int = len(self.vocabulary)
        
        self.encoder: dict[str, int] = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.decoder: dict[int, str] = {i: ch for i, ch in enumerate(self.vocabulary)}

        if data_indices == None:                                    # raw_data was not provided
            data_indices = self.encode(self.raw_data).flatten()     # always flatten
            if dataset_path != None:
                np.save(dataset_path, arr=data_indices)                     # e.g. path provided .txt file exists but no .npy file exists 

        # drop last characters to have multiple of N_tokens
        data_indices = data_indices[:(len(data_indices)-(len(data_indices) % N_tokens))+1]

        if fold != None and k_fold != None:
            # cross validation will segment |train0|train1|...|test|train_i|...|train_n, n = len(chunks)/k_fold
            fraction = 1/k_fold                           # test chunk is somewhere in the middle of the train data
            length_test = int(len(data_indices) * fraction)     # it is from character [val_start, val_end] of length seg.

            val_start = fold * length_test                # val_start is index of start of validation chunk, NOTE : this is not test data per say, as model
            val_end = val_start + length_test             # selection will be done on it. Test performance is done on data the model has NEVER encountered
                                                        # and hence has NEVER influenced the model.
            print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" % (0, val_start, val_end, len(data_indices), val_start, val_end))   
            
            train_left_indices = list(range(val_start))
            train_right_indices = list(range(val_end, len(data_indices)))
            
            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(val_start, val_end))
            
            self.train_chunks = data_indices[train_indices]
            self.validation_chunks = data_indices[val_indices]  
        else:
            n = len(data_indices)
            self.train_chunks = data_indices[:int(n*0.9)]
            self.validation_chunks = data_indices[int(n*0.9):]

                               
    def get_vocab_size(self):
        return len(self.vocabulary)

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
    print(path)
    with open(path, 'r') as file:
        data = "".join(file.readlines())
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


def getLoaderDataset(N, B, dataset_path, fold, k_fold, is_training=True, shuffle=True):
    tokenized_data = CharDataSet(N, fold, k_fold, dataset_path, is_training)
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

