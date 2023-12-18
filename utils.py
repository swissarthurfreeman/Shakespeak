import os
import torch
import argparse
import numpy as np
from modules import GPT
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class Args(argparse.Namespace): 
    """
    Helper bundle of parameters class. 
    Allows passing parameters from jupyter notebook.
    To be provided to a Training object.
    """
    def __init__(self, 
                 batch_size=10, n_tokens=64, n_layers=4, n_heads=4, 
                 d_model=128, use_lr_decay=True, lr=1e-3, dataset_path="./datasets/shakespear_corpus.txt", 
                 max_iter=100, out_dir="./runs/", n_warm_iters=100, 
                 lr_decay_iter=5000, min_lr=1e-4, 
                 n_validation_batch=200, betas=(0.9, 0.99), 
                 n_epochs=10, val_int = 100, save=True, save_int=200, name="milkshake",
                 cross_val=False, k_fold=10,
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
        """Number of gradient updates to perform."""
        self.out_dir = out_dir
        """Where to save models, results, plots."""
        self.n_warm_iters = n_warm_iters
        self.lr_decay_iter = lr_decay_iter
        """No of iter during which lr will decay, after which lr=min_lr"""
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
        self.save = save
        """Wether to save the model every save_int steps."""
        self.save_int = save_int
        """No of ints before saving model state to file"""
        self.name = name
        """Name of the model / run """
        self.cross_val = cross_val
        """Wether to run cross-validation or not."""
        self.k_fold = k_fold
        """Number of k-folds."""

    @staticmethod
    def parse_args() -> argparse.Namespace: 
        default_args = Args()
        parser = argparse.ArgumentParser()
        # TODO : refactor to for loop
        parser.add_argument("--batch_size", "-b", 
                            help=f"Batch size (default: {default_args.batch_size}).", 
                            type=int, default=default_args.batch_size,)
        
        parser.add_argument("--n_tokens", "-N", 
                            help=f"Number of tokens (default: {default_args.n_tokens}).", 
                            type=int, default=default_args.n_tokens,)
        
        parser.add_argument("--n_layers", "-L", help=f"Number of layers (default: {default_args.n_layers}).", 
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

        parser.add_argument("--name", nargs='?', help=f'''Name of the model / run 
                            (default: {default_args.name}).''', type=str, 
                            default=default_args.name,)
        
        parser.add_argument("--save_int", nargs='?', help=f'''No of ints before saving model state to file. BEWARE ! Large
                            models can be HUNDREDS OF MEGABYTES large, so saving every 10 iterations will fill your drive. 
                            (default: {default_args.save_int}).''', type=int, 
                            default=default_args.save_int,)

        parser.add_argument("--save", nargs='?', help=f'''Wether to save the model every save_int steps 
                            (default: {default_args.save}).''', type=bool, 
                            default=default_args.save,)
        
        parser.add_argument("--cross_val", nargs='?', help=f'''Run cross-validation  
                            (default: {default_args.cross_val}).''', type=bool, 
                            default=default_args.cross_val,)

        parser.add_argument("--k_fold", nargs='?', help=f'''Number of k-folds  
                            (default: {default_args.k_fold}).''', type=int, 
                            default=default_args.k_fold,)
        
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
            #print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" % (0, val_start, val_end, len(data_indices), val_start, val_end))   
            
            train_left_indices = list(range(val_start))
            train_right_indices = list(range(val_end, len(data_indices)))
            
            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(val_start, val_end))
            
            self.train_idx = data_indices[train_indices]
            self.valid_idx = data_indices[val_indices]  
        else:
            n = len(data_indices)
            self.train_idx = data_indices[:int(n*0.9)]
            self.valid_idx = data_indices[int(n*0.9):]

                               
    def get_vocab_size(self):
        return len(self.vocabulary)

    def __len__(self):
        # number of encoded sentences loaded
        chunks = self.train_idx if self.is_training else self.valid_idx
        return chunks.size(0) - self.N_tokens
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        '''Grabs sliding window chunk n°idx of N_token characters
        from the data starting at character n°idx, encodes every 
        character to an integer and returns the chunk and the 
        shifted version as tensors.
        '''
        chunks = self.train_idx if self.is_training else self.valid_idx
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
        chars = [self.decoder[i] for i in idx.tolist()]
        return ''.join(chars)

def load_data(path):
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
def generate(model: GPT, tokenized_data: CharDataSet, prompt: str, max_new_tokens) -> str:
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = tokenized_data.encode(prompt).to(device)
    for k in range(max_new_tokens):
        logits = model(input)
        char_id = torch.distributions.categorical.Categorical(
            logits=logits[0, -1, :]  # let Categorical deal with the logits
        ).sample()

        new_c = char_id.reshape(shape=(1, 1))
        input = torch.concat((input, new_c), dim=-1)
    return tokenized_data.decode(input.flatten())

def cv_losses_graph(train_loss: Tensor, val_loss: Tensor, val_int: str, path: str = None, 
                    save: bool = False, name: str = None, args: argparse.Namespace = None):
    """
    Plot cross-validation train/validation losses w.r.t batch index.
    Plots variance areas over folds. 
    """
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    train_mean = train_loss.mean(dim=0)    # (k-fold, n_steps) -> (1 x n_steps)
    val_mean = val_loss.mean(dim=0)
    plt.plot(range(train_mean.size(0)), train_mean, label='Train Mean')
    plt.fill_between(
        range(train_mean.size(0)),
        train_mean - torch.std(train_loss, dim=0),
        train_mean + torch.std(train_loss, dim=0),
        alpha=0.3, label='Training Variance')
    
    plt.plot(torch.arange(0, val_mean.size(0)) * val_int, val_mean, label='Validation_mean')
    
    plt.fill_between(
        torch.arange(0, val_mean.size(0)) * val_int, 
        val_mean - torch.std(val_loss, dim=0),
        val_mean + torch.std(val_loss, dim=0),
        alpha=0.3, label='Validation Deviation')
    
    plt.xlabel('Batch idx')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training and Validation Loss w.r.t. Batch Index.')
    plt.scatter([], [], color="w", alpha=0, label=stringify_hyparams(args))
    plt.legend(loc = 'upper right')
    if save: plt.savefig(path+name)
    plt.show()

def perplexity_graph(train_loss: Tensor, val_loss: Tensor, val_int: int, path: str = None, 
                     save: bool = False, name: str = None, args: argparse.Namespace = None,
                     baseline_perplexity_mean: float = 7.91, baseline_perplexity_std: float = 0.67, 
                     baseline_name: str = 'Trigram'):
    """
    Plot cross-validation train/validation perplexities w.r.t batch index.
    Plots variance areas over folds. Path has to finish by '/' !
    """
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    val_perplex = 2**val_loss     # (k-fold x n_steps)
    train_perplex = 2**train_loss
    val_perplex_mean = val_perplex.mean(dim=0)
    train_perplex_mean = train_perplex.mean(dim=0)

    plt.plot(range(train_perplex_mean.size(0)), train_perplex_mean, label='Train Mean')
    plt.fill_between(
        range(train_perplex_mean.size(0)),
        train_perplex_mean - torch.std(train_perplex, dim=0),
        train_perplex_mean + torch.std(train_perplex, dim=0),
        alpha=0.3, label='Training Variance')
    
    plt.plot(torch.arange(0, val_perplex_mean.size(0)) * val_int, val_perplex_mean, label='Validation Mean')
    plt.fill_between(
        torch.arange(0, val_perplex_mean.size(0)) * val_int, 
        val_perplex_mean - torch.std(val_perplex, dim=0),
        val_perplex_mean + torch.std(val_perplex, dim=0),
        alpha=0.3, label='Validation Deviation')
    
    if baseline_perplexity_mean != None:
        bperx_mean = torch.ones(size=(train_perplex_mean.size(0),)) * baseline_perplexity_mean
        plt.plot(torch.arange(0, train_perplex_mean.size(0)), bperx_mean, label=f'{baseline_name} Validation Mean')
        plt.fill_between(
            torch.arange(0, train_perplex_mean.size(0)), 
            bperx_mean - baseline_perplexity_std,
            bperx_mean + baseline_perplexity_std,
            alpha=0.3, label=f'{baseline_name} Validation Deviation'
        )

    plt.xlabel('Batch idx')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity w.r.t. Batch Index.')
    plt.legend()
    plt.scatter([], [], color="w", alpha=0, label=stringify_hyparams(args))
    if save: plt.savefig(path+name)
    plt.show()

def stringify_hyparams(namespace) -> str:
    hyperparams = {
        'batch_size': 'B', 'n_tokens': 'N', 'n_layers': 'L', 'n_heads': 'h', 'd_model': 'd', 'lr' : 'lr', 
        'use_lr_decay': 'decay', 'decay_iters' : 'decay_iters', 'dataset_path' : 'data', 'max_iter' :' max_iter', 
        'betas' : 'betas', 'n_warm_iters' : 'warmup', 'min_lr' : 'min_lr', 'name' : 'name', 'k_fold' : 'k_fold'
    }
    
    res = "("
    for param in namespace:
        if param in hyperparams: 
            if param == 'dataset_path':
                file = namespace[param].split('/')[-1]
                res += f"{hyperparams[param]}={file}, "
            else:
                res += f"{hyperparams[param]}={namespace[param]}, "
    res = res.rstrip(', ')
    res = res.split(', ')
    res.insert(6, '\n')
    res.insert(9, '\n')
    res.insert(14, '\n')
    res = ', '.join(res)
    res += ")"
    return res

def load_model_metrics(path: str, V: int, device='cpu') -> tuple[GPT, ...]: 
    '''V is number of tokens in dataset vocabulary.'''
    checkpoint = torch.load(path, map_location=torch.device(device))
    params = checkpoint['params']
    model = GPT(
        B=params['batch_size'], L=params['n_layers'], 
        d=params['d_model'], d_ff=3*params['d_model'],
        N=params['n_tokens'], h=params['n_heads'], V=V)
    model.load_state_dict(checkpoint['model'])
    return model, params, checkpoint['train_loss'], checkpoint['valid_loss']