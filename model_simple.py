import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
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
        
        text_chunk: Tensor = self.raw_data[idx:idx+self.N_tokens]
        shifted_text_chunk: Tensor = self.raw_data[idx+1:idx+1+self.N_tokens]
        # TODO : preprocess encode batch before and return readout from tensor already on gpu
        chunk_idx = self.encode(text_chunk).squeeze()
        shifted_idx = self.encode(shifted_text_chunk).squeeze()
        return chunk_idx, shifted_idx   # (N_token,), (N_token,) tuple.

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


def load_data(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data


def getLoaderDataset(N, B, path):
    raw = load_data(path)
    tokenized_data = CharDataSet(N, raw)
    
    data_loader = DataLoader(
        tokenized_data,
        shuffle=True,
        pin_memory=True,
        batch_size=B,
        num_workers=8,
    )

    return data_loader, tokenized_data

class WPE(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        pos = x.unsqueeze(2)
        i = torch.arange(self.d).to(x.device)
        angles = pos * (1 / torch.pow(10000, (2 * i) / self.d))

        torch.sin_(angles[:, :, 0::2])  # Sinus of even elements
        torch.cos_(angles[:, :, 1::2])  # Cosinus of odd elements
        return angles

class QKVAttention(nn.Module):
    def __init__(
        self, dim_in, dim_qk, dim_v, nb_heads=1, causal=True, attention_dropout=0.0
    ):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.attention_dropout = attention_dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q

        q = torch.einsum("ntc,hdc->nhtd", x_q, self.w_q)
        k = torch.einsum("ntc,hdc->nhtd", x_kv, self.w_k)
        v = torch.einsum("ntc,hdc->nhtd", x_kv, self.w_v)

        a = torch.einsum("nhtd,nhsd->nhts", q, k) / math.sqrt(q.size(3))

        if self.causal:
            forbidden_attention = (
                torch.arange(a.size(2), device=q.device)[None, None, :, None]
                < torch.arange(a.size(3), device=q.device)[None, None, None, :]
            )
            a = a.masked_fill(forbidden_attention, float("-inf"))

        a = a.softmax(dim=3)
        y = torch.einsum("nhts,nhsd->nthd", a, v).flatten(2)

        y = y @ self.w_o

        return y


class Block(nn.Module):
    def __init__(self, h, d, d_ff):
        super().__init__()
        
        self.CausalSelfAttn = QKVAttention(dim_in=d, dim_qk=d//h, dim_v=d//h, nb_heads=h, causal=True, attention_dropout=0)
        self.W1 = nn.Linear(d, d_ff)
        self.W2 = nn.Linear(d_ff, d)

        self.LayerNorm1 = nn.LayerNorm(d)
        self.LayerNorm2 = nn.LayerNorm(d)

    def forward(self, X):
        X = X + self.CausalSelfAttn(self.LayerNorm1(X))
        out = X + self.W2(F.relu(self.W1(self.LayerNorm2(X))))
        return out

class GPT(nn.Module):
    def __init__(self, B, L, d, d_ff, N, h, V):
        super().__init__()
        self.d = d
        self.d_ff = d_ff
        self.N = N
        self.h = h
        self.B = B

        self.WTE = nn.Embedding(V, d)
        self.WPE = WPE(d)
        self.blocks = nn.ModuleList()

        for _ in range(L):
            self.blocks.append(Block(h, d, d_ff))
        
        self.Dropout = nn.Dropout(0.2)
        self.Final_LayerNorm = nn.LayerNorm(d)
        self.LM_Head = nn.Linear(d, V)
    
    def forward(self, X):
        X = X.long().to('cuda')
        tok_emb = self.WTE(X)
        pos_emb = self.WPE(X)

        X = self.Dropout(pos_emb + tok_emb)

        for block in self.blocks:
            X = block(X)

        X = self.Final_LayerNorm(X)
        X = self.LM_Head(X)
        return X



@torch.no_grad()
def generate(model, idx, max_new_tokens):
    input = idx.to('cuda')
    for k in range(max_new_tokens):
        logits = model(input) 
        
        #char_id = torch.multinomial(logits[0, -1, :], num_samples=1).item()
        
        char_id = torch.distributions.categorical.Categorical(logits=logits[0, -1, :]).sample()

        new_c = torch.tensor(char_id).reshape(shape=(1, 1))
        input = torch.concat( (input, new_c), dim=-1 )
    return input.flatten()


def train_model(L, B, N, d, h): 
    data_loader, tokenized_data = getLoaderDataset(N, B, "./shakespear_corpus.txt")
    losses = []
    model = GPT(B, L, d, 3 * d, N, h, tokenized_data.get_vocab_size()).to('cuda')
    model.train()

    criterion = nn.CrossEntropyLoss(reduction='mean').to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=10e-9)
    
    #inputs, targets = next(iter(data_loader))
    for batch_idx, (inputs, targets) in enumerate(data_loader):
    #for batch_idx in range(1000):
        optimizer.zero_grad()

        if batch_idx == 500:
            break

        logits: Tensor = model(inputs)
        loss = criterion(
            logits.flatten(0, -2),
            targets.view(-1).long().to('cuda')
        )
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        print(f"batch {batch_idx}, Loss : {loss.item()}")

        if batch_idx % 100 == 0:
          torch.save(model.state_dict(), f"model_{batch_idx}.pt")

    return model, losses

import matplotlib.pyplot as plt
    
if __name__ == '__main__':
    B = 64
    N = 256 # context of up to 256 previous characters
    L = 6
    h = 6
    d = 384

    # Directorz unique pour chaque run
    """
    # Saving
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 10,
    }
    torch.save(ckpt, ckpt_path)

    # Loading
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    # opt
    """
    model, losses = train_model(L, B, N, d, h)

    plt.plot(range(len(losses)), losses)
    loader, tokenized_data  = getLoaderDataset(N, B, "./shakespear_corpus.txt")
    print(tokenized_data.decode(generate(model, tokenized_data.encode("Oh"), 200)))