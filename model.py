import torch
import torch.nn as nn
from modules import WPE
from torch import Tensor
from modules import Block
import torch.optim as optim
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader

from utils import CharDataSet

class GPT(nn.Module):
    def __init__(self, B, L, d, d_ff, N, h, V):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.to(self.device)
    
    def forward(self, X):
        X = X.long().to(self.device)
        tok_emb = self.WTE(X)
        pos_emb = self.WPE(X)

        X = self.Dropout(pos_emb + tok_emb)

        for block in self.blocks:
            X = block(X)

        X = self.Final_LayerNorm(X)
        X = self.LM_Head(X)
        return X

def train_model(model: Module, loader: DataLoader, tokenizer: CharDataSet) -> tuple[nn.Module, list[float]]: 
    losses: list[float] = []
    model.train()

    criterion = nn.CrossEntropyLoss(reduction='mean').to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=10e-9)
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        
        optimizer.zero_grad()

        logits: Tensor = model(inputs)
        loss = criterion(
            logits.flatten(0, -2),
            targets.view(-1).long().to(model.device)
        )
        losses.append(loss.item())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        
        optimizer.step()

        print(f"batch {batch_idx}, Loss : {loss.item()}")

        if batch_idx % 100 == 0 and batch_idx != 0:
            torch.save(model.state_dict(), f"./runs/model_{batch_idx}.pt")

    return model, losses