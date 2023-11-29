from typing import Callable

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from Tokenizer import CharDataset


class Trainer:
    def __init__(self, epochs: int, batch_size: int, device: str, optimizer: Callable, optimizer_args: dict) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args


def forward_and_loss(model, criterion, batch_x, batch_y):
    result = model(batch_x)
    loss = criterion(result, batch_y)
    return result, loss


class LLMTrainer(Trainer):
    def train(self, dataset, model, n_tokens: int):
        tokenized_data = CharDataset(n_tokens, dataset)
        data_loader = DataLoader(
            tokenized_data,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=2,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(model.parameters(), **self.optimizer_args)

        for epoch in range(self.epochs):
            model.train()

            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_out, batch_loss = forward_and_loss(
                    model, criterion, batch_y)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
