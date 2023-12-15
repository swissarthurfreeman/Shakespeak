import os
import math
import torch
import argparse
import torch.nn as nn
from model import GPT
from utils import Args, generate
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import getLoaderDataset, DataLoader


class Training:
    """Helper class to train or cross-validate models.
    Defines a cross_validation() method, that'll run
    a series of trainings with specified arguments 
    with k-fold partitions of the data."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.train_loss: Tensor = Tensor()
        """Vector of ce_loss at every grad update of all folds on train."""
        self.val_loss: Tensor = Tensor()
        """Vector of ce_loss at every grad update of all folds on validation."""
        self.tokenized_data = None

    def calculate_lr(self, iteration: int) -> float:
        if iteration < self.args.n_warm_iters:
            return self.args.lr * iteration / self.args.n_warm_iters
        if iteration > self.args.lr_decay_iter:
            return self.args.min_lr
        decay_ratio = (iteration - self.args.n_warm_iters) / \
            (self.args.lr_decay_iter - self.args.n_warm_iters)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coefficient * (self.args.lr - self.args.min_lr)

    @torch.no_grad()
    def evaluate_model(self, model: GPT, validation_data: DataLoader, criterion) -> Tensor:
        model.eval()
        losses = torch.zeros(self.args.n_validation_batch).to(model.device)
        for idx, (inputs, targets) in enumerate(validation_data):
            if idx > self.args.n_validation_batch - 1: break
            logits = model(inputs)
            loss = criterion(
                logits.flatten(0, -2),
                targets.view(-1).long().to(model.device)
            )
            losses[idx] = loss
        model.train()
        return losses.mean()

    def cross_validation(self, k_fold: int = 10) -> tuple[list[GPT], Tensor, Tensor]:
        """
        Returns of a tuple of lists of size k_fold. out1[k] is the k-th "fold" 
        trained GPT model, out2 is the matrix of cross entropy values on the train
        data (k-fold x n°steps) and out3 is the validation loss matrix (k-fold x n°steps % val_int).
        """
        models = []
        train_loss = []
        val_loss = []
        for i in range(k_fold):
            print("---------------------------------\nFold n°%s" % i)
            model, fold_metrics = self.train_model(fold=i)
            models.append(model)
            train_loss.append(fold_metrics['train'])
            val_loss.append(fold_metrics['validation'])
        
        self.train_loss = Tensor(train_loss)
        """Matrix (k-fold x n°steps) of ce_loss at every grad update of all folds on train"""
        self.val_loss = Tensor(val_loss)
        """Matrix (k-fold x n°steps) of ce_loss at every grad update of all folds on validation."""
        return models, self.train_loss, self.val_loss

    def train_model(self, fold=None, k_fold=None) -> tuple[GPT, dict[str, list]]:
        """Leave fold, k_fold as None if we're not doing cross validation. In which
        case DataLoader will split data into 90% train and 10% validation."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        training_data_loader, self.tokenized_data = getLoaderDataset(
            N=self.args.n_tokens, B=self.args.batch_size, dataset_path=self.args.dataset_path, 
            fold=fold, k_fold=k_fold, is_training=True, shuffle=True)
        
        validation_data_loader, _ = getLoaderDataset(
            N=self.args.n_tokens, B=self.args.batch_size, dataset_path=self.args.dataset_path, 
            fold=fold, k_fold=k_fold, is_training=False, shuffle=True)

        losses = {'train': [],'validation': []}
        model = GPT(self.args.batch_size, self.args.n_layers, self.args.d_model, 3*self.args.d_model, self.args.n_tokens, self.args.n_heads,
                    self.tokenized_data.get_vocab_size()).to(device)
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        curr_iter = 1   # count from 1
        for epoch in range(self.args.n_epochs):
            for batch_idx, (inputs, targets) in enumerate(training_data_loader):
                lr = self.calculate_lr(
                    curr_iter) if self.args.use_lr_decay else self.args.lr

                for param_group in optimizer.param_groups: param_group['lr'] = lr

                optimizer.zero_grad()

                logits: Tensor = model(inputs)
                loss = criterion(
                    logits.flatten(0, -2),
                    targets.view(-1).long().to(device)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                optimizer.step()

                if curr_iter % self.args.val_int == 0:   # every val_int, compute val loss.
                    validation_loss = self.evaluate_model(model, validation_data_loader, criterion).item()
                    losses['validation'].append(validation_loss)
                    print(f'Epoch: {epoch}, Batch index {curr_iter}, Training Loss: {"{:.4f}".format(loss.item())}, Validation Loss: {"{:.4f}".format(validation_loss)}')
                    
                if self.args.save and curr_iter % self.args.save_int == 0:
                    ckpt = {'model': model.state_dict(), 'params': vars(self.args)}
                    if not os.path.isdir(f"./runs/{self.args.name}"):
                        os.makedirs(f"./runs/{self.args.name}")
                    
                    torch.save(ckpt, f"./runs/{self.args.name}/{self.args.name}_{epoch}_{curr_iter}.pt")
                    
                if curr_iter > self.args.max_iter:
                    return model, losses

                losses['train'].append(loss.item())
                curr_iter += 1

    def losses_graph(self, path: str = None, save: bool = False):
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        train_mean = self.train_loss.mean(dim=0)    # (k-fold, n_steps) -> (1 x n_steps)
        val_mean = self.val_loss.mean(dim=0)
        plt.plot(range(train_mean.size(0)), train_mean, label='Train Mean')
        plt.fill_between(
            range(train_mean.size(0)),
            train_mean - torch.std(self.train_loss, dim=0),
            train_mean + torch.std(self.train_loss, dim=0),
            alpha=0.3, label='Training Variance')
        
        plt.plot(torch.arange(1, val_mean.size(0) + 1) * self.args.val_int, val_mean, label='Validation_mean')
        plt.fill_between(
            torch.arange(1, val_mean.size(0) + 1) * self.args.val_int, 
            val_mean - torch.std(self.val_loss, dim=0),
            val_mean + torch.std(self.val_loss, dim=0),
            alpha=0.3, label='Validation Deviation')
        
        plt.xlabel('Batch idx')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Training and Validation Loss w.r.t. Batch Index.')
        plt.legend()
        if save: plt.savefig(path)
        plt.show()

    def perplexity_graph(self, path: str = None, save: bool = False):
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        val_perplex = 2**self.val_loss     # (k-fold x n_steps)
        train_perplex = 2**self.train_loss
        val_perplex_mean = val_perplex.mean(dim=0)
        train_perplex_mean = train_perplex.mean(dim=0)

        plt.plot(range(train_perplex_mean.size(0)), train_perplex_mean, label='Train Mean')
        plt.fill_between(
            range(train_perplex_mean.size(0)),
            train_perplex_mean - torch.std(train_perplex, dim=0),
            train_perplex_mean + torch.std(train_perplex, dim=0),
            alpha=0.3, label='Training Variance')
        
        plt.plot(torch.arange(1, val_perplex_mean.size(0) + 1) * self.args.val_int, val_perplex_mean, label='Validation Mean')
        plt.fill_between(
            torch.arange(1, val_perplex_mean.size(0) + 1) * self.args.val_int, 
            val_perplex_mean - torch.std(val_perplex, dim=0),
            val_perplex_mean + torch.std(val_perplex, dim=0),
            alpha=0.3, label='Validation Deviation')
        
        plt.xlabel('Batch idx')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity w.r.t. Batch Index.')
        plt.legend()
        if save: plt.savefig(path)
        plt.show()

if __name__ == '__main__':
    args = Args.parse_args()
    print(args)
    train = Training(args)
    model, metrics = train.train_model()
    print(train.tokenized_data.decode(
        generate(model, train.tokenized_data.encode("Oh God Oh God !"), 50)
    ))
    