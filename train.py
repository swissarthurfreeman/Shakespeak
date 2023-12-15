import os
import math
import torch
import argparse
import torch.nn as nn
from model import GPT
from utils import Args
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import getLoaderDataset


class Training:
    """Helper class to run a series of trainings.
    Defines a cross_validation() method, that'll run
    a series of trainings with specified arguments 
    with k-fold partitions of the data."""
    def __init__(self, args: Args):
        self.args = args

    def calculate_lr(self, iteration: int):
        if iteration < self.n_warm_iters:
            return self.args.lr * iteration / self.args.n_warm_iters
        if iteration > self.args.lr_decay_iter:
            return self.min_lr
        decay_ratio = (iteration - self.args.n_warm_iters) / \
            (self.args.lr_decay_iter - self.args.n_warm_iters)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coefficient * (self.args.lr - self.args.min_lr)

    @torch.no_grad()
    def evaluate_model(self, model, validation_data, criterion):
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

    def cross_validation(self, k_fold=10) -> tuple[list[GPT], list[dict[str, list]]]:
        """Returns of a tuple of lists of size k_fold. out1[i] is the i-th fold 
        trained GPT model, out2[i] is the metrics dictionary keyed by 'train' 
        and 'validation', out2[i]['train'|'validation'] is a list of cross entropy
        losses for every batch or every batch validation interval."""
        models = []
        metrics = []
        for i in range(k_fold):
            print("---------------------------------\nFold n°%s" % i)
            model, fold_metrics = self.train_model(fold=i)
            models.append(model)
            metrics.append(fold_metrics)
        return models, metrics

    def train_model(self, fold=1, k_fold=10) -> tuple[GPT, dict[str, list]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        training_data_loader, tokenized_data = getLoaderDataset(
            self.args.n_tokens, self.args.batch_size, self.args.dataset_path, 
            fold, k_fold, is_training=True, shuffle=True)
        
        validation_data_loader, _ = getLoaderDataset(
            self.args.n_tokens, self.args.batch_size, self.args.dataset_path, 
            fold, k_fold, is_training=False, shuffle=True)

        losses = {'train': [],'validation': []}
        model = GPT(self.args.batch_size, self.args.n_layers, self.args.d_model, 3*self.args.d_model, self.args.n_tokens, self.args.n_heads,
                    tokenized_data.get_vocab_size()).to(device)
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=self.betas, eps=self.eps)
        
        curr_iter = 1   # count from 1
        for epoch in range(self.n_epochs):
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

                if curr_iter % self.val_int == 0:   # every val_int, compute val loss.
                    validation_loss = self.evaluate_model(model, validation_data_loader, criterion).item()
                    losses['validation'].append(validation_loss)
                    print(f'Epoch: {epoch}, Batch {batch_idx}, Training Loss: {loss.item()}, Validation Loss: {validation_loss}')
                    
                    torch.save(model.state_dict(), f"./runs/model_{curr_iter+1}.pt")
                    
                if curr_iter > self.args.max_iter:
                    return model, losses

                losses['train'].append(loss.item())
                curr_iter += 1

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--batch_size", "-b", help=f"Batch size (default: {self.args.batch_size}).", type=int, default=self.args.batch_size,)
        parser.add_argument(
            "--n_tokens", "-n", help=f"Number of tokens (default: {self.args.n_tokens}).", type=int, default=self.args.n_tokens,)
        parser.add_argument(
            "--n_layers", "-l", help=f"Number of layers (default: {self.args.n_layers}).", type=int, default=self.args.layers,)
        parser.add_argument(
            "--n_heads", help=f"Number of heads (default: {self.args.n_heads}).", type=int, default=self.args.n_heads,)
        parser.add_argument(
            "--d_model", "-d", help=f"Dimension of model (default: {self.args.d_model}).", type=int, default=self.args.d_model,)
        parser.add_argument("--lr", "-lr",
                            help=f"Learning Rate (default: {self.args.lr}).", type=float, default=self.args.lr,)
        parser.add_argument(
            "--use_lr_decay", help=f"Use learning rate decay strategy (default: {self.args.use_lr_decay}).", type=bool, default=self.args.use_lr_decay,)
        parser.add_argument(
            "--dataset", help=f"Dataset file to use for training (default: {self.args.dataset_path}).", type=str, default=self.args.dataset_path,)
        parser.add_argument(
            "--max_iter", help=f"Maximum Number of iterations for training (default: {self.args.max_iter}).", type=int, default=self.args.max_iter,)
        parser.add_argument(
            "--out", help=f"Directory containing the saved models (default: {self.args.out_dir}).", type=str, default=self.args.out_dir,)

        return parser.parse_args()

    def save_losses_graph(self, path, losses):
        plt.clf()
        plt.plot(range(len(losses['train'])), losses['train'], label='Training_mean')
        plt.fill_between(range(len(losses['train'])), losses['train'] - losses['train_var'], losses['train'] + losses['train_var'], alpha=0.3, label='Variance Area for train')
        plt.plot(range(0, len(losses['train']), self.val_int), losses['validation'], label='Validation_mean')
        plt.fill_between(range(0, len(losses['train']), self.val_int), losses['validation'] - losses['validation_var'], losses['validation'] + losses['validation_var'], alpha=0.3, label='Variance Area for validation')
        plt.xlabel('Number of batches')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over number of batches')
        plt.legend()
        plt.savefig(path)
        plt.show()

    def save_perplexity_graph(self, path, perplexities):
        plt.clf()
        plt.plot(range(len(perplexities['train'])), perplexities['train'], label='Training_mean')
        plt.fill_between(range(len(perplexities['train'])), perplexities['train'] - perplexities['train_var'], perplexities['train'] + perplexities['train_var'], alpha=0.3, label='Variance Area for train')
        
        plt.plot(range(0, len(perplexities['train']), self.val_int), perplexities['validation'], label='Validation_mean')
        plt.fill_between(range(0, len(perplexities['train']), self.val_int), perplexities['validation'] - perplexities['validation_var'], perplexities['validation'] + perplexities['validation_var'], alpha=0.3, label='Variance Area for validation')
        plt.xlabel('Number of batches')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity Over number of batches')
        plt.legend()
        plt.savefig(path)
        plt.show()

