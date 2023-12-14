import os
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from model import GPT
from tqdm import tqdm
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import generate, getLoaderDataset


class Training:
    def __init__(self, args):
        self.args = args
        self.n_warmup_iterations = 100
        self.learning_rate_decay_iterations = 5000
        self.min_learning_rate = 1e-4
        self.n_validation_batch = 200
        self.betas = (0.9, 0.99)
        self.eps = 10e-9
        self.n_epochs = 10
        self.validation_interval = 100

    def calculate_learning_rate(self, iteration: int):
        if iteration < self.n_warmup_iterations:
            return self.args.learning_rate * iteration / self.n_warmup_iterations
        if iteration > self.learning_rate_decay_iterations:
            return self.min_learning_rate
        decay_ratio = (iteration - self.n_warmup_iterations) / \
            (self.learning_rate_decay_iterations - self.n_warmup_iterations)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_learning_rate + coefficient * (self.args.learning_rate - self.min_learning_rate)


    @torch.no_grad()
    def evaluate_model(self, model, validation_data, criterion):
        model.eval()
        losses = torch.zeros(self.n_validation_batch).to(model.device)
        # As shuffle of the DataLoader is true, the batches are randomized at each validation
        for idx, (inputs, targets) in enumerate(validation_data):
            if idx > self.n_validation_batch - 1:
                break
            logits = model(inputs)
            loss = criterion(
                logits.flatten(0, -2),
                targets.view(-1).long().to(model.device)
            )
            losses[idx] = loss
        model.train()
        return losses.mean()

    # define a cross validation function
    def crossvalid(self, k_fold=10) -> dict[str, list[GPT | dict[str, list]]]:
        """Returns dictionary of keyed by model, losses and perplexities.
        Values are a list where element i corresponds to a particular fold
        and the values in said lists are either models if looking at the 
        models key, or are dictionaries keyed by losses, validation or perplexity
        and yield lists of the latter."""
        results = {
            'models': [],
            'losses': [],
            'perplexities': []
        }
        for i in range(k_fold):
            model, losses, perplexities = self.train_model(fold=i)
            results['models'].append(model)
            results['losses'].append(losses)
            results['perplexities'].append(perplexities)
        return results


    def train_model(self, fold=1, k_fold=10) -> tuple[GPT, dict[str, list], dict[str, list]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        training_data_loader, tokenized_data = getLoaderDataset(
            self.args.n_tokens, self.args.batch_size, self.args.dataset, fold, k_fold, is_training=True, shuffle=True)
        validation_data_loader, _ = getLoaderDataset(
            self.args.n_tokens, self.args.batch_size, self.args.dataset, fold, k_fold, is_training=False, shuffle=True)

        losses = {'train': [],'validation': [],'epochs': []}
        model = GPT(self.args.batch_size, self.args.n_layers, self.args.d_model, 3*self.args.d_model, self.args.n_tokens, self.args.n_heads,
                    tokenized_data.get_vocab_size()).to(device)
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate, betas=self.betas, eps=self.eps)
        
        current_iteration = 0
        for epoch in range(self.n_epochs):      # TODO : Remove epoch notion. We never exhaust training iterator with the size of our dataset. 
            epoch_loss = 0
            for batch_idx, (inputs, targets) in enumerate(training_data_loader):
                lr = self.calculate_learning_rate(
                    current_iteration) if self.args.use_lr_decay else self.args.learning_rate

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.zero_grad()

                logits: Tensor = model(inputs)
                loss = criterion(
                    logits.flatten(0, -2),
                    targets.view(-1).long().to(device)
                )
                losses['train'].append(loss.item())
                epoch_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

                if current_iteration % self.validation_interval == 0:
                    validation_loss = self.evaluate_model(model, validation_data_loader, criterion)
                    
                    losses['validation'].append(validation_loss.item())
                    print(f'Epoch: {epoch}, Batch {batch_idx}, Training Loss: {loss.item()}, Validation Loss: {validation_loss.item()}')

                if ((current_iteration + 1) % self.validation_interval == 0 and current_iteration != 0) or current_iteration > self.args.max_iterations:
                    torch.save(model.state_dict(), f"./runs/model_{current_iteration+1}.pt")
                    
                    if current_iteration > self.args.max_iterations:
                        return model, losses, self.calculate_perplexity(losses)
                current_iteration += 1
            losses['epochs'].append(epoch_loss/len(training_data_loader))


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
        parser.add_argument("--learning_rate", "-lr",
                            help=f"Learning Rate (default: {self.args.learning_rate}).", type=float, default=self.args.learning_rate,)
        parser.add_argument(
            "--use_lr_decay", help=f"Use learning rate decay strategy (default: {self.args.use_lr_decay}).", type=bool, default=self.args.use_lr_decay,)
        parser.add_argument(
            "--dataset", help=f"Dataset file to use for training (default: {self.args.dataset}).", type=str, default=self.args.dataset,)
        parser.add_argument(
            "--max_iterations", help=f"Maximum Number of iterations for training (default: {self.args.max_iterations}).", type=int, default=self.args.max_iterations,)
        parser.add_argument(
            "--out", help=f"Directory containing the saved models (default: {self.args.out_dir}).", type=str, default=self.args.out_dir,)

        return parser.parse_args()


    def calculate_perplexity(self, losses): # TODO : move this to train, directly append in train loop. 
        perplexities = {}
        perplexities['validation'] = [2 ** loss for loss in losses['validation']]
        perplexities['train'] = [2 ** loss for loss in losses['train']]
        perplexities['epochs'] = [2 ** loss for loss in losses['epochs']]
        return perplexities


    def save_losses_graph(self, path, losses):
        plt.clf()

        plt.plot(range(len(losses['train'])), losses['train'], label='Training_mean')
        plt.fill_between(range(len(losses['train'])), losses['train'] - losses['train_var'], losses['train'] + losses['train_var'], alpha=0.3, label='Variance Area for train')
        
        plt.plot(range(0, len(losses['train']), self.validation_interval), losses['validation'], label='Validation_mean')
        plt.fill_between(range(0, len(losses['train']), self.validation_interval), losses['validation'] - losses['validation_var'], losses['validation'] + losses['validation_var'], alpha=0.3, label='Variance Area for validation')
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
        
        plt.plot(range(0, len(perplexities['train']), self.validation_interval), perplexities['validation'], label='Validation_mean')
        plt.fill_between(range(0, len(perplexities['train']), self.validation_interval), perplexities['validation'] - perplexities['validation_var'], perplexities['validation'] + perplexities['validation_var'], alpha=0.3, label='Variance Area for validation')
        plt.xlabel('Number of batches')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity Over number of batches')
        plt.legend()
        plt.savefig(path)
        plt.show()


'''
if __name__ == '__main__':
    # nanoGPT --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
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
    python train.py --use_lr_decay=True
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('./results/', exist_ok=True)
    model, losses, perplexities = train_model(parse_args())
    save_losses_graph('./results/losses.png', losses)
    save_perplexity_graph('./results/perplexity.png', perplexities)
    _, tokenized_data = getLoaderDataset(
        N, B, "./datasets/shakespear_corpus.txt")
    print(tokenized_data.decode(generate(model, tokenized_data.encode("Oh"), 200)))
'''