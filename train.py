import math
import os
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
        """Matrix (k-fold, n_step) of ce_loss at every grad update of all folds on train.
        Will be size (n_step,) if cross-validation is False."""
        self.val_loss: Tensor = Tensor()
        """Matrix (k-fold, n_step) of ce_loss at every grad update of all folds on validation.
        Will be size (n_step,) if cross-validation is False."""
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

    def cross_validation(self) -> tuple[list[GPT], Tensor, Tensor]:
        """
        Returns of a tuple of lists of size k_fold. out1[k] is the k-th "fold" 
        trained GPT model, out2 is the matrix of cross entropy values on the train
        data (k-fold x n°steps) and out3 is the validation loss matrix (k-fold x n°steps % val_int).
        """
        models = []
        kfolds_train_losses = []
        kfolds_val_losses = []
        for i in range(self.args.k_fold):
            print("---------------------------------\nFold n°%s" % i)
            model, train_loss, val_loss = self._train_model(fold=i)
            models.append(model)
            kfolds_train_losses.append(train_loss)
            kfolds_val_losses.append(val_loss)
        
        self.train_loss = torch.stack(kfolds_train_losses)
        """Matrix (k-fold x n°steps) of ce_loss at every grad update of all folds on train"""
        self.val_loss = torch.stack(kfolds_val_losses)
        """Matrix (k-fold x n°steps) of ce_loss at every grad update of all folds on validation."""
        if not os.path.isdir(f"./runs/{self.args.name}"):   # if no checkpoints were saved
            os.makedirs(f"./runs/{self.args.name}")

        torch.save(
            {'k_fold_train_loss': self.train_loss, 'k_fold_valid_loss': self.val_loss, 'params': vars(self.args)}, 
            f"./runs/{self.args.name}/total_cross_val_metrics.pt"
        )
        return models, self.train_loss, self.val_loss

    def train(self, fold=None, k_fold=None):
        if self.args.cross_val:
            return self.cross_validation()
        else:
            return self._train_model(fold)

    def _train_model(self, fold=None) -> tuple[GPT, Tensor, Tensor]:
        """
        Train a GPT model using self.args and return the model, the vector of train
        losses at every step and the vector of validation losses at every val_int.
        Leave fold as None if we're not doing cross validation. 
        
        Parameters
        ----------
        - fold : int | None
        If left to None, the data loaders will simply split the dataset in 90% train and
        10% validation in the usual way. If true, the data loader will select validation
        data from the specified fold, and train data from everywhere else. 
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        training_data_loader, self.tokenized_data = getLoaderDataset(
            N=self.args.n_tokens, B=self.args.batch_size, dataset_path=self.args.dataset_path, 
            fold=fold, k_fold=self.args.k_fold, is_training=True, shuffle=True)
        
        validation_data_loader, _ = getLoaderDataset(
            N=self.args.n_tokens, B=self.args.batch_size, dataset_path=self.args.dataset_path, 
            fold=fold, k_fold=self.args.k_fold, is_training=False, shuffle=True)

        train_loss = []
        valid_loss = []
        model = GPT(self.args.batch_size, self.args.n_layers, self.args.d_model, 3*self.args.d_model, self.args.n_tokens, self.args.n_heads,
                    self.tokenized_data.get_vocab_size()).to(device)
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        curr_iter = 0   # count from 1
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
                    valid_loss.append(validation_loss)
                    print(f'Epoch: {epoch}, Batch index {curr_iter}, Training Loss: {"{:.4f}".format(loss.item())}, Validation Loss: {"{:.4f}".format(validation_loss)}')
                    
                if self.args.save and curr_iter != 0 and curr_iter % self.args.save_int == 0:
                    ckpt = {        # saving all this allows rebuilding plots etc as needed.
                        'model': model.state_dict(), 
                        'valid_loss': Tensor(valid_loss),
                        'train_loss': Tensor(train_loss),
                        'params': vars(self.args)
                    }

                    if not os.path.isdir(f"./runs/{self.args.name}"):
                        os.makedirs(f"./runs/{self.args.name}")
                    
                    if self.args.cross_val: # if this is a cross validation run, make sure to include fold number. 
                        torch.save(ckpt, f"./runs/{self.args.name}/cross_val_{fold}_{self.args.name}_{epoch}_{curr_iter}.pt")
                    else:
                        torch.save(ckpt, f"./runs/{self.args.name}/{self.args.name}_{epoch}_{curr_iter}.pt")

                if curr_iter > self.args.max_iter:
                    return model, Tensor(train_loss), Tensor(valid_loss)

                train_loss.append(loss.item())
                curr_iter += 1

if __name__ == '__main__':
    args = Args.parse_args()
    print(args)
    train = Training(args)
    models, train_loss, val_loss = train.train()
    if args.cross_val:
        for i, model in enumerate(models):
            print("Fold n°%s model generation, with prompt Oh God Oh God ! \n ------------------------------------" % i)
            print(train.tokenized_data.decode(
                generate(model, train.tokenized_data.encode("Oh God Oh God !"), 50)
            ))
            print("------------------------------------")
            
    
