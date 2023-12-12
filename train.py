import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

from model import GPT
from utils import generate, getLoaderDataset


def calculate_learning_rate(iteration: int):
    if iteration < n_warmup_iterations:
        return learning_rate * iteration / n_warmup_iterations
    if iteration > learning_rate_decay_iterations:
        return min_learning_rate
    decay_ratio = (iteration - n_warmup_iterations) / \
        (learning_rate_decay_iterations - n_warmup_iterations)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coefficient * (learning_rate - min_learning_rate)


@torch.no_grad()
def evaluate_model(model, validation_data, criterion):
    model.eval()
    losses = torch.zeros(n_validation_batch).to(model.device)
    # As shuffle of the DataLoader is true, the batches are randomized at each validation
    for idx, (inputs, targets) in enumerate(validation_data):
        if idx > n_validation_batch - 1:
            break
        logits = model(inputs)
        loss = criterion(
            logits.flatten(0, -2),
            targets.view(-1).long().to(model.device)
        )
        losses[idx] = loss
    model.train()
    return losses.mean()


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_data_loader, tokenized_data = getLoaderDataset(
        args.n_tokens, args.batch_size, args.dataset,
        is_training=True, shuffle=True)
    validation_data_loader, _ = getLoaderDataset(
        args.n_tokens, args.batch_size, args.dataset, is_training=False, shuffle=True)

    losses = {
        'train': [],
        'validation': [],
        'epochs': []
    }
    model = GPT(args.batch_size, args.n_layers, args.d_model, 3*args.d_model, args.n_tokens, args.n_heads,
                tokenized_data.get_vocab_size()).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=betas, eps=eps)
    current_iteration = 0
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(training_data_loader):
            lr = calculate_learning_rate(
                current_iteration) if args.use_lr_decay else learning_rate

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

            if current_iteration % validation_interval == 0:
                validation_loss = evaluate_model(
                    model, validation_data_loader, criterion)
                losses['validation'].append(validation_loss.item())
                print(
                    f'Epoch: {epoch}, Batch {batch_idx}, Training Loss: {loss.item()}, Validation Loss: {validation_loss.item()}')

            if ((current_iteration + 1) % 1000 == 0 and current_iteration != 0) or current_iteration > args.max_iterations:
                torch.save(model.state_dict(),
                           f"./runs/model_{current_iteration+1}.pt")
                if current_iteration > args.max_iterations:
                    return model, losses
            current_iteration += 1
        losses['epochs'].append(epoch_loss/len(training_data_loader))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", "-b", help=f"Batch size (default: {B}).", type=int, default=B,)
    parser.add_argument(
        "--n_tokens", "-n", help=f"Number of tokens (default: {N}).", type=int, default=N,)
    parser.add_argument(
        "--n_layers", "-l", help=f"Number of layers (default: {L}).", type=int, default=L,)
    parser.add_argument(
        "--n_heads", help=f"Number of heads (default: {h}).", type=int, default=h,)
    parser.add_argument(
        "--d_model", "-d", help=f"Dimension of model (default: {d}).", type=int, default=d,)
    parser.add_argument("--learning_rate", "-lr",
                        help=f"Learning Rate (default: {learning_rate}).", type=float, default=learning_rate,)
    parser.add_argument(
        "--use_lr_decay", help=f"Use learning rate decay strategy (default: {use_lr_decay}).", type=bool, default=use_lr_decay,)
    parser.add_argument(
        "--dataset", help=f"Dataset file to use for training (default: {dataset}).", type=str, default=dataset,)
    parser.add_argument(
        "--max_iterations", help=f"Maximum Number of iterations for training (default: {max_iterations}).", type=int, default=max_iterations,)
    parser.add_argument(
        "--out", help=f"Directory containing the saved models (default: {out_dir}).", type=str, default=out_dir,)

    return parser.parse_args()


if __name__ == '__main__':
    # nanoGPT --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
    B = 12
    N = 64  # context of up to 256 previous characters
    L = 4
    h = 4
    d = 128
    learning_rate = 1e-3
    betas = (0.9, 0.99)
    eps = 10e-9
    n_warmup_iterations = 100
    learning_rate_decay_iterations = 5000
    min_learning_rate = 1e-4
    use_lr_decay = False
    dataset = './datasets/shakespear_corpus.txt'
    out_dir = './runs/'
    # number of batch to use to compute average loss on validation set
    n_validation_batch = 100
    # Validation loss will be computed every {validation_interval} batches.
    validation_interval = 10

    # Training will stop as soon as we reach {max_iterations} or the model saw {n_epochs} times the full dataset. (depends which one we reach first)
    max_iterations = 100
    n_epochs = 10

    args = parse_args()
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
    model, losses = train_model(parse_args())
    print(losses)
    # plt.plot(range(len(losses)), losses)
    _, tokenized_data = getLoaderDataset(
        N, B, "./datasets/shakespear_corpus.txt")
    print(tokenized_data.decode(generate(model, tokenized_data.encode("Oh"), 200)))
