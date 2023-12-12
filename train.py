import math
import torch
import argparse
from model import GPT
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from utils import getLoaderDataset, generate

def calculate_learning_rate(iteration: int):
    if iteration < n_warmup_iterations:
        return learning_rate * iteration / n_warmup_iterations
    if iteration > learning_rate_decay_iterations:
        return min_learning_rate
    decay_ratio = (iteration - n_warmup_iterations) / \
        (learning_rate_decay_iterations - n_warmup_iterations)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coefficient * (learning_rate - min_learning_rate)

from tqdm import tqdm

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    data_loader, tokenized_data = getLoaderDataset(
        args.n_tokens, args.batch_size, args.dataset)
    losses = []
    model = GPT(args.batch_size, args.n_layers, args.d_model, 3*args.d_model, args.n_tokens, args.n_heads,
                tokenized_data.get_vocab_size()).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=betas, eps=eps)

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        
        lr = calculate_learning_rate(
            batch_idx) if args.use_lr_decay else learning_rate
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()


        logits: Tensor = model(inputs)
        loss = criterion(
            logits.flatten(0, -2),
            targets.view(-1).long().to(device)
        )
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        print(f"batch {batch_idx}, Loss : {loss.item()}")

        if batch_idx % 100 == 0 and batch_idx != 0:
            torch.save(model.state_dict(), f"./runs/model_{batch_idx}.pt")

    return model, losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", help=f"Batch size (default: {B}).", type=int, default=B,)
    parser.add_argument("--n_tokens", "-n", help=f"Number of tokens (default: {N}).", type=int, default=N,)
    parser.add_argument("--n_layers", "-l", help=f"Number of layers (default: {L}).",type=int, default=L,)
    parser.add_argument("--n_heads", help=f"Number of heads (default: {h}).", type=int, default=h,)
    parser.add_argument("--d_model", "-d", help=f"Dimension of model (default: {d}).", type=int, default=d,)
    parser.add_argument("--learning_rate", "-lr", help=f"Learning Rate (default: {learning_rate}).", type=float, default=learning_rate,)
    parser.add_argument("--use_lr_decay", help=f"Use learning rate decay strategy (default: {use_lr_decay}).", type=bool, default=use_lr_decay,)
    parser.add_argument("--dataset", help=f"Dataset file to use for training (default: {dataset}).", type=str, default=dataset,)
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
    use_lr_decay = True
    dataset = './datasets/shakespear_corpus.txt'

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
    model, losses = train_model(parse_args())

    # plt.plot(range(len(losses)), losses)
    loader, tokenized_data = getLoaderDataset(N, B, "./datasets/shakespear_corpus.txt")
    print(tokenized_data.decode(generate(model, tokenized_data.encode("Oh"), 200)))
