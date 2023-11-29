import argparse

import torch

from SLLM import SmallLargeLanguageModel
from trainer import LLMTrainer


def load_data(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data


def main(args):
    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.Adam
    raw_data = load_data(args.dataset)
    n_tokens = args.n_tokens
    model = SmallLargeLanguageModel()
    trainer = LLMTrainer(args.epochs, args.batch_size,
                         device, optimizer, optimizer_args={})

    trainer.train(raw_data, model, n_tokens)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        "-b",
        help=f"Batch size (default: 128).",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n_tokens",
        "-t",
        help=f"Number of tokens (default: 128).",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='./datasets/shakespear_corpus.txt',
        help="Path to dataset file",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        help=f"Epochs (default: 10).",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--cpu", "-c", help="Force using cpu?", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
