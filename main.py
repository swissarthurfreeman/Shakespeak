from torch.utils.data.dataloader import DataLoader

from model import ShakespearModel
from Tokenizer import CharDataset


def load_data(filename) -> str:
    with open(filename, 'r') as file:
        data = file.read()
    return data


if __name__ == '__main__':
    N_TOKENS = 128  # N
    BATCH_SIZE = N_TOKENS  # B

    raw_data = load_data('./datasets/shakespear_corpus.txt')

    tokenized_data = CharDataset(N_TOKENS, raw_data)
    data_loader = DataLoader(
        tokenized_data,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    # get first batch of sentences
    tokenized_sentence, _ = next(iter(data_loader))  # (128,128) = (B,N)

    # default value given by teacher assist. We should play with it when it's working
    model = ShakespearModel(12, 8, 768)

    model(tokenized_sentence)
