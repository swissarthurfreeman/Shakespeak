from torch.utils.data.dataloader import DataLoader

from model import ShakespearModel
from Tokenizer import CharDataset


def load_data(filename) -> str:
    with open(filename, 'r') as file:
        data = file.read()
    return data


if __name__ == '__main__':
    N_TOKENS = 128  # N
    N_LAYERS = 12
    N_HEADS = 8
    N_EMBEDDINGS = 768
    N_WORKERS = 2
    BATCH_SIZE = N_TOKENS  # B
    D_MODEL = 512
    D_FF = 2048
    RAW_DATA_PATH = './datasets/shakespear_corpus.txt'

    raw_data = load_data(RAW_DATA_PATH)

    tokenized_data = CharDataset(N_TOKENS, raw_data)
    data_loader = DataLoader(
        tokenized_data,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
    )

    # get first batch of sentences
    tokenized_sentence, _ = next(iter(data_loader))  # (128,128) = (B,N)

    # default value given by teacher assist. We should play with it when it's working
    model = ShakespearModel(N_LAYERS, N_HEADS, N_EMBEDDINGS,
                            D_MODEL, tokenized_data.get_vocab_size())

    model(tokenized_sentence)
