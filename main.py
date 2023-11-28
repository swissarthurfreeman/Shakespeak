from torch.utils.data.dataloader import DataLoader

from Tokenizer import CharDataset


def load_data(filename):
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

    # get first sentence in the batch
    print(tokenized_sentence[0][0])

    # decode the sentence
    print(tokenized_data.decode(tokenized_sentence[0][0]))

    # try with a prompt smaller than 128 tokens
    prompt = 'bonjour!    \n  le monde!'
    tokenized_prompt = tokenized_data.encode(prompt)
    print(tokenized_prompt)
    print(tokenized_data.decode(tokenized_prompt[0]))
