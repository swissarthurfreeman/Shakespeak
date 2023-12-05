import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from model import ShakespearModel
from parsing.CharDataSet import CharDataSet


def load_data(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data


if __name__ == '__main__':
    N_EPOCHS = 2
    N_TOKENS = 128  # N
    N_LAYERS = 12  # L
    N_HEADS = 8  # h
    N_WORKERS = 2
    BATCH_SIZE = N_TOKENS  # B
    D_MODEL = 768  # d
    D_K = 64
    D_V = D_K
    D_FF = 2048
    RAW_DATA_PATH = './datasets/shakespear_corpus.txt'

    raw_data = load_data(RAW_DATA_PATH)
    tokenized_data = CharDataSet(N_TOKENS, raw_data)
    
    data_loader = DataLoader(
        tokenized_data,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
    )


    model = ShakespearModel(
        N_LAYERS, N_HEADS, D_MODEL, 
        D_FF, D_K, D_V, BATCH_SIZE, 
        N_TOKENS, tokenized_data.get_vocab_size()
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        # batch_idx is index of the batch, inputs/targets are B x N 
        # tensors where inputs[b] is the sequence of word indexes of 
        # sequence n°b in the batch, targets[b] is the sequence of 
        # 1 to the right shifted words indexes of the sequence.  
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            optimizer.zero_grad()

            logits = model(inputs)                  # logits is B x N x V vector
            loss = criterion(
                logits.view(-1, logits.size(-1)),   # B*N x V vector
                targets.long().view(-1)             # flattens targets to B*N vector
            )
            
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            
            print(f"batch {batch_idx} - Loss: {loss}")

        average_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {average_loss:.4f}')
