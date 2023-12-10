import torch
from torch import Tensor
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

def train_model(N_EPOCHS, N_TOKENS, N_LAYERS, N_HEADS, BATCH_SIZE, D_MODEL, D_K, D_V, D_FF): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N_WORKERS = 2
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
    ).to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.98), eps=10e-9)

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        # batch_idx is index of the batch, inputs/targets are B x N 
        # tensors where inputs[b] is the sequence of word indexes of 
        # sequence n°b in the batch, targets[b] is the sequence of 
        # 1 to the right shifted words indexes of the sequence.  
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            optimizer.zero_grad()
            
            logits: Tensor = model(inputs)                  # logits is B x N x V vector
            logits = logits.reshape(shape=(BATCH_SIZE*N_TOKENS, tokenized_data.get_vocab_size())) # B*N x V vector
            loss = criterion(
                logits,   
                targets.long().view(-1).to(device)             # flattens targets (B x N) to B*N vector
            )
            
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            
            print(f"batch {batch_idx} - Loss: {loss}")

        average_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {average_loss:.4f}')

    return model

if __name__ == '__main__':

    N_EPOCHS = 4
    N_TOKENS = 64  # N
    N_LAYERS = 6  # L
    N_HEADS = 4  # h
    N_WORKERS = 2
    BATCH_SIZE = 30  # B
    D_MODEL = 200  # d
    D_K = 32
    D_V = D_K
    D_FF = 50
    RAW_DATA_PATH = './datasets/shakespear_corpus.txt'

    trained_mod = train_model(N_EPOCHS, N_TOKENS, N_LAYERS, N_HEADS, BATCH_SIZE, D_MODEL, D_K, D_V, D_FF)


    raw_data = load_data(RAW_DATA_PATH)

    tokenized_data = CharDataSet(N_TOKENS, raw_data)
    data_loader = DataLoader(
        tokenized_data,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
    )

    seed = "O God, O God!"
    idx = tokenized_data.encode(seed)
    new_tokens = trained_mod.generate(idx, n_new_tokens=50)
    print(new_tokens)
    print(tokenized_data.decode(new_tokens))