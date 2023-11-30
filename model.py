
import torch


class ShakespearModel(torch.nn.Module):
    def __init__(self, n_layers, n_heads, n_embeddings):
        super(ShakespearModel, self).__init__()
        self.blocks = []

    def forward(self, idx):
        print(idx)
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device).unsqueeze(
            0)  # shape (1, t)

        tok_emb = WTE(idx)  # token embeddings
        pos_emb = WPE(pos)  # position embeddings
        x = torch.nn.Dropout(tok_emb + pos_emb)
        for Block in self.Blocks:
            x = Block(x)
        x = Final_LayerNorm(x)
        logits = LM_Head(x)
        return logits
