
import torch


class ShakespearModel(torch.nn.Module):
    def __init__(self, n_layers, n_heads, n_embeddings):
        super(ShakespearModel, self).__init__()
        self.blocks = []

    def forward(self, idx, target=None):
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

    def generate(self, idx, n_new_tokens: int):

        # loop up to the number of desired new tokens.
        for _ in range(n_new_tokens):
            # Get logits
            logits = self(idx)

            # TODO - Get probabilities from logits
            probabilities = None
            # Take element with highest probability
            # TODO - Implement other way to choose the element
            _, new_char_idx = torch.topk(probabilities, k=1, dim=-1)
            # Update text with new element
            idx = torch.cat((idx, new_char_idx), dim=1)
        return idx
