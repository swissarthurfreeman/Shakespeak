import torch


class WPE(torch.nn.Module):
    def __init__(self, d):
        super(WPE, self).__init__()
        self.d = d

    def forward(self, x):
        pos = x.unsqueeze(2)
        i = torch.arange(self.d).to(x.device)
        angles = pos * (1 / torch.pow(10000, (2 * i) / self.d))

        sin = torch.sin(angles[:, :, 0::2])  # Sinus of even elements
        cos = torch.cos(angles[:, :, 1::2])  # Cosinus of odd elements

        embedding = torch.cat([sin, cos], dim=-1)
        return embedding
