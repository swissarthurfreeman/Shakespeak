import torch


class WPE(torch.nn.Module):
    def __init__(self, d):
        super(WPE, self).__init__()
        self.d = d

    def forward(self, X):
        """X is B x N tensor where every line is a range 0...N"""
        self.B, self.N = X.size()
        pos = X.unsqueeze(-1)                                    # add dimension at end, B x N x 1
        i = torch.arange(self.d).to(X.device)                    # i = 0...d, angles is  B x N x d
        angles = pos * (1 / torch.pow(10000, (2 * i) / self.d))  # broadcast over last dim, 
        
        pair_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(self.d)]).expand(size=(self.B, self.N, self.d))
        
        sin = torch.sin(angles) * pair_mask  # Sinus of even elements, B x N x d
        cos = torch.cos(angles) * torch.logical_not(pair_mask)         # Cosinus of odd elements B x N x d


        print(sin[0][5][:7])
        print(cos[0][5][:7])
        embedding = sin + cos   # BUG this is no longer interleaved B x N x d.
        return embedding
