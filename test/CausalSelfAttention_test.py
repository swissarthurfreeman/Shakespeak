import torch
import unittest
import torch.nn as nn
from modules.CausalSelfAttention import CausalSelfAttention


class TestCausalAttention(unittest.TestCase):
    def test_CausalAttention(self):
        print("[TEST CASE] Running test_CausalAttention...")
        # This is based on TALN's exercice on Self Attention.
        # we just consider the first head and first element in batch. 
        B, N, d = 5, 3, 4
        h, d_k, d_v = 1, 3, 3
        
        csa = CausalSelfAttention(B, N, d, h, d_k, d_v)
        # see https://stackoverflow.com/questions/76949152/why-does-pytorchs-linear-layer-store-the-weight-in-shape-out-in-and-transpos
        # stores as (out, in and does left hand side multiplication X Q)
        print(list(csa.to_Q.parameters()))
        csa.to_Q.weight = nn.Parameter( # (d x d_k) = (4 x 3)
            torch.transpose(
                torch.tensor([
                    [2, 2, 1],
                    [3, 0, 2],
                    [1, 4, 2],
                    [0, 1, 2]
                ], dtype=torch.float64), 
            dim0=0, dim1=1)
        )
        #csa.to_Q.bias = nn.Parameter(torch.zeros(size=(d, 1)))

        csa.to_K.weight = nn.Parameter( # (d x d_k) = (4 x 3)
            torch.tensor([
                [1, 3, 2],
                [2, 1, 1],
                [2, 0, 1],
                [1, 1, 3]
            ], dtype=torch.float64).repeat(1, h)
        )

        csa.to_V.weight = nn.Parameter( # (d x d_v) = (4 x 3)
            torch.tensor([
                [3, 1, 0],
                [0, 2, 1],
                [2, 1, 2],
                [2, 3, 0]
            ], dtype=torch.float64).repeat(1, h)
        )

        csa.W_O.weight = nn.Parameter(  # ( h*d_v x d ) = (d_v x d) = (3 x 4)
            torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]
            ], dtype=torch.float64).repeat(1, h)
        )
        
        #print(list(csa.named_parameters()))
        
        # (N x d) = (3 x 4)
        X = torch.tensor([
            [1, 2, 1, 2], 
            [0, 1, 3, 0], 
            [2, 2, 1, 2]
        ], dtype=torch.float64)
        X = X.expand(size=(B, N, d))
        
        csa_X = csa(X)
        print(csa_X)

        print("[TEST CASE] test_CausalAttention Successful...")

if __name__ == '__main__':
    unittest.main()