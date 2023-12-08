import torch
import unittest
import torch.nn as nn
from modules.CausalSelfAttention import CausalSelfAttention


class TestCausalAttention(unittest.TestCase):
    def test_CausalAttention(self):
        print("[TEST CASE] Running test_CausalAttention...")
        # This is based on TALN's exercice on Self Attention.
        # we just consider the first head and first element in batch. 
        B, N, d = 5, 3, 4       # note, with this test d, d_k, d_v MUST REMAIN FIXED
        h, d_k, d_v = 2, 3, 3
        
        csa = CausalSelfAttention(B, N, d, h, d_k, d_v)
        # see https://stackoverflow.com/questions/76949152/why-does-pytorchs-linear-layer-store-the-weight-in-shape-out-in-and-transpos
        # stores as (out, in and does left hand side multiplication X Q)
        old_size = csa.to_Q.weight.size()
        
        self.assertListEqual(list(csa.to_Q.weight.size()), [d_k*h, d])
        
        csa.to_Q.weight = nn.Parameter( # (d x d_k) = (4 x 3)
            torch.transpose(
                torch.tensor([
                    [2, 2, 1],
                    [3, 0, 2],
                    [1, 4, 2],
                    [0, 1, 2]
                ], dtype=torch.float32).repeat(1, h), 
            dim0=0, dim1=1)
        )

        self.assertListEqual(list(old_size), list(csa.to_Q.weight.size()))

        csa.to_Q.bias = nn.Parameter(
            torch.zeros(csa.to_Q.bias.size())
        )
        
        #csa.to_Q.bias = nn.Parameter(torch.zeros(size=(d, 1)))
        old_size = csa.to_K.weight.size()
        self.assertListEqual(list(csa.to_K.weight.size()), [d_k*h, d])
        
        print("to_K", old_size)
        csa.to_K.weight = nn.Parameter( # (d x d_k) = (4 x 3)
            torch.transpose(
                torch.tensor([
                    [1, 3, 2],
                    [2, 1, 1],
                    [2, 0, 1],
                    [1, 1, 3]
                ], dtype=torch.float32).repeat(1, h),
            dim0=0, dim1=1)
        )

        self.assertListEqual(list(old_size), list(csa.to_K.weight.size()))

        csa.to_K.bias = nn.Parameter(
            torch.zeros(csa.to_K.bias.size())
        )

        old_size = csa.to_V.weight.size()
        self.assertListEqual(list(csa.to_V.weight.size()), [d_v*h, d])
        
        csa.to_V.weight = nn.Parameter( # (d x d_v) = (4 x 3)
            torch.transpose(
                torch.tensor([ # [1, 2, 1, 2]
                    [3, 1, 0],
                    [0, 2, 1],
                    [2, 1, 2],
                    [2, 3, 0]
                ], dtype=torch.float32).repeat(1, h),
                dim0=0, dim1=1
            )
        )
        self.assertListEqual(list(old_size), list(csa.to_V.weight.size()))

        csa.to_V.bias = nn.Parameter(
            torch.zeros(csa.to_V.bias.size())
        )

        old_size = csa.W_O.weight.size()
        
        self.assertListEqual(list(csa.W_O.weight.size()), [d, d_v*h])
        
        # W_O.weight is d x h * d_v when (head_1 | ... | head_h) W_O, 
        # F.linear(x, W) is x @ W.T  
        csa.W_O.weight = nn.Parameter(  # ( h*d_v = 6 x d = 4 ) = (2 * d_v x d) = (6 x 4)
            torch.tensor([
                [1, 0, 0],  # d x d_v, 4 x 3 -> d x h*d_v, 4 x 6
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]
            ], dtype=torch.float32).repeat(1, h)
        )
        
        self.assertListEqual(list(old_size), list(csa.W_O.weight.size()))
        

        csa.W_O.bias = nn.Parameter(
            torch.zeros(csa.W_O.bias.size())
        )
        
        # (B x N x d) = (2 x 3 x 4)
        X = torch.tensor([
            [1, 2, 1, 2], 
            [0, 1, 3, 0], 
            [2, 2, 1, 2]
        ], dtype=torch.float32)
        X = X.expand(size=(B, N, d))
        
        QKT = csa.ScoreMatrix(X)
        
        self.assertAlmostEqual(QKT[0, 0, 0, 0].item(), 148.96, 1)
        self.assertAlmostEqual(QKT[1, 0, 0, 1].item(), 71.59, 1)    # TALN exercise is wrong here
        self.assertAlmostEqual(QKT[0, 0, 0, 2].item(), 180.71, 1)
        self.assertAlmostEqual(QKT[0, 0, 1, 0].item(), 130.48, 1)
        self.assertAlmostEqual(QKT[0, 0, 1, 1].item(), 53.12, 1)
        self.assertAlmostEqual(QKT[1, 0, 1, 2].item(), 163.97, 1)
        self.assertAlmostEqual(QKT[0, 0, 2, 0].item(), 173.78, 1)
        self.assertAlmostEqual(QKT[0, 0, 2, 1].item(), 84.29, 1)
        self.assertAlmostEqual(QKT[0, 0, 2, 2].item(), 211.31, 1)

        A = csa.AttentionMatrix(X)

        # rows sum to 1
        self.assertEqual(torch.sum(A[0, 0, 1, :]), 1)
        self.assertEqual(torch.sum(A[0, 0, 2, :]), 1)

        # upper triangle is 0
        self.assertEqual(torch.sum(A[0, 0, 0, 1]), 0)
        self.assertEqual(torch.sum(A[0, 0, 0, 2]), 0)   
        self.assertEqual(torch.sum(A[0, 0, 1, 2]), 0)

        V = csa.ValuesMatrix(X)
        self.assertListEqual(V[0][0].flatten().tolist(), [9, 12, 4, 6, 5, 7, 12, 13, 4])
        self.assertListEqual(V[1][1].flatten().tolist(), [9, 12, 4, 6, 5, 7, 12, 13, 4])
        
        SA_out = csa(X)

        self.assertListEqual(SA_out[0][0].tolist(), (torch.tensor([9, 12, 4, 0])*h).tolist())
        self.assertListEqual(SA_out[0][1].tolist(), (torch.tensor([9, 12, 4, 0])*h).tolist())
        self.assertListEqual(SA_out[0][2].tolist(), (torch.tensor([12, 13, 4, 0])*h).tolist())

        print("[TEST CASE] test_CausalAttention Successful...")

if __name__ == '__main__':
    unittest.main()