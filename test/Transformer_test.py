import torch
import unittest
import torch.nn as nn
from torch import Tensor
from modules.Transformer import Transformer


class TestTransformer(unittest.TestCase):
    def test_Transformer(self):
        print("[TEST CASE] Running test_Transformer...")
        
        B, N, d = 20, 128, 768
        h, d_k, d_v = 8, 64, 32
        V, d_ff, L = 1000, 2048, 12
        
        t = Transformer(L, B, N, h, d, d_k, d_v, d_ff, V)
        X = torch.rand(size=(B, N, d)) * 5456   # test stability

        Y: Tensor = t(X)
        # test size is correct
        self.assertListEqual(list(Y.size()), [B, N, V])

        # test last dimension is indeed a probability distribution
        self.assertAlmostEqual(Y[5, 60, :].sum().item(), 1, 6)
        self.assertAlmostEqual(Y[7, 30, :].sum().item(), 1, 6)
        self.assertAlmostEqual(Y[10, 47, :].sum().item(), 1, 6)
        self.assertAlmostEqual(Y[12, 104, :].sum().item(), 1, 6)
        
        
        print("[TEST CASE] test_Transformer Successful...")

if __name__ == '__main__':
    unittest.main()