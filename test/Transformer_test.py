import torch
import unittest
import torch.nn as nn
from modules.Transformer import Transformer


class TestTransformer(unittest.TestCase):
    def test_Transformer(self):
        print("[TEST CASE] Running test_Transformer...")
        
        B, N, d = 20, 128, 768       # note, with this test d, d_k, d_v MUST REMAIN FIXED
        h, d_k, d_v = 8, 64, 32
        V, d_ff, L = 1000, 2048, 12
        
        E = nn.Embedding(V, d)
        t = Transformer(L, B, N, h, d, d_k, d_v, d_ff, V, E)
        X = torch.rand(size=(B, N, d))

        Y = t(X)
        
        print("[TEST CASE] test_Transformer Successful...")

if __name__ == '__main__':
    unittest.main()