import unittest
import torch
from modules.LanguageHead import LanguageHead


class TestLanguageHead(unittest.TestCase):
    def test_TestLanguageHead(self):
        print("[TEST CASE] Running test_TestLanguageHead...")
        
        B, N, d = 20, 128, 768
        h, d_k, d_v = 8, 64, 32
        V, d_ff, L = 1000, 2048, 12
        
        LH = LanguageHead(d, V)
        X = torch.rand(size=(B, N, d))
        Y = LH(X)

        self.assertListEqual(list(Y.size()), [B, N, V])

        print("[TEST CASE] test_TestLanguageHead Successful...")

if __name__ == '__main__':
    unittest.main()