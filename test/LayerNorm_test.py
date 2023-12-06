import torch
import unittest
from modules.LayerNorm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_LayerNorm(self):
        print("[TEST CASE] Running test_LayerNorm...")
        LY = LayerNorm()
        B, N, d = 5, 128, 300
        X = torch.rand(size=(B, N, d))*10 - 5
        
        X_norm = LY(X)
        self.assertAlmostEqual(torch.mean(X_norm[0, 0, :]).item(), 0, 7)
        self.assertAlmostEqual(torch.std(X_norm[0, 0, :]).item(), 1, 7)
        
        self.assertAlmostEqual(torch.mean(X_norm[4, 54, :]).item(), 0, 7)
        self.assertAlmostEqual(torch.std(X_norm[4, 64, :]).item(), 1, 7)
        
        print("[TEST CASE] test_LayerNorm Successful...")

if __name__ == '__main__':
    unittest.main()