import unittest
import torch
from Shakespeak.modules import WPE


class TestPositionalEncoding(unittest.TestCase):
    def test_PositionalEncoding(self):
        print("[TEST CASE] Running test_PositionalEncoding...")
        B, N, d = 5, 128, 300
        encoder = WPE(d)
        X = torch.arange(N).expand(size=(B, N))
        pos_embeddings = encoder(X)
        # embedding of position 2 for first element in batch should be same as 
        # the embedding of pos 2 for the second element in batch
        self.assertEqual(pos_embeddings[0][2][10], pos_embeddings[1][2][10])
        self.assertEqual(pos_embeddings[1][5][4], pos_embeddings[4][5][4])

        POS, i = torch.tensor(0), torch.tensor(10)  # sin pair, cos unpair
        self.assertEqual(
            pos_embeddings[0][POS][i], 
            torch.sin(POS * (1 / torch.pow(10000, (2*i)/d )) ) 
        )

        # position POS = 8, dimension i = 10, d = 30
        POS, i = torch.tensor(8), torch.tensor(10)
        self.assertEqual(
            pos_embeddings[0][POS][i], 
            torch.sin(POS * (1 / torch.pow(10000, (2*i)/d )) ) 
        )
        print("[TEST CASE] test_PositionalEncoding Successful...")

if __name__ == '__main__':
    unittest.main()