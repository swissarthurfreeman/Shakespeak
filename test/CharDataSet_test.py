import unittest
from Shakespeak.utils import CharDataSet


class TestCharDataSet(unittest.TestCase):
    def test_encoding(self):
        print("[TEST CASE] Running test_encoding...")
        cd = CharDataSet(N_tokens=3, is_training=True, raw_data="The quick brown fox rabbit runs over the fox.", p_train=1)
        # vocabulary is [' ', '.', 'T', 'a', 'b', 'c', 'e', 'f', 'h', 'i', 'k', 'n', 'o', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x']
        #                 0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   19   20   21
        chunk, shifted = cd[0]
        self.assertEqual(chunk.tolist(), [2., 8., 6.])
        self.assertEqual(shifted.tolist(), [8., 6., 0.])

        chunk, shifted = cd[5]  # uic, ick
        self.assertEqual(chunk.tolist(), [17., 9., 5.])
        self.assertEqual(shifted.tolist(), [9., 5., 10.])

        print("[TEST CASE] test_encoding Successful...")

    
    def test_chunking(self):
        print("[TEST CASE] Running test_chunking...")
        cd = CharDataSet(N_tokens=5, is_training=True, raw_data="The quick brown fox rabbit runs over the fox.", p_train=0.8)
        # 20 % of characters are left to test set, e.g. CharDataSet only yields elements of "The quick brown fox rabbit runs over". 
        # vocabulary is [' ', '.', 'T', 'a', 'b', 'c', 'e', 'f', 'h', 'i', 'k', 'n', 'o', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x']
        #                 0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   19   20   21
        chunk, _ = cd[31]
        self.assertEqual(chunk.tolist(), cd.encode(" over").flatten().tolist())

        print("[TEST CASE] test_chunking Successful...")
    
    def test_decoding(self):
        print("[TEST CASE] Running test_decoding...")
        cd = CharDataSet(N_tokens=5, is_training=True, raw_data="The quick brown fox rabbit runs over the fox.", p_train=1)
        
        # vocabulary is [' ', '.', 'T', 'a', 'b', 'c', 'e', 'f', 'h', 'i', 'k', 'n', 'o', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x']
        #                 0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   19   20   21
        chunk, shifted = cd[5]
        dec_chunk = cd.decode(chunk)
        dec_shift = cd.decode(shifted)
        self.assertEqual(dec_chunk, "uick ")
        self.assertEqual(dec_shift, "ick b")

        print("[TEST CASE] test_decoding Successful...")
    
if __name__ == '__main__':
    unittest.main()