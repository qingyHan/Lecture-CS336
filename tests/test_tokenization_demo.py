import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tokenization_demo import ByteTokenizer, CharacterTokenizer, ToyBPE


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            "",
            "Hello, world!",
            "你好，世界",
            "🌍🙂",
            "e\u0301",  # e + combining acute accent
            "abababab",
            "def f(x): return x + 1",
        ]

    def test_character_round_trip(self):
        tokenizer = CharacterTokenizer()
        for text in self.samples:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_byte_round_trip(self):
        tokenizer = ByteTokenizer()
        for text in self.samples:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_toy_bpe_round_trip(self):
        tokenizer = ToyBPE.train(["ababab", "abab", "abac"], num_merges=4)
        for text in self.samples + ["ababab", "abac"]:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_expected_two_merges(self):
        tokenizer = ToyBPE.train(["ababab"], num_merges=2)
        self.assertEqual(tokenizer.vocab[256], b"ab")
        self.assertEqual(tokenizer.vocab[257], b"abab")
        self.assertEqual(tokenizer.encode("ababab"), [257, 256])
        self.assertEqual(tokenizer.decode([257, 256]), "ababab")

    def test_negative_merge_budget_rejected(self):
        with self.assertRaises(ValueError):
            ToyBPE.train(["abc"], num_merges=-1)


if __name__ == "__main__":
    unittest.main()
