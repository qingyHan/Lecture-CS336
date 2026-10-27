"""第一讲单元测试：测试函数名仍使用 unittest 的英文命名规范。"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tokenization_demo import ByteTokenizer, CharacterTokenizer, ToyBPE, merge_pair


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            "",
            "Hello, world!",
            "你好，世界",
            "🌍🙂",
            "e\u0301",  # e 与组合重音符：多个码点可构成一个视觉字符
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

    def test_overlapping_pair_merge(self):
        """相邻 pair 的统计可以重叠，但替换必须从左往右且不重叠。"""
        self.assertEqual(merge_pair([256, 256, 256], (256, 256), 257), [257, 256])

    def test_negative_merge_budget_rejected(self):
        with self.assertRaises(ValueError):
            ToyBPE.train(["abc"], num_merges=-1)


if __name__ == "__main__":
    unittest.main()
