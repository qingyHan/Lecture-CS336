"""Stanford CS336 第一讲：文本分词算法的教学实现。

这个示例只依赖 Python 标准库，适合在 CPU 上学习与验证。
ToyBPE 用于理解合并算法，不是完整的生产级 tokenizer。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


class CharacterTokenizer:
    """将 Unicode 码点编码为整数并解码；码点不一定对应一个视觉字符。"""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class ByteTokenizer:
    """在 UTF-8 字节与 0～255 的整数之间进行双向转换。"""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


def merge_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    """从左到右替换互不重叠的相邻 pair。

例如 [256, 256, 256] 会合并为 [257, 256]：
位置 1-2 与 2-3 的候选 pair 重叠，因此一次不能同时合并。
"""
    result: list[int] = []
    i = 0
    while i < len(sequence):
        if (
            i + 1 < len(sequence)
            and sequence[i] == pair[0]
            and sequence[i + 1] == pair[1]
        ):
            result.append(new_id)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return result


@dataclass
class ToyBPE:
    """教学版字节级 BPE：按顺序保存词表和训练得到的合并规则。"""

    vocab: dict[int, bytes]
    merges: list[tuple[tuple[int, int], int]]

    @classmethod
    def train(cls, texts: Iterable[str], num_merges: int) -> "ToyBPE":
        if num_merges < 0:
            raise ValueError("合并次数 num_merges 不能为负数")

        sequences = [list(text.encode("utf-8")) for text in texts]
        vocab = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[tuple[int, int], int]] = []

        for _ in range(num_merges):
            # 每条训练文本单独计数，避免错误地跨文本统计相邻 pair。
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))

            if not pair_counts:
                break

            # 先选出现次数最多的 pair；频率相同时选字典序更大的 pair。
            # 这种规则便于复现，不代表所有生产 tokenizer 都如此规定。
            pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            # 新 token 的原始字节串由左右两个旧 token 拼接得到。
            new_id = len(vocab)
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            merges.append((pair, new_id))
            sequences = [
                merge_pair(sequence, pair, new_id)
                for sequence in sequences
            ]

        return cls(vocab=vocab, merges=merges)

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8"))
        # 编码时按照训练所得的合并顺序操作，不能随意交换顺序。
        for pair, new_id in self.merges:
            sequence = merge_pair(sequence, pair, new_id)
        return sequence

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(self.vocab[token_id] for token_id in ids)
        return byte_string.decode("utf-8")


def run_demo() -> None:
    sample = "Hello, 🌍! 你好!"
    char_tok = CharacterTokenizer()
    byte_tok = ByteTokenizer()
    bpe = ToyBPE.train(["ababab"], num_merges=2)

    assert char_tok.decode(char_tok.encode(sample)) == sample
    assert byte_tok.decode(byte_tok.encode(sample)) == sample
    assert bpe.encode("ababab") == [257, 256]
    assert bpe.decode(bpe.encode("ababab")) == "ababab"

    print(f"示例文本：{sample!r}")
    print(f"Unicode 码点数量：{len(sample)}")
    print(f"UTF-8 字节数量：{len(sample.encode('utf-8'))}")
    print(f"字节级 token 数量：{len(byte_tok.encode(sample))}")
    print("教学版 BPE 合并规则：")
    for (left, right), new_id in bpe.merges:
        print(f"  ({left}, {right}) -> {new_id}: {bpe.vocab[new_id]!r}")
    print(f"BPE 编码结果（ababab）：{bpe.encode('ababab')}")
    print(f"BPE 解码结果：{bpe.decode(bpe.encode('ababab'))!r}")
    print("编码-解码往返检查：通过")


if __name__ == "__main__":
    run_demo()
