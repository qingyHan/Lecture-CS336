"""Teaching implementations for Stanford CS336 Lecture 1: tokenization.

This is a deliberately small, CPU-only example. The ToyBPE implementation is
for understanding the merge algorithm; it is not a production tokenizer.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


class CharacterTokenizer:
    """Map Unicode code points to integer IDs and back."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class ByteTokenizer:
    """Map UTF-8 bytes to integers in [0, 255] and back."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


def merge_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    """Replace non-overlapping occurrences of pair from left to right."""
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
    """A tiny byte-level BPE model with a fixed sequence of learned merges."""

    vocab: dict[int, bytes]
    merges: list[tuple[tuple[int, int], int]]

    @classmethod
    def train(cls, texts: Iterable[str], num_merges: int) -> "ToyBPE":
        if num_merges < 0:
            raise ValueError("num_merges must be non-negative")

        sequences = [list(text.encode("utf-8")) for text in texts]
        vocab = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[tuple[int, int], int]] = []

        for _ in range(num_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))

            if not pair_counts:
                break

            # Deterministic tie-break: highest count, then lexicographically
            # highest pair. Production tokenizers may define another policy.
            pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
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

    print(f"sample: {sample!r}")
    print(f"Unicode code points: {len(sample)}")
    print(f"UTF-8 bytes: {len(sample.encode('utf-8'))}")
    print(f"byte token count: {len(byte_tok.encode(sample))}")
    print("toy BPE merges:")
    for (left, right), new_id in bpe.merges:
        print(f"  ({left}, {right}) -> {new_id}: {bpe.vocab[new_id]!r}")
    print(f"toy BPE encode('ababab'): {bpe.encode('ababab')}")
    print(f"toy BPE decode: {bpe.decode(bpe.encode('ababab'))!r}")
    print("round-trip checks: PASS")


if __name__ == "__main__":
    run_demo()
