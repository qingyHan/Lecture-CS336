import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_03"))

from tiny_transformer import (  # noqa: E402
    CausalSelfAttention,
    TinyGPT,
    TransformerConfig,
    count_parameters,
    next_token_loss,
)


class TinyTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.config = TransformerConfig(
            vocab_size=17, max_seq_len=10, d_model=16, n_heads=4, n_layers=2
        )
        self.model = TinyGPT(self.config).eval()

    def test_logits_and_attention_shapes(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 7))
        logits, attentions = self.model(tokens, return_attentions=True)
        self.assertEqual(tuple(logits.shape), (2, 7, self.config.vocab_size))
        self.assertEqual(len(attentions), self.config.n_layers)
        self.assertEqual(tuple(attentions[0].shape), (2, 4, 7, 7))

    def test_attention_is_normalized_and_masks_future(self):
        attention = CausalSelfAttention(d_model=8, n_heads=2).eval()
        x = torch.randn(1, 5, 8)
        _, weights = attention(x)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1, 2, 5), atol=1e-6))
        future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.equal(weights[0, :, future], torch.zeros_like(weights[0, :, future])))

    def test_future_tokens_do_not_change_earlier_logits(self):
        tokens_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
        tokens_b = tokens_a.clone()
        tokens_b[:, 4:] = torch.tensor([[11, 12]])
        logits_a = self.model(tokens_a)
        logits_b = self.model(tokens_b)
        self.assertTrue(torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6, rtol=1e-6))

    def test_next_token_loss_and_backward_are_finite(self):
        tokens = torch.randint(0, self.config.vocab_size, (3, 8))
        model = TinyGPT(self.config).train()
        loss = next_token_loss(model(tokens), tokens)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(gradients), 0)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))

    def test_rejects_sequence_longer_than_position_embedding(self):
        tokens = torch.zeros(1, self.config.max_seq_len + 1, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            self.model(tokens)

    def test_config_rejects_incompatible_head_dimension(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerConfig(vocab_size=10, d_model=10, n_heads=3)

    def test_parameter_count_is_positive(self):
        self.assertGreater(count_parameters(self.model), 0)


if __name__ == "__main__":
    unittest.main()
