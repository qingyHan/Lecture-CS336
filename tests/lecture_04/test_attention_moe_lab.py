import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_04"))

from attention_moe_lab import (  # noqa: E402
    GroupedQueryAttention, TopKMoE, expert_loads, kv_cache_bytes, load_balance_loss,
)


class AttentionAndMoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_gqa_output_and_attention_shapes(self):
        layer = GroupedQueryAttention(d_model=32, n_query_heads=4, n_kv_heads=2)
        output, weights = layer(torch.randn(2, 7, 32))
        self.assertEqual(tuple(output.shape), (2, 7, 32))
        self.assertEqual(tuple(weights.shape), (2, 4, 7, 7))

    def test_gqa_masks_future_positions(self):
        layer = GroupedQueryAttention(d_model=16, n_query_heads=4, n_kv_heads=2).eval()
        _, weights = layer(torch.randn(1, 5, 16))
        future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.equal(weights[0, :, future], torch.zeros_like(weights[0, :, future])))

    def test_future_tokens_do_not_change_earlier_outputs(self):
        layer = GroupedQueryAttention(d_model=16, n_query_heads=4, n_kv_heads=2).eval()
        a = torch.tensor([[[1.0] * 16, [2.0] * 16, [3.0] * 16, [4.0] * 16, [5.0] * 16]])
        b = a.clone()
        b[:, 3:] = -7.0
        out_a, _ = layer(a)
        out_b, _ = layer(b)
        self.assertTrue(torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-6, rtol=1e-6))

    def test_kv_cache_formula_and_ratio(self):
        common = dict(n_layers=12, batch_size=2, seq_len=128, head_dim=64, bytes_per_element=2)
        gqa = kv_cache_bytes(**common, n_kv_heads=4)
        mha = kv_cache_bytes(**common, n_kv_heads=16)
        self.assertEqual(gqa, 3 * 1024 * 1024)
        self.assertEqual(mha, 4 * gqa)

    def test_topk_moe_shapes_and_selected_weights_sum_to_one(self):
        moe = TopKMoE(d_model=12, hidden_dim=24, n_experts=5, top_k=2)
        output, routing = moe(torch.randn(2, 6, 12), return_routing=True)
        self.assertEqual(tuple(output.shape), (2, 6, 12))
        self.assertEqual(tuple(routing["topk_indices"].shape), (2, 6, 2))
        self.assertTrue(torch.allclose(routing["topk_weights"].sum(-1), torch.ones(2, 6), atol=1e-6))
        self.assertTrue((routing["topk_indices"] < 5).all().item())

    def test_expert_loads_count_assignments_not_unique_tokens(self):
        indices = torch.tensor([[[0, 1], [1, 2]], [[0, 2], [2, 3]]])
        loads = expert_loads(indices, n_experts=4)
        self.assertEqual(loads.tolist(), [2, 2, 3, 1])
        self.assertEqual(int(loads.sum()), indices.numel())

    def test_load_balance_proxy_has_finite_gradient(self):
        logits = torch.randn(2, 5, 4, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        indices = probs.detach().topk(2, dim=-1).indices
        loss = load_balance_loss(probs, indices)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_router_and_expert_parameters_receive_gradients(self):
        moe = TopKMoE(d_model=8, hidden_dim=16, n_experts=4, top_k=2)
        moe(torch.randn(2, 3, 8)).square().mean().backward()
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertTrue(torch.isfinite(moe.router.weight.grad).all().item())
        self.assertTrue(any(p.grad is not None for e in moe.experts for p in e.parameters()))

    def test_invalid_head_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            GroupedQueryAttention(d_model=24, n_query_heads=6, n_kv_heads=4)

    def test_invalid_cache_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            kv_cache_bytes(n_layers=0, batch_size=1, seq_len=4, n_kv_heads=1, head_dim=8, bytes_per_element=2)


if __name__ == "__main__":
    unittest.main()
