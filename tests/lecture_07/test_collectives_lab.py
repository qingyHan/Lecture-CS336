import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_07"))

from collectives_lab import (  # noqa: E402
    all_reduce_mean, all_reduce_sum, ring_all_reduce_bytes_per_rank,
    ring_all_reduce_sum, ring_all_reduce_time_seconds, ring_all_reduce_total_bytes,
    weighted_mean_gradient,
)


class CollectiveLabTests(unittest.TestCase):
    def test_all_reduce_sum_replicates_global_sum(self):
        ranks = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        results = all_reduce_sum(ranks)
        for result in results:
            self.assertTrue(torch.equal(result, torch.tensor([9.0, 12.0])))

    def test_all_reduce_mean_for_equal_local_counts(self):
        ranks = [torch.tensor([1.0, 3.0]), torch.tensor([5.0, 7.0])]
        results = all_reduce_mean(ranks)
        for result in results:
            self.assertTrue(torch.equal(result, torch.tensor([3.0, 5.0])))

    def test_ring_all_reduce_matches_reference_for_multiple_world_sizes(self):
        for world_size in (1, 2, 3, 4):
            ranks = [torch.arange(12, dtype=torch.float32) + 10 * rank for rank in range(world_size)]
            results = ring_all_reduce_sum(ranks)
            expected = torch.stack(ranks).sum(0)
            for result in results:
                self.assertTrue(torch.equal(result, expected), msg=f"world_size={world_size}")

    def test_ring_all_reduce_requires_even_flat_chunking(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ring_all_reduce_sum([torch.arange(5.0), torch.arange(5.0)])

    def test_weighted_mean_gradient_handles_unequal_sample_counts(self):
        g0 = torch.tensor([1.0, 3.0])
        g1 = torch.tensor([5.0, 7.0])
        weighted = weighted_mean_gradient([g0, g1], [1, 3])
        self.assertTrue(torch.equal(weighted, torch.tensor([4.0, 6.0])))

    def test_weighted_mean_differs_from_rank_mean_for_unequal_counts(self):
        g0 = torch.tensor([1.0, 3.0])
        g1 = torch.tensor([5.0, 7.0])
        rank_mean = all_reduce_mean([g0, g1])[0]
        sample_mean = weighted_mean_gradient([g0, g1], [1, 3])
        self.assertTrue(torch.equal(rank_mean, torch.tensor([3.0, 5.0])))
        self.assertTrue(torch.equal(sample_mean, torch.tensor([4.0, 6.0])))
        self.assertFalse(torch.equal(rank_mean, sample_mean))

    def test_ring_per_rank_bytes_formula(self):
        # 1,024 FP32 elements = 4,096 bytes; P=4 => 1.5 * tensor bytes.
        self.assertEqual(ring_all_reduce_bytes_per_rank(1024, 4, 4), 6144.0)
        self.assertEqual(ring_all_reduce_bytes_per_rank(1024, 4, 1), 0.0)

    def test_ring_alpha_beta_time_formula(self):
        observed = ring_all_reduce_time_seconds(1e-6, 1e9, 4096, 4)
        self.assertAlmostEqual(observed, 12.144e-6, places=12)
        self.assertEqual(ring_all_reduce_time_seconds(1e-6, 1e9, 4096, 1), 0.0)

    def test_ring_total_network_bytes(self):
        per_rank = ring_all_reduce_bytes_per_rank(2048, 2, 4)
        self.assertEqual(ring_all_reduce_total_bytes(2048, 2, 4), 4 * per_rank)

    def test_invalid_rank_tensor_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different tensor shape"):
            all_reduce_sum([torch.ones(2), torch.ones(3)])

    def test_weighted_gradient_requires_positive_sample_counts(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_mean_gradient([torch.ones(2)], [0])


if __name__ == "__main__":
    unittest.main()
