import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_08"))

from sharding_lab import (  # noqa: E402
    activation_checkpoint_tradeoff,
    bytes_to_gib,
    cuda_checkpointing_probe,
    estimate_zero_stage_memory,
    partition_sizes,
)


class ShardingLabTests(unittest.TestCase):
    def test_partition_sizes_balance_remainders(self):
        self.assertEqual(partition_sizes(10, 4), [3, 3, 2, 2])
        self.assertEqual(partition_sizes(3, 5), [1, 1, 1, 0, 0])
        self.assertEqual(sum(partition_sizes(101, 8)), 101)

    def test_partition_rejects_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            partition_sizes(-1, 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            partition_sizes(10, 0)

    def test_ddp_memory_formula(self):
        result = estimate_zero_stage_memory(100, world_size=4, stage=0)
        self.assertEqual(result["parameter_bytes"], 400)
        self.assertEqual(result["gradient_bytes"], 400)
        self.assertEqual(result["optimizer_state_bytes"], 800)
        self.assertEqual(result["persistent_state_bytes"], 1600)

    def test_zero_stages_reduce_persistent_state(self):
        states = [
            estimate_zero_stage_memory(101, world_size=4, stage=stage)["persistent_state_bytes"]
            for stage in range(4)
        ]
        self.assertEqual(states, sorted(states, reverse=True))
        self.assertEqual(states, [1616, 1016, 716, 416])

    def test_stage3_adds_transient_gather_only_when_requested(self):
        no_gather = estimate_zero_stage_memory(100, world_size=4, stage=3)
        with_gather = estimate_zero_stage_memory(
            100, world_size=4, stage=3, gathered_parameter_count=20
        )
        self.assertEqual(no_gather["transient_gather_bytes"], 0)
        self.assertEqual(with_gather["transient_gather_bytes"], 80)
        self.assertEqual(
            with_gather["estimated_peak_bytes"] - no_gather["estimated_peak_bytes"], 80
        )
        stage2 = estimate_zero_stage_memory(
            100, world_size=4, stage=2, gathered_parameter_count=20
        )
        self.assertEqual(stage2["transient_gather_bytes"], 0)

    def test_activation_and_communication_buffers_are_additive(self):
        base = estimate_zero_stage_memory(100, world_size=4, stage=3)
        extra = estimate_zero_stage_memory(
            100, world_size=4, stage=3, activation_bytes=1024,
            communication_buffer_bytes=2048,
        )
        self.assertEqual(extra["estimated_peak_bytes"] - base["estimated_peak_bytes"], 3072)

    def test_world_size_one_matches_unsharded_state_for_all_stages(self):
        values = [
            estimate_zero_stage_memory(13, world_size=1, stage=stage)["persistent_state_bytes"]
            for stage in range(4)
        ]
        self.assertEqual(values, [208, 208, 208, 208])

    def test_rejects_invalid_memory_inputs(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            estimate_zero_stage_memory(0, world_size=2, stage=1)
        with self.assertRaisesRegex(ValueError, "one of 0"):
            estimate_zero_stage_memory(10, world_size=2, stage=4)
        with self.assertRaisesRegex(ValueError, "one of 0"):
            estimate_zero_stage_memory(10, world_size=2, stage=True)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            estimate_zero_stage_memory(10, world_size=2, stage=3, gathered_parameter_count=11)

    def test_bytes_to_gib_and_checkpoint_tradeoff(self):
        self.assertEqual(bytes_to_gib(1024**3), 1.0)
        self.assertEqual(
            activation_checkpoint_tradeoff(4096, 12345),
            {"saved_activation_bytes": 4096, "recompute_flops": 12345},
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            activation_checkpoint_tradeoff(-1, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_checkpointing_probe_returns_measurements(self):
        result = cuda_checkpointing_probe(
            batch_size=1, sequence_length=16, hidden_dim=16, num_layers=1,
            expansion=2, warmup=1, iterations=1, seed=5,
        )
        self.assertIn("checkpointing_off", result)
        self.assertIn("checkpointing_on", result)
        for key in ("checkpointing_off", "checkpointing_on"):
            self.assertGreater(result[key]["mean_step_ms"], 0)
            self.assertGreater(result[key]["peak_allocated_bytes"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(result[key]["final_loss"])).item())


if __name__ == "__main__":
    unittest.main()
