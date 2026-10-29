import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_10"))

from kv_cache_lab import (  # noqa: E402
    kv_cache_bytes_per_token,
    kv_cache_memory_report,
    max_tokens_for_kv_budget,
    summarize_request_trace,
)


class KVCacheLabTests(unittest.TestCase):
    def test_kv_bytes_per_token_formula(self):
        # 2 (K/V) * 32 layers * 8 KV heads * 128 dim * 2 bytes.
        self.assertEqual(kv_cache_bytes_per_token(32, 8, 128), 131072)

    def test_int8_and_int4_payload_ratios(self):
        bf16 = kv_cache_bytes_per_token(32, 8, 128, bits_per_element=16)
        int8 = kv_cache_bytes_per_token(32, 8, 128, bits_per_element=8)
        int4 = kv_cache_bytes_per_token(32, 8, 128, bits_per_element=4)
        self.assertEqual(bf16, 4 * int4)
        self.assertEqual(bf16, 2 * int8)

    def test_contiguous_padding_and_paged_allocation(self):
        report = kv_cache_memory_report(
            [1, 17, 33], num_layers=1, num_kv_heads=1, head_dim=4,
            bits_per_element=16, block_size=16,
        )
        self.assertEqual(report["logical_tokens"], 51)
        self.assertEqual(report["contiguous_allocated_tokens"], 99)
        self.assertEqual(report["page_count"], 6)
        self.assertEqual(report["paged_allocated_tokens"], 96)
        self.assertEqual(report["logical_bytes"], 816)
        self.assertEqual(report["contiguous_allocated_bytes"], 1584)
        self.assertEqual(report["paged_allocated_bytes"], 1536)
        self.assertLess(report["paged_waste_bytes"], report["contiguous_waste_bytes"])

    def test_paged_utilization_is_bounded(self):
        report = kv_cache_memory_report(
            [15, 17, 31], num_layers=2, num_kv_heads=4, head_dim=64, block_size=16
        )
        self.assertGreaterEqual(report["paged_utilization"], 0)
        self.assertLessEqual(report["paged_utilization"], 1)

    def test_empty_or_invalid_sequence_lengths_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            kv_cache_memory_report([], num_layers=1, num_kv_heads=1, head_dim=8)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            kv_cache_memory_report([10, -1], num_layers=1, num_kv_heads=1, head_dim=8)
        with self.assertRaisesRegex(ValueError, "positive"):
            kv_cache_memory_report([10], num_layers=1, num_kv_heads=1, head_dim=8, block_size=0)

    def test_invalid_head_or_bit_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            kv_cache_bytes_per_token(0, 8, 128)
        with self.assertRaisesRegex(ValueError, "one of 4"):
            kv_cache_bytes_per_token(2, 4, 64, bits_per_element=6)

    def test_token_capacity_from_budget(self):
        per_token = kv_cache_bytes_per_token(1, 1, 4, bits_per_element=16)
        self.assertEqual(per_token, 16)
        self.assertEqual(
            max_tokens_for_kv_budget(160, num_layers=1, num_kv_heads=1, head_dim=4),
            10,
        )
        self.assertEqual(
            max_tokens_for_kv_budget(0, num_layers=1, num_kv_heads=1, head_dim=4),
            0,
        )

    def test_request_trace_summary(self):
        report = summarize_request_trace(
            [
                {"prompt_tokens": 100, "output_tokens": 20},
                {"prompt_tokens": 50, "output_tokens": 10},
            ],
            num_layers=2, num_kv_heads=2, head_dim=8,
        )
        self.assertEqual(report["total_prompt_tokens"], 150)
        self.assertEqual(report["total_output_tokens"], 30)
        self.assertEqual(report["final_sequence_lengths"], [120, 60])
        self.assertEqual(report["max_final_context_tokens"], 120)
        self.assertEqual(report["cache_memory"]["logical_tokens"], 180)

    def test_request_trace_rejects_missing_or_invalid_fields(self):
        with self.assertRaisesRegex(ValueError, "must contain"):
            summarize_request_trace([{"prompt_tokens": 10}], num_layers=1, num_kv_heads=1, head_dim=8)
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_request_trace(
                [{"prompt_tokens": 0, "output_tokens": 2}],
                num_layers=1, num_kv_heads=1, head_dim=8,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            summarize_request_trace(
                [{"prompt_tokens": 10, "output_tokens": -1}],
                num_layers=1, num_kv_heads=1, head_dim=8,
            )

    def test_quantized_cache_payload_budget(self):
        budget = 4 * 1024**3
        bf16_tokens = max_tokens_for_kv_budget(
            budget, num_layers=32, num_kv_heads=8, head_dim=128, bits_per_element=16
        )
        int8_tokens = max_tokens_for_kv_budget(
            budget, num_layers=32, num_kv_heads=8, head_dim=128, bits_per_element=8
        )
        self.assertEqual(int8_tokens, 2 * bf16_tokens)


if __name__ == "__main__":
    unittest.main()
