import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_05"))

from accelerator_lab import (  # noqa: E402
    arithmetic_intensity, benchmark_matmul, dtype_bytes, matmul_flops,
    matmul_ideal_bytes, roofline_bound, tensor_bytes,
)


class AcceleratorLabTests(unittest.TestCase):
    def test_dtype_bytes_and_tensor_storage(self):
        self.assertEqual(dtype_bytes(torch.float32), 4)
        self.assertEqual(dtype_bytes(torch.bfloat16), 2)
        self.assertEqual(tensor_bytes((2, 3, 4), torch.float32), 96)
        self.assertEqual(tensor_bytes((0, 5), torch.float16), 0)

    def test_matmul_flops_counts_multiply_add_as_two(self):
        self.assertEqual(matmul_flops(2, 3, 4), 48)

    def test_ideal_matmul_bytes_include_inputs_and_output(self):
        # A: 2x4, B: 4x3, C: 2x3; FP32 => (8 + 12 + 6) * 4 bytes.
        self.assertEqual(matmul_ideal_bytes(2, 3, 4, torch.float32), 104)

    def test_arithmetic_intensity(self):
        self.assertEqual(arithmetic_intensity(200, 20), 10.0)

    def test_roofline_memory_bound_case(self):
        self.assertEqual(roofline_bound(1000, 100, 2), 200)

    def test_roofline_compute_bound_case(self):
        self.assertEqual(roofline_bound(1000, 100, 20), 1000)

    def test_invalid_dimensions_and_zero_traffic_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            matmul_flops(0, 3, 4)
        with self.assertRaisesRegex(ValueError, "positive"):
            arithmetic_intensity(10, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            tensor_bytes((-1, 4), torch.float32)

    def test_benchmark_returns_finite_latency_and_error(self):
        result = benchmark_matmul(16, 12, 8, device="cpu", warmup=1, iterations=2, seed=7)
        self.assertGreater(result["latency_ms"], 0)
        self.assertGreater(result["estimated_flops"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(result["relative_rmse_vs_fp32"])).item())
        self.assertLess(result["relative_rmse_vs_fp32"], 1e-6)
        self.assertEqual(result["device"], "cpu")

    def test_cuda_request_without_cuda_fails_explicitly(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available in this environment")
        with self.assertRaisesRegex(RuntimeError, "not available"):
            benchmark_matmul(4, 4, 4, device="cuda", iterations=1)


if __name__ == "__main__":
    unittest.main()
