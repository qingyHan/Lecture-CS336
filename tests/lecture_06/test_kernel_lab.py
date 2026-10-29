import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_06"))

from kernel_lab import (  # noqa: E402
    TRITON_AVAILABLE, gelu_reference, gelu_triton, max_abs_error,
    matmul_relu_reference, matmul_relu_triton, relative_l2_error,
    row_sum_reference, row_sum_triton, softmax_reference, softmax_triton,
)


class KernelLabTests(unittest.TestCase):
    def test_gelu_reference_matches_pytorch_tanh_approximation(self):
        x = torch.linspace(-5, 5, 101)
        expected = torch.nn.functional.gelu(x, approximate="tanh")
        actual = gelu_reference(x)
        self.assertTrue(torch.allclose(actual, expected, atol=0, rtol=0))

    def test_softmax_is_stable_for_large_logits(self):
        x = torch.tensor([[1000.0, 1001.0, 999.0], [0.0, 0.0, 100.0]])
        y = softmax_reference(x)
        self.assertTrue(torch.isfinite(y).all().item())
        self.assertTrue(torch.allclose(y.sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertGreater(y[1, 2].item(), 0.999)

    def test_softmax_rejects_non_matrix_input(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            softmax_reference(torch.randn(3))

    def test_row_sum_reference(self):
        x = torch.tensor([[1.0, 2, 3, 4], [5, 6, 7, 8]])
        self.assertTrue(torch.equal(row_sum_reference(x), torch.tensor([10.0, 26.0])))

    def test_matmul_relu_reference(self):
        a = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        b = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
        expected = torch.relu(a @ b)
        self.assertTrue(torch.equal(matmul_relu_reference(a, b), expected))

    def test_matmul_reference_rejects_incompatible_shapes(self):
        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            matmul_relu_reference(torch.randn(2, 3), torch.randn(4, 2))

    def test_error_metrics(self):
        reference = torch.tensor([1.0, 2.0])
        self.assertEqual(max_abs_error(reference, reference), 0.0)
        self.assertEqual(relative_l2_error(reference, reference), 0.0)
        self.assertAlmostEqual(max_abs_error(reference, torch.tensor([1.0, 2.5])), 0.5)
        with self.assertRaisesRegex(ValueError, "same shape"):
            max_abs_error(torch.ones(2), torch.ones(3))

    def test_triton_wrapper_fails_clearly_without_cuda_kernel_environment(self):
        if TRITON_AVAILABLE and torch.cuda.is_available():
            self.skipTest("Triton and CUDA are available; GPU wrapper test belongs to GPU QA")
        with self.assertRaises((RuntimeError, ValueError)):
            gelu_triton(torch.randn(8))

    @unittest.skipUnless(TRITON_AVAILABLE and torch.cuda.is_available(), "requires Triton and CUDA")
    def test_triton_gelu_and_softmax_match_references(self):
        x = torch.randn(257, device="cuda", dtype=torch.float32)
        self.assertLess(max_abs_error(gelu_reference(x), gelu_triton(x)), 2e-5)
        matrix = torch.randn(7, 129, device="cuda", dtype=torch.float32)
        self.assertTrue(torch.allclose(softmax_reference(matrix), softmax_triton(matrix), atol=2e-5, rtol=2e-5))

    @unittest.skipUnless(TRITON_AVAILABLE and torch.cuda.is_available(), "requires Triton and CUDA")
    def test_triton_row_sum_matches_reference_for_non_multiple_length(self):
        x = torch.randn(5, 2051, device="cuda", dtype=torch.float32)
        self.assertTrue(torch.allclose(row_sum_reference(x), row_sum_triton(x, block=256), atol=2e-4, rtol=2e-5))

    @unittest.skipUnless(TRITON_AVAILABLE and torch.cuda.is_available(), "requires Triton and CUDA")
    def test_triton_matmul_relu_matches_reference_for_edge_tiles(self):
        a = torch.randn(37, 51, device="cuda", dtype=torch.float32)
        b = torch.randn(51, 29, device="cuda", dtype=torch.float32)
        actual = matmul_relu_triton(a, b, block_m=16, block_n=16, block_k=32)
        expected = matmul_relu_reference(a, b)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-3, rtol=2e-3))


if __name__ == "__main__":
    unittest.main()
