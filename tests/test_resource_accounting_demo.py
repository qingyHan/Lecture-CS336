"""第二讲教学代码的单元测试，覆盖公式与实际张量形状。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import torch
from einops import einsum, rearrange, reduce
from resource_accounting_demo import (
    matmul_flops_approx,
    matmul_flops_exact,
    matmul_min_bytes,
    roofline_limit,
    tensor_nbytes,
)


class ResourceAccountingTests(unittest.TestCase):
    def test_tensor_bytes_and_dtype(self):
        self.assertEqual(tensor_nbytes(torch.zeros(4, 8, dtype=torch.float32)), 128)
        self.assertEqual(tensor_nbytes(torch.zeros(4, 8, dtype=torch.bfloat16)), 64)

    def test_matmul_flops(self):
        self.assertEqual(matmul_flops_exact(2, 3, 4), 40)
        self.assertEqual(matmul_flops_approx(2, 3, 4), 48)
        with self.assertRaises(ValueError):
            matmul_flops_exact(2, 0, 4)

    def test_arithmetic_intensity_bound(self):
        self.assertEqual(matmul_min_bytes(2, 3, 4, 2), 52)
        self.assertEqual(roofline_limit(0.25, 100, 40), 10.0)
        self.assertEqual(roofline_limit(10, 100, 40), 100.0)

    def test_einops_matches_pytorch(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        w = torch.ones(4, 5)
        self.assertTrue(torch.equal(einsum(x, w, "batch seq hidden, hidden out -> batch seq out"), x @ w))
        self.assertEqual(tuple(reduce(x, "batch seq hidden -> batch hidden", "mean").shape), (2, 4))
        transformed = rearrange(x, "batch seq (heads dim) -> batch heads seq dim", heads=2)
        self.assertEqual(tuple(transformed.shape), (2, 2, 3, 2))
        self.assertTrue(torch.equal(rearrange(transformed, "batch heads seq dim -> batch seq (heads dim)"), x))

    def test_autograd_gradients(self):
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        w = torch.ones((3, 2), requires_grad=True)
        (x @ w).sum().backward()
        self.assertTrue(torch.equal(w.grad, torch.tensor([[5., 5.], [7., 7.], [9., 9.]])))


if __name__ == "__main__":
    unittest.main()
