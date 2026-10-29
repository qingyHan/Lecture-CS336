"""Lecture 6: CPU references and optional Triton GPU kernels.

The PyTorch reference functions are testable on CPU. Triton wrappers require
both the Triton package and a CUDA tensor. No GPU performance claim is made by
this module; the custom kernels must be compiled and measured on target hardware.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def gelu_reference(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


def softmax_reference(x: Tensor) -> Tensor:
    if x.ndim != 2:
        raise ValueError("softmax input must be a 2D [rows, columns] tensor")
    return F.softmax(x, dim=-1)


def row_sum_reference(x: Tensor) -> Tensor:
    if x.ndim != 2:
        raise ValueError("row_sum input must be a 2D [rows, columns] tensor")
    return x.sum(dim=1)


def matmul_relu_reference(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul inputs must both be 2D")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul inner dimensions must match")
    return F.relu(a @ b)


def max_abs_error(reference: Tensor, actual: Tensor) -> float:
    if reference.shape != actual.shape:
        raise ValueError("reference and actual tensors must have the same shape")
    if reference.numel() == 0:
        return 0.0
    return (reference.float() - actual.float()).abs().max().item()


def relative_l2_error(reference: Tensor, actual: Tensor) -> float:
    if reference.shape != actual.shape:
        raise ValueError("reference and actual tensors must have the same shape")
    numerator = torch.linalg.vector_norm(reference.float() - actual.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return (numerator / denominator).item()


def _require_triton_cuda(x: Tensor, name: str = "input") -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed; install Triton in a compatible CUDA environment")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if x.dtype != torch.float32:
        raise ValueError(f"{name} must use torch.float32 for this reference kernel lab")


if TRITON_AVAILABLE:
    @triton.jit
    def _gelu_kernel(X, Y, N, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0.0)
        inner = 0.7978845608 * (x + 0.044715 * x * x * x)
        exp_neg = tl.exp(-2.0 * tl.abs(inner))
        sign = tl.where(inner >= 0.0, 1.0, -1.0)
        tanh_inner = sign * (1.0 - exp_neg) / (1.0 + exp_neg)
        y = 0.5 * x * (1.0 + tanh_inner)
        tl.store(Y + offsets, y, mask=mask)


    @triton.jit
    def _softmax_kernel(X, Y, ROWS, COLS, X_ROW_STRIDE, Y_ROW_STRIDE, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        x_ptrs = X + row * X_ROW_STRIDE + cols
        x = tl.load(x_ptrs, mask=cols < COLS, other=float("-inf"))
        x = x - tl.max(x, axis=0)
        exp_x = tl.exp(x)
        probs = exp_x / tl.sum(exp_x, axis=0)
        y_ptrs = Y + row * Y_ROW_STRIDE + cols
        tl.store(y_ptrs, probs, mask=cols < COLS)


    @triton.jit
    def _row_sum_kernel(X, Y, COLS, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(0, COLS, BLOCK):
            idx = start + cols
            values = tl.load(X + row * COLS + idx, mask=idx < COLS, other=0.0)
            acc += values
        result = tl.sum(acc, axis=0)
        tl.store(Y + row, result)


    @triton.jit
    def _matmul_relu_kernel(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k_start in range(0, K, BK):
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_start < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] + k_start < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk
        acc = tl.maximum(acc, 0.0)
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gelu_triton(x: Tensor) -> Tensor:
    _require_triton_cuda(x)
    y = torch.empty_like(x)
    n = x.numel()
    _gelu_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return y


def softmax_triton(x: Tensor) -> Tensor:
    _require_triton_cuda(x)
    if x.ndim != 2:
        raise ValueError("softmax input must be a 2D [rows, columns] tensor")
    rows, cols = x.shape
    if cols <= 0 or cols > 8192:
        raise ValueError("this teaching softmax kernel supports 1 <= columns <= 8192")
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    num_warps = 4 if block <= 2048 else 8
    _softmax_kernel[(rows,)](x, y, rows, cols, x.stride(0), y.stride(0), BLOCK=block, num_warps=num_warps)
    return y


def row_sum_triton(x: Tensor, block: int = 1024) -> Tensor:
    _require_triton_cuda(x)
    if x.ndim != 2:
        raise ValueError("row_sum input must be a 2D [rows, columns] tensor")
    if block <= 0 or block & (block - 1):
        raise ValueError("block must be a positive power of two")
    rows, cols = x.shape
    y = torch.empty((rows,), device=x.device, dtype=x.dtype)
    _row_sum_kernel[(rows,)](x, y, cols, BLOCK=block)
    return y


def matmul_relu_triton(a: Tensor, b: Tensor, block_m: int = 32, block_n: int = 32, block_k: int = 32) -> Tensor:
    _require_triton_cuda(a, "a")
    _require_triton_cuda(b, "b")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul inputs must both be 2D")
    if a.device != b.device:
        raise ValueError("matmul inputs must be on the same device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul inner dimensions must match")
    if any(tile < 16 or tile & (tile - 1) for tile in (block_m, block_n, block_k)):
        raise ValueError("tile sizes must be powers of two and at least 16")
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _matmul_relu_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=block_m, BN=block_n, BK=block_k,
    )
    return c
