"""CS336 Lecture 5: accelerator resource and Roofline teaching utilities.

The estimates are deliberately simple. They help reason about shapes, FLOPs,
bytes and upper bounds; they are not substitutes for device-specific profiling.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch


_DTYPE_BYTES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


def dtype_bytes(dtype: torch.dtype) -> int:
    """Return the storage size for a common PyTorch dtype."""
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype for this teaching estimator: {dtype}") from exc


def tensor_bytes(shape: tuple[int, ...] | list[int], dtype: torch.dtype) -> int:
    """Estimate dense tensor storage; excludes allocator and alignment overhead."""
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError("tensor dimensions must be non-negative integers")
    elements = math.prod(shape)
    return elements * dtype_bytes(dtype)


def matmul_flops(m: int, n: int, k: int) -> int:
    """FLOPs for [M,K] @ [K,N], counting each multiply-add as 2 FLOPs."""
    if min(m, n, k) <= 0:
        raise ValueError("M, N and K must be positive")
    return 2 * m * n * k


def matmul_ideal_bytes(m: int, n: int, k: int, dtype: torch.dtype) -> int:
    """Idealized bytes: read A and B once, write C once, all in the same dtype.

    Real HBM traffic can differ due to cache reuse, tiling, write behavior,
    intermediate buffers, alignment, and implementation details.
    """
    if min(m, n, k) <= 0:
        raise ValueError("M, N and K must be positive")
    bpe = dtype_bytes(dtype)
    return (m * k + k * n + m * n) * bpe


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte. The traffic model must be stated alongside this value."""
    if flops < 0:
        raise ValueError("flops must be non-negative")
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def roofline_bound(peak_flops_per_second: float, bandwidth_bytes_per_second: float, intensity_flops_per_byte: float) -> float:
    """Roofline throughput upper bound min(peak compute, bandwidth × intensity)."""
    if peak_flops_per_second <= 0 or bandwidth_bytes_per_second <= 0:
        raise ValueError("peak compute and bandwidth must be positive")
    if intensity_flops_per_byte < 0:
        raise ValueError("arithmetic intensity must be non-negative")
    return min(peak_flops_per_second, bandwidth_bytes_per_second * intensity_flops_per_byte)


def benchmark_matmul(
    m: int,
    n: int,
    k: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
    warmup: int = 3,
    iterations: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Benchmark a small matmul with synchronization appropriate to the device.

    CUDA uses CUDA events; CPU uses perf_counter. The returned latency is a
    local observation for this environment and shape, not a hardware guarantee.
    """
    if min(m, n, k) <= 0:
        raise ValueError("M, N and K must be positive")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if dtype not in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("matmul benchmark supports floating-point dtypes only")

    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("this teaching benchmark supports CPU and CUDA only")
    if selected_device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU float16 matmul support is backend-dependent; choose float32 or bfloat16")

    generator = torch.Generator(device=selected_device).manual_seed(seed)
    a = torch.randn((m, k), generator=generator, device=selected_device, dtype=dtype)
    b = torch.randn((k, n), generator=generator, device=selected_device, dtype=dtype)

    for _ in range(warmup):
        output = a @ b

    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = a @ b
        end.record()
        torch.cuda.synchronize(selected_device)
        latency_ms = start.elapsed_time(end) / iterations
    else:
        start_time = time.perf_counter()
        for _ in range(iterations):
            output = a @ b
        latency_ms = (time.perf_counter() - start_time) * 1000.0 / iterations

    # Accuracy check is outside the timed region and uses FP32 accumulation.
    reference = a.float() @ b.float()
    relative_rmse = (
        torch.linalg.vector_norm(output.float() - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    ).item()
    flops = matmul_flops(m, n, k)
    traffic = matmul_ideal_bytes(m, n, k, dtype)
    return {
        "device": str(selected_device),
        "dtype": str(dtype),
        "shape": {"M": m, "N": n, "K": k},
        "latency_ms": latency_ms,
        "estimated_flops": flops,
        "idealized_bytes": traffic,
        "idealized_arithmetic_intensity_flops_per_byte": arithmetic_intensity(flops, traffic),
        "relative_rmse_vs_fp32": relative_rmse,
        "warmup": warmup,
        "iterations": iterations,
    }


def run_demo() -> None:
    m, n, k = 512, 512, 512
    flops = matmul_flops(m, n, k)
    traffic = matmul_ideal_bytes(m, n, k, torch.float32)
    intensity = arithmetic_intensity(flops, traffic)
    print(f"matmul [{m},{k}] @ [{k},{n}]")
    print(f"FLOPs (multiply-add=2): {flops:,}")
    print(f"idealized FP32 traffic: {traffic:,} bytes")
    print(f"idealized arithmetic intensity: {intensity:.2f} FLOPs/byte")
    result = benchmark_matmul(m, n, k, dtype=torch.float32, warmup=2, iterations=5)
    print(f"device: {result['device']}; mean latency: {result['latency_ms']:.3f} ms")
    print(f"relative RMSE vs FP32 reference: {result['relative_rmse_vs_fp32']:.3e}")
    if result["device"] == "cpu":
        print("CPU fallback only: do not interpret this latency as GPU performance.")


if __name__ == "__main__":
    run_demo()
