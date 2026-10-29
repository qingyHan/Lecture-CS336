"""Lecture 7: CPU simulations for collective operations and communication models.

This module simulates collective semantics on one CPU process. It is not a
network benchmark and does not replace torch.distributed/NCCL execution.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _validate_rank_tensors(rank_tensors: Sequence[Tensor]) -> tuple[tuple[int, ...], torch.dtype]:
    if not rank_tensors:
        raise ValueError("at least one rank tensor is required")
    shape = tuple(rank_tensors[0].shape)
    dtype = rank_tensors[0].dtype
    for rank, value in enumerate(rank_tensors):
        if tuple(value.shape) != shape:
            raise ValueError(f"rank {rank} has a different tensor shape")
        if value.dtype != dtype:
            raise ValueError(f"rank {rank} has a different dtype")
        if value.device.type != "cpu":
            raise ValueError("this teaching simulator accepts CPU tensors only")
    return shape, dtype


def all_reduce_sum(rank_tensors: Sequence[Tensor]) -> list[Tensor]:
    """Simulate sum all-reduce: every rank receives the elementwise total."""
    _validate_rank_tensors(rank_tensors)
    total = torch.stack([value.clone() for value in rank_tensors], dim=0).sum(dim=0)
    return [total.clone() for _ in rank_tensors]


def all_reduce_mean(rank_tensors: Sequence[Tensor]) -> list[Tensor]:
    """Simulate rank-mean all-reduce; this is a sample mean only for equal local counts."""
    return [value / len(rank_tensors) for value in all_reduce_sum(rank_tensors)]


def weighted_mean_gradient(local_mean_gradients: Sequence[Tensor], sample_counts: Sequence[int]) -> Tensor:
    """Combine local mean gradients using their sample counts as weights."""
    if len(local_mean_gradients) == 0 or len(local_mean_gradients) != len(sample_counts):
        raise ValueError("gradients and sample_counts must be non-empty and have equal length")
    shape, _ = _validate_rank_tensors(local_mean_gradients)
    if any(not isinstance(count, int) or count <= 0 for count in sample_counts):
        raise ValueError("sample counts must be positive integers")
    total_samples = sum(sample_counts)
    weighted = torch.zeros(shape, dtype=local_mean_gradients[0].dtype)
    for gradient, count in zip(local_mean_gradients, sample_counts):
        weighted += gradient * count
    return weighted / total_samples


def ring_all_reduce_sum(rank_tensors: Sequence[Tensor]) -> list[Tensor]:
    """Simulate ring reduce-scatter + all-gather over equal flat chunks.

    For simple deterministic teaching, the flattened element count must divide
    evenly by world_size. Every returned rank tensor should equal the global sum.
    """
    shape, _ = _validate_rank_tensors(rank_tensors)
    world_size = len(rank_tensors)
    flat_values = [value.contiguous().view(-1) for value in rank_tensors]
    numel = flat_values[0].numel()
    if numel == 0:
        raise ValueError("rank tensors must not be empty")
    if numel % world_size:
        raise ValueError("tensor element count must be divisible by world_size")
    chunk_size = numel // world_size
    chunks = [
        [flat_values[rank][index * chunk_size:(index + 1) * chunk_size].clone()
         for index in range(world_size)]
        for rank in range(world_size)
    ]

    # Reduce-scatter: each step forwards one partial chunk and accumulates the
    # received chunk. After P-1 steps each rank owns one fully reduced chunk.
    for step in range(world_size - 1):
        messages = [chunks[rank][(rank - step) % world_size].clone() for rank in range(world_size)]
        for rank in range(world_size):
            receive_index = (rank - step - 1) % world_size
            chunks[rank][receive_index] += messages[(rank - 1) % world_size]

    # All-gather: circulate each fully reduced chunk so every rank gets all chunks.
    for step in range(world_size - 1):
        messages = [chunks[rank][(rank + 1 - step) % world_size].clone() for rank in range(world_size)]
        for rank in range(world_size):
            receive_index = (rank - step) % world_size
            chunks[rank][receive_index] = messages[(rank - 1) % world_size]

    return [torch.cat(rank_chunks).reshape(shape) for rank_chunks in chunks]


def ring_all_reduce_bytes_per_rank(numel: int, element_size_bytes: int, world_size: int) -> float:
    """Approximate bytes sent per rank by ring all-reduce: 2(P-1)/P * tensor_bytes."""
    if numel < 0 or element_size_bytes <= 0 or world_size <= 0:
        raise ValueError("numel must be non-negative; element size and world size must be positive")
    if world_size == 1:
        return 0.0
    return 2.0 * (world_size - 1) / world_size * numel * element_size_bytes


def ring_all_reduce_time_seconds(alpha_seconds: float, bandwidth_bytes_per_second: float, tensor_bytes: int, world_size: int) -> float:
    """Simple alpha-beta ring model; excludes contention, overlap and protocol effects."""
    if alpha_seconds < 0 or bandwidth_bytes_per_second <= 0 or tensor_bytes < 0 or world_size <= 0:
        raise ValueError("alpha/bytes must be non-negative; bandwidth/world_size must be positive")
    if world_size == 1:
        return 0.0
    p = world_size
    return 2 * (p - 1) * alpha_seconds + (2 * (p - 1) / p) * tensor_bytes / bandwidth_bytes_per_second


def ring_all_reduce_total_bytes(numel: int, element_size_bytes: int, world_size: int) -> float:
    """Approximate total bytes sent across all ranks in the ring."""
    return world_size * ring_all_reduce_bytes_per_rank(numel, element_size_bytes, world_size)


def run_demo() -> None:
    rank_values = [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0]),
        torch.tensor([100.0, 200.0, 300.0, 400.0]),
        torch.tensor([1000.0, 2000.0, 3000.0, 4000.0]),
    ]
    expected = all_reduce_sum(rank_values)[0]
    ring_results = ring_all_reduce_sum(rank_values)
    print("expected all-reduce sum:", expected.tolist())
    print("ring result per rank:", [result.tolist() for result in ring_results])
    print("all ranks match:", all(torch.equal(result, expected) for result in ring_results))
    print("bytes sent per rank for 1 MiB tensor, P=4:",
          f"{ring_all_reduce_bytes_per_rank(1024 * 1024, 1, 4) / 1024**2:.2f} MiB")
    print("alpha-beta estimate for 1 MiB, P=4:",
          f"{ring_all_reduce_time_seconds(5e-6, 20e9, 1024**2, 4) * 1e6:.2f} us")


if __name__ == "__main__":
    run_demo()
