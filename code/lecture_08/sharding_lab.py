"""Lecture 8: ZeRO/FSDP memory accounting and activation-checkpointing probe.

Memory estimates are transparent teaching models, not exact framework allocator
predictions. The optional CUDA probe measures a small single-GPU MLP step; it is
not a multi-GPU FSDP benchmark.
"""
from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


STAGE_NAMES = {
    0: "DDP / ZeRO-0",
    1: "ZeRO-1: optimizer states sharded",
    2: "ZeRO-2: gradients + optimizer states sharded",
    3: "ZeRO-3 / FSDP full sharding",
}


def _non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def partition_sizes(total_items: int, world_size: int) -> list[int]:
    """Return balanced shard sizes; earlier ranks receive any remainder."""
    _non_negative_int(total_items, "total_items")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    quotient, remainder = divmod(total_items, world_size)
    return [quotient + (rank < remainder) for rank in range(world_size)]


def estimate_zero_stage_memory(
    parameter_count: int,
    *,
    world_size: int,
    stage: int,
    weight_bytes_per_param: int = 4,
    gradient_bytes_per_param: int = 4,
    optimizer_bytes_per_param: int = 8,
    activation_bytes: int = 0,
    gathered_parameter_count: int = 0,
    communication_buffer_bytes: int = 0,
) -> dict[str, int | str]:
    """Estimate per-rank training memory using a simplified ZeRO state model.

    Stage 0: parameters, gradients and optimizer states are replicated.
    Stage 1: optimizer states are sharded.
    Stage 2: gradients and optimizer states are sharded.
    Stage 3: parameters, gradients and optimizer states are sharded.

    The local shard count uses ceil(P/world_size), a conservative per-rank bound
    for uneven partitioning. Stage 3 optionally adds one transient full-parameter
    FSDP unit (gathered_parameter_count) on top of persistent local state shards.
    Activations and communication buffers are user-supplied estimates.
    """
    _non_negative_int(parameter_count, "parameter_count")
    if parameter_count == 0:
        raise ValueError("parameter_count must be positive")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if stage not in STAGE_NAMES:
        raise ValueError("stage must be one of 0, 1, 2, 3")
    for value, name in (
        (weight_bytes_per_param, "weight_bytes_per_param"),
        (gradient_bytes_per_param, "gradient_bytes_per_param"),
        (optimizer_bytes_per_param, "optimizer_bytes_per_param"),
        (activation_bytes, "activation_bytes"),
        (gathered_parameter_count, "gathered_parameter_count"),
        (communication_buffer_bytes, "communication_buffer_bytes"),
    ):
        _non_negative_int(value, name)
    if gathered_parameter_count > parameter_count:
        raise ValueError("gathered_parameter_count cannot exceed parameter_count")

    local_count = max(partition_sizes(parameter_count, world_size))
    parameter_count_local = local_count if stage == 3 else parameter_count
    gradient_count_local = local_count if stage >= 2 else parameter_count
    optimizer_count_local = parameter_count if stage == 0 else local_count

    parameter_bytes = parameter_count_local * weight_bytes_per_param
    gradient_bytes = gradient_count_local * gradient_bytes_per_param
    optimizer_bytes = optimizer_count_local * optimizer_bytes_per_param
    state_bytes = parameter_bytes + gradient_bytes + optimizer_bytes
    transient_gather_bytes = (
        gathered_parameter_count * weight_bytes_per_param if stage == 3 else 0
    )
    estimated_peak_bytes = (
        state_bytes
        + transient_gather_bytes
        + activation_bytes
        + communication_buffer_bytes
    )
    return {
        "stage": stage,
        "stage_name": STAGE_NAMES[stage],
        "parameter_count": parameter_count,
        "world_size": world_size,
        "local_parameter_count_upper_bound": local_count,
        "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "persistent_state_bytes": state_bytes,
        "transient_gather_bytes": transient_gather_bytes,
        "activation_bytes": activation_bytes,
        "communication_buffer_bytes": communication_buffer_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
    }


def bytes_to_gib(byte_count: int | float) -> float:
    """Convert bytes to GiB (base 1024), not decimal GB."""
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    return byte_count / (1024**3)


def activation_checkpoint_tradeoff(
    saved_activation_bytes: int, recompute_flops: int
) -> dict[str, int]:
    """Record the user-supplied memory/compute tradeoff for checkpointing."""
    _non_negative_int(saved_activation_bytes, "saved_activation_bytes")
    _non_negative_int(recompute_flops, "recompute_flops")
    return {
        "saved_activation_bytes": saved_activation_bytes,
        "recompute_flops": recompute_flops,
    }


class TinyBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.fc2 = nn.Linear(hidden_dim * expansion, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class TinyStack(nn.Module):
    """A small MLP stack used to demonstrate activation checkpointing."""

    def __init__(
        self, hidden_dim: int = 256, num_layers: int = 4, expansion: int = 4
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_layers <= 0 or expansion <= 0:
            raise ValueError("hidden_dim, num_layers and expansion must be positive")
        self.layers = nn.ModuleList(
            TinyBlock(hidden_dim, expansion) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, *, checkpoint_activations: bool = False) -> Tensor:
        for layer in self.layers:
            if checkpoint_activations and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.head(self.norm(x))


def cuda_checkpointing_probe(
    *,
    batch_size: int = 2,
    sequence_length: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 4,
    expansion: int = 4,
    warmup: int = 2,
    iterations: int = 5,
    seed: int = 8,
) -> dict[str, dict[str, Any]]:
    """Measure a tiny single-GPU train step with/without activation checkpointing.

    CUDA events time steady-state steps after warm-up. Peak allocated memory is
    a local observation; it does not measure all driver/reserved memory and does
    not compare FSDP or multi-GPU sharding.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this memory probe")
    if min(batch_size, sequence_length, hidden_dim, num_layers, expansion) <= 0:
        raise ValueError("batch, sequence, hidden, layers and expansion must be positive")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(batch_size, sequence_length, hidden_dim, device=device)
    target = torch.randn_like(x)
    results: dict[str, dict[str, Any]] = {}

    for use_checkpoint in (False, True):
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = TinyStack(hidden_dim, num_layers, expansion).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, checkpoint_activations=use_checkpoint)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            return loss.detach()

        for _ in range(warmup):
            loss = step()
        torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        baseline_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            loss = step()
        end.record()
        torch.cuda.synchronize(device)

        peak_bytes = torch.cuda.max_memory_allocated(device)
        key = "checkpointing_on" if use_checkpoint else "checkpointing_off"
        results[key] = {
            "mean_step_ms": start.elapsed_time(end) / iterations,
            "baseline_allocated_bytes": baseline_bytes,
            "peak_allocated_bytes": peak_bytes,
            "peak_above_baseline_bytes": max(0, peak_bytes - baseline_bytes),
            "final_loss": float(loss.item()),
        }
        del model, optimizer, loss
        torch.cuda.empty_cache()

    results["metadata"] = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "warmup": warmup,
        "iterations": iterations,
    }
    return results


def run_demo() -> None:
    parameters = 1_000_000_000
    print("Illustrative 1B-parameter model, FP32 weights/gradients, Adam moments:")
    for stage in range(4):
        result = estimate_zero_stage_memory(
            parameters,
            world_size=4,
            stage=stage,
            gathered_parameter_count=50_000_000,
        )
        print(
            f"{result['stage_name']}: persistent state="
            f"{bytes_to_gib(result['persistent_state_bytes']):.2f} GiB; "
            f"transient gathered params="
            f"{bytes_to_gib(result['transient_gather_bytes']):.2f} GiB"
        )
    print("These are simplified estimates, not measured GPU peak memory.")
    if torch.cuda.is_available():
        print("CUDA available. Run with --cuda-probe to measure a tiny single-GPU step.")
    else:
        print("CUDA unavailable in this process; no GPU measurement was attempted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-probe", action="store_true", help="measure a tiny CUDA training step")
    args = parser.parse_args()
    if args.cuda_probe:
        print(cuda_checkpointing_probe())
    else:
        run_demo()
