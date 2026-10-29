"""Lecture 10: KV-cache memory accounting for inference serving.

The calculations model KV tensor payload only. They exclude allocator metadata,
quantization scales/zero-points, alignment, model weights, activations and runtime
buffers unless the caller explicitly adds them.
"""
from __future__ import annotations

import math
from typing import Any, Sequence


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def kv_cache_bytes_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    bits_per_element: int = 16,
) -> int:
    """KV payload bytes per cached token: 2 * layers * KV heads * head_dim."""
    _positive_int(num_layers, "num_layers")
    _positive_int(num_kv_heads, "num_kv_heads")
    _positive_int(head_dim, "head_dim")
    if bits_per_element not in (4, 8, 16, 32):
        raise ValueError("bits_per_element must be one of 4, 8, 16, 32")
    total_bits = 2 * num_layers * num_kv_heads * head_dim * bits_per_element
    return math.ceil(total_bits / 8)


def kv_cache_memory_report(
    sequence_lengths: Sequence[int],
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bits_per_element: int = 16,
    block_size: int = 16,
) -> dict[str, int | float]:
    """Compare logical KV bytes with contiguous padding and paged allocation.

    Contiguous allocation reserves len(sequences) * max(sequence_lengths) token
    slots. Paged allocation rounds each sequence up independently to block_size.
    """
    if not sequence_lengths:
        raise ValueError("sequence_lengths must not be empty")
    for length in sequence_lengths:
        _non_negative_int(length, "each sequence length")
    _positive_int(block_size, "block_size")

    per_token_bytes = kv_cache_bytes_per_token(
        num_layers, num_kv_heads, head_dim, bits_per_element=bits_per_element
    )
    lengths = list(sequence_lengths)
    logical_tokens = sum(lengths)
    contiguous_tokens = len(lengths) * max(lengths)
    page_count = sum(math.ceil(length / block_size) for length in lengths)
    paged_tokens = page_count * block_size

    logical_bytes = logical_tokens * per_token_bytes
    contiguous_bytes = contiguous_tokens * per_token_bytes
    paged_bytes = paged_tokens * per_token_bytes
    return {
        "sequence_count": len(lengths),
        "logical_tokens": logical_tokens,
        "contiguous_allocated_tokens": contiguous_tokens,
        "paged_allocated_tokens": paged_tokens,
        "block_size": block_size,
        "page_count": page_count,
        "bytes_per_token": per_token_bytes,
        "logical_bytes": logical_bytes,
        "contiguous_allocated_bytes": contiguous_bytes,
        "paged_allocated_bytes": paged_bytes,
        "contiguous_waste_bytes": contiguous_bytes - logical_bytes,
        "paged_waste_bytes": paged_bytes - logical_bytes,
        "contiguous_utilization": logical_bytes / contiguous_bytes if contiguous_bytes else 1.0,
        "paged_utilization": logical_bytes / paged_bytes if paged_bytes else 1.0,
    }


def max_tokens_for_kv_budget(
    usable_memory_bytes: int,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bits_per_element: int = 16,
) -> int:
    """Upper bound on total cached tokens for a KV payload budget.

    The budget must already exclude weights, activations, workspaces, fragmentation
    and other allocations. Quantization metadata is not included.
    """
    _non_negative_int(usable_memory_bytes, "usable_memory_bytes")
    per_token = kv_cache_bytes_per_token(
        num_layers, num_kv_heads, head_dim, bits_per_element=bits_per_element
    )
    return usable_memory_bytes // per_token


def summarize_request_trace(
    requests: Sequence[dict[str, Any]],
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bits_per_element: int = 16,
    block_size: int = 16,
) -> dict[str, Any]:
    """Summarize a static request trace and estimate final-context KV payload."""
    if not requests:
        raise ValueError("requests must not be empty")
    prompts: list[int] = []
    outputs: list[int] = []
    for index, request in enumerate(requests):
        if "prompt_tokens" not in request or "output_tokens" not in request:
            raise ValueError(f"request {index} must contain prompt_tokens and output_tokens")
        prompt = request["prompt_tokens"]
        output = request["output_tokens"]
        _positive_int(prompt, f"request {index} prompt_tokens")
        _non_negative_int(output, f"request {index} output_tokens")
        prompts.append(prompt)
        outputs.append(output)

    final_lengths = [prompt + output for prompt, output in zip(prompts, outputs)]
    cache = kv_cache_memory_report(
        final_lengths,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bits_per_element=bits_per_element,
        block_size=block_size,
    )
    return {
        "request_count": len(requests),
        "total_prompt_tokens": sum(prompts),
        "total_output_tokens": sum(outputs),
        "mean_prompt_tokens": sum(prompts) / len(prompts),
        "mean_output_tokens": sum(outputs) / len(outputs),
        "max_final_context_tokens": max(final_lengths),
        "final_sequence_lengths": final_lengths,
        "cache_memory": cache,
    }


def bytes_to_gib(byte_count: int | float) -> float:
    """Convert bytes to GiB (base 1024)."""
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    return byte_count / (1024**3)


def run_demo() -> None:
    requests = [
        {"prompt_tokens": 2049, "output_tokens": 128},
        {"prompt_tokens": 4097, "output_tokens": 256},
        {"prompt_tokens": 8193, "output_tokens": 64},
        {"prompt_tokens": 1025, "output_tokens": 192},
        {"prompt_tokens": 6145, "output_tokens": 96},
    ]
    common = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16}
    report = summarize_request_trace(requests, bits_per_element=16, **common)
    cache = report["cache_memory"]
    print("Synthetic request trace; payload-only KV estimates, not a serving benchmark.")
    print("requests:", report["request_count"])
    print("prompt tokens:", report["total_prompt_tokens"], "| output tokens:", report["total_output_tokens"])
    print("final context lengths:", report["final_sequence_lengths"])
    print(f"KV bytes/token (BF16/FP16): {cache['bytes_per_token']:,}")
    print(f"logical KV payload: {bytes_to_gib(cache['logical_bytes']):.3f} GiB")
    print(f"contiguous padded allocation: {bytes_to_gib(cache['contiguous_allocated_bytes']):.3f} GiB")
    print(f"paged allocation: {bytes_to_gib(cache['paged_allocated_bytes']):.3f} GiB")
    print(f"contiguous waste: {bytes_to_gib(cache['contiguous_waste_bytes']):.3f} GiB")
    print(f"paged block waste: {bytes_to_gib(cache['paged_waste_bytes']):.3f} GiB")
    print("Quantization metadata, weights, activations and runtime buffers are excluded.")


if __name__ == "__main__":
    run_demo()
