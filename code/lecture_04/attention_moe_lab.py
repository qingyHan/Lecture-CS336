"""Educational GQA and Top-k MoE implementations for CS336 Lecture 4.

These implementations prioritize inspectable tensor shapes and correctness.
They are not production kernels: GQA explicitly repeats K/V heads for attention
math, and MoE dispatch uses Python expert loops rather than fused/all-to-all
dispatch.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    """Causal GQA. Multiple query heads share each key/value head."""

    def __init__(self, d_model: int, n_query_heads: int, n_kv_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if min(d_model, n_query_heads, n_kv_heads) <= 0:
            raise ValueError("d_model and head counts must be positive")
        if d_model % n_query_heads:
            raise ValueError("d_model must be divisible by n_query_heads")
        if n_query_heads % n_kv_heads:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_query_heads
        self.group_size = n_query_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("input must have shape [batch, time, d_model]")
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Educational expansion only. A production GQA kernel can reuse K/V
        # without materializing this repeated representation.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        attended = self.dropout(weights) @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attended), weights


def kv_cache_bytes(*, n_layers: int, batch_size: int, seq_len: int, n_kv_heads: int, head_dim: int, bytes_per_element: int) -> int:
    """Estimate K+V cache bytes, excluding padding, allocator and metadata overhead."""
    dims = (n_layers, batch_size, seq_len, n_kv_heads, head_dim, bytes_per_element)
    if any(value <= 0 for value in dims):
        raise ValueError("all KV-cache dimensions and bytes_per_element must be positive")
    return 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_element


class FeedForwardExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TopKMoE(nn.Module):
    """Token-choice top-k routing with a readable, non-optimized dispatch loop."""

    def __init__(self, d_model: int, hidden_dim: int, n_experts: int, top_k: int = 2) -> None:
        super().__init__()
        if min(d_model, hidden_dim, n_experts, top_k) <= 0:
            raise ValueError("d_model, hidden_dim, n_experts and top_k must be positive")
        if top_k > n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(FeedForwardExpert(d_model, hidden_dim) for _ in range(n_experts))

    def forward(self, x: Tensor, *, return_routing: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("input must have shape [..., d_model]")
        leading_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_model)
        router_probs = F.softmax(self.router(flat_x), dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize selected probabilities so each token's selected gates sum to one.
        # Other papers/implementations may use a different convention.
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        flat_output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_indices, choice_indices = torch.where(topk_indices == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat_x.index_select(0, token_indices))
            weights = topk_weights[token_indices, choice_indices].unsqueeze(-1)
            flat_output = flat_output.index_add(0, token_indices, expert_output * weights)
        output = flat_output.reshape(*leading_shape, self.d_model)
        if not return_routing:
            return output
        routing = {
            "router_probs": router_probs.reshape(*leading_shape, self.n_experts),
            "topk_indices": topk_indices.reshape(*leading_shape, self.top_k),
            "topk_weights": topk_weights.reshape(*leading_shape, self.top_k),
        }
        return output, routing


def expert_loads(topk_indices: Tensor, n_experts: int) -> Tensor:
    """Count expert assignments; a token routed to top-k contributes k assignments."""
    if topk_indices.numel() == 0:
        raise ValueError("topk_indices must not be empty")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if ((topk_indices < 0) | (topk_indices >= n_experts)).any().item():
        raise ValueError("topk_indices contains an invalid expert ID")
    return torch.bincount(topk_indices.reshape(-1), minlength=n_experts)


def load_balance_loss(router_probs: Tensor, topk_indices: Tensor) -> Tensor:
    """Switch-style teaching proxy: E * sum(f_i * mean_router_probability_i).

    f_i is the fraction of all top-k assignments sent to expert i. Exact
    auxiliary losses and normalization differ by paper and implementation.
    """
    if router_probs.ndim < 2 or topk_indices.ndim != router_probs.ndim:
        raise ValueError("router_probs [..., E] and topk_indices [..., K] must match rank")
    if router_probs.shape[:-1] != topk_indices.shape[:-1]:
        raise ValueError("router_probs and topk_indices leading dimensions must match")
    n_experts = router_probs.shape[-1]
    if n_experts <= 0 or topk_indices.numel() == 0:
        raise ValueError("router probabilities and assignments must be non-empty")
    if ((topk_indices < 0) | (topk_indices >= n_experts)).any().item():
        raise ValueError("topk_indices contains an invalid expert ID")
    flat_indices = topk_indices.reshape(-1, topk_indices.shape[-1])
    assignments = F.one_hot(flat_indices, num_classes=n_experts).float()
    fraction_per_expert = assignments.sum(dim=(0, 1)) / flat_indices.numel()
    mean_probability = router_probs.reshape(-1, n_experts).mean(dim=0)
    return n_experts * torch.sum(fraction_per_expert * mean_probability)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def run_demo() -> None:
    torch.manual_seed(17)
    x = torch.randn(2, 8, 32)
    gqa = GroupedQueryAttention(d_model=32, n_query_heads=4, n_kv_heads=2)
    gqa_out, attention = gqa(x)
    moe = TopKMoE(d_model=32, hidden_dim=64, n_experts=4, top_k=2)
    moe_out, routing = moe(x, return_routing=True)
    aux = load_balance_loss(routing["router_probs"], routing["topk_indices"])
    print(f"GQA output: {tuple(gqa_out.shape)}; attention weights: {tuple(attention.shape)}")
    print(f"MoE output: {tuple(moe_out.shape)}; expert assignments: {expert_loads(routing['topk_indices'], 4).tolist()}")
    print(f"MoE trainable parameters: {count_parameters(moe):,}; load-balance proxy: {aux.item():.4f}")


if __name__ == "__main__":
    run_demo()
