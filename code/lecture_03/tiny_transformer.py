"""A tiny decoder-only Transformer for CS336 Lecture 3.

This is an educational reference implementation, not a production LLM.
It intentionally exposes tensor shapes and attention weights so that the
causal mask and the next-token objective can be tested directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 64
    max_seq_len: int = 32
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers <= 0 or self.mlp_ratio <= 0:
            raise ValueError("n_layers and mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (no future-token) mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("attention input must have shape [batch, time, channels]")
        batch_size, seq_len, _ = x.shape

        # [B,T,3D] -> [3,B,H,T,D/H] -> three tensors [B,H,T,D/H].
        qkv = self.qkv(x).view(
            batch_size, seq_len, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # [B,H,T,Dh] @ [B,H,Dh,T] -> [B,H,T,T].
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        future_positions = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future_positions, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        attended = self.attn_dropout(weights) @ v

        # [B,H,T,Dh] -> [B,T,H,Dh] -> [B,T,D].
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.resid_dropout(self.out_proj(attended)), weights


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: causal attention and MLP, each with a residual."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.ln_mlp = nn.LayerNorm(config.d_model)
        hidden_dim = config.mlp_ratio * config.d_model
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn_out, weights = self.attn(self.ln_attn(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_mlp(x))
        return x, weights


class TinyGPT(nn.Module):
    """Small decoder-only language model returning logits of shape [B,T,V]."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self, input_ids: Tensor, *, return_attentions: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("input sequence must not be empty")
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        attentions: list[Tensor] = []
        for block in self.blocks:
            x, weights = block(x)
            if return_attentions:
                attentions.append(weights)
        logits = self.lm_head(self.final_norm(x))
        if return_attentions:
            return logits, attentions
        return logits


def next_token_loss(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Cross-entropy for predicting input_ids[:, 1:] from logits[:, :-1]."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [B,T,V] and input_ids [B,T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids batch/time dimensions must match")
    if input_ids.shape[1] < 2:
        raise ValueError("at least two tokens are required for next-token loss")
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().view(-1),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters (does not count activations or optimizer state)."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_demo() -> None:
    torch.manual_seed(7)
    config = TransformerConfig(vocab_size=32, max_seq_len=12, d_model=24, n_heads=4, n_layers=2)
    model = TinyGPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits, attentions = model(input_ids, return_attentions=True)
    loss = next_token_loss(logits, input_ids)
    loss.backward()

    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"attention weights (layer 0): {tuple(attentions[0].shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")
    print(f"next-token loss: {loss.item():.4f}")
    print(f"all parameter gradients finite: {all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())}")


if __name__ == "__main__":
    run_demo()
