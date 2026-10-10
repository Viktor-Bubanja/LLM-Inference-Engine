"""Rotary position embedding (RoPE)."""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Builds the cos and sin tables that rotate queries and keys by position.

    Frequencies follow the usual RoPE schedule with base ``theta``. Angles are
    computed in float32 whatever the model's dtype: at position 30,000, bf16
    can't represent the angle precisely enough.
    """

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        raise NotImplementedError("M1 exercise")

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(b, num_tokens) integer positions -> (cos, sin), each
        (b, num_tokens, head_dim) in float32.
        """
        raise NotImplementedError("M1 exercise")


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each head of ``x`` by the angles in ``cos`` and ``sin``.

    Uses the "rotate-half" layout of GPT-NeoX, Llama and Qwen: dimension ``i``
    is paired with ``i + head_dim // 2``, not with ``i + 1``. The pretrained
    weights were trained this way, so the other layout gives wrong outputs.

    x: (b, n_heads, num_tokens, head_dim); cos, sin: (b, num_tokens, head_dim)
    -> (b, n_heads, num_tokens, head_dim), in ``x.dtype``
    """
    raise NotImplementedError("M1 exercise")
