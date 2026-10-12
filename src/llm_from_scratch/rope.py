"""Rotary position embedding (RoPE)."""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Builds the cos and sin tables that rotate queries and keys by position.

    Pair ``j`` rotates at frequency ``base ** (-2j / head_dim)``. Hugging Face
    configs call ``base`` ``rope_theta``.

    Angles are computed in float32 whatever the model's dtype: at position
    30,000, bf16 can't represent the angle precisely enough.
    """

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(b, num_tokens) integer positions -> (cos, sin), each
        (b, num_tokens, head_dim) in float32.
        """
        # Recomputed on every call rather than stored as a buffer, because
        # model.to(torch.bfloat16) would cast a buffer and lose the precision.
        exponents = (
            torch.arange(
                0, self.head_dim, 2, device=positions.device, dtype=torch.float32
            )
            / self.head_dim
        )
        inv_freq = 1.0 / (self.base**exponents)
        angles = positions.float()[..., None] * inv_freq
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each head of ``x`` by the angles in ``cos`` and ``sin``.

    Uses the "rotate-half" layout of GPT-NeoX, Llama and Qwen: dimension ``i``
    is paired with ``i + head_dim // 2``, not with ``i + 1``. The pretrained
    weights were trained this way, so the other layout gives wrong outputs.

    x: (b, n_heads, num_tokens, head_dim); cos, sin: (b, num_tokens, head_dim)
    -> (b, n_heads, num_tokens, head_dim), in ``x.dtype``
    """
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin
