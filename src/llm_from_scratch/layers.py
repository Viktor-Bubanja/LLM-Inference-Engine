import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Divides by the root mean square over the last dimension, then applies a
    learned gain. Unlike LayerNorm it neither centres the input nor adds a shift.

    The statistics are computed in float32 and the result cast back before the
    gain is applied, as in Qwen3 and Llama, so bf16 outputs match theirs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., dim) -> (..., dim)"""
        x_fp32 = x.float()
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x_fp32 * torch.rsqrt(mean_square + self.eps)
        return self.weight * norm_x.to(x.dtype)


class SwiGLU(nn.Module):
    """Gated MLP: ``down(silu(gate(x)) * up(x))``, with no biases."""

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., emb_dim) -> (..., emb_dim)"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
