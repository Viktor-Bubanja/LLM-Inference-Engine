import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Normalizes over the last dimension, with learned scale and shift."""

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., emb_dim) -> (..., emb_dim)"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    """Position-wise MLP. ``hidden_dim`` defaults to the usual 4x expansion."""

    def __init__(self, emb_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * emb_dim
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., emb_dim) -> (..., emb_dim)"""
        return self.layers(x)


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
