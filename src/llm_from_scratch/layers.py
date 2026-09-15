import torch
import torch.nn as nn


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
