import torch
import torch.nn.functional as F

from llm_from_scratch.kv_cache import KVCache
from llm_from_scratch.layers import RMSNorm
from llm_from_scratch.rope import apply_rotary


def cached_causal_mask(num_new: int, total: int, device=None) -> torch.Tensor:
    """(num_new, total) bool: True where query i may attend to key j."""
    return torch.cat(
        (
            torch.ones((num_new, total - num_new), dtype=torch.bool, device=device),
            torch.tril(torch.ones((num_new, num_new), dtype=torch.bool, device=device)),
        ),
        dim=-1,
    )


class GroupedQueryAttention(torch.nn.Module):
    """Causal self-attention in which each group of ``num_heads // num_kv_heads``
    query heads shares one key/value head. With ``num_kv_heads == num_heads`` it
    is ordinary multi-head attention.

    Query head ``h`` uses key/value head ``h // (num_heads // num_kv_heads)``.
    ``head_dim`` is set independently: ``num_heads * head_dim`` need not equal
    ``d_in``.

    With ``qk_norm``, queries and keys are RMS-normalised per head, over
    ``head_dim``, before RoPE is applied.

    Submodules are named as in the Qwen3 checkpoint, so weights load by name:
    ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj`` (``nn.Linear`` without
    bias), plus ``q_norm`` and ``k_norm`` (``RMSNorm(head_dim)``) when
    ``qk_norm`` is set. Attention itself uses ``F.scaled_dot_product_attention``.
    """

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(d_in, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d_in, num_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_in, num_kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, d_in, bias=False)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, rms_norm_eps) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer: int | None = None,
    ) -> torch.Tensor:
        """(b, num_tokens, d_in) -> (b, num_tokens, d_in)

        ``cos`` and ``sin`` come from ``RotaryEmbedding``, shaped
        (b, num_tokens, head_dim).
        """
        b, num_tokens, _ = x.shape
        q = self.q_proj(x).view(b, num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, num_tokens, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rotary(q.transpose(1, 2), cos, sin)
        k = apply_rotary(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)

        mask = None
        if cache is not None and layer is not None:
            k, v = cache.update(layer, k, v)
            mask = cached_causal_mask(num_tokens, k.shape[2], device=x.device)

        group_size = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

        is_causal = mask is None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, num_tokens, -1))
