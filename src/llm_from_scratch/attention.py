import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.scale = self.head_dim**-0.5

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(b, num_tokens, d_out) -> (b, num_heads, num_tokens, head_dim)"""
        b, num_tokens, _ = x.shape
        return x.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(b, num_tokens, d_in) -> (b, num_tokens, d_out)"""
        b, num_tokens, _ = x.shape
        if num_tokens > self.context_length:
            raise ValueError(
                "x must not contain more tokens than context_length. "
                f"num_tokens: {num_tokens}, context_length: {self.context_length}"
            )

        queries = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))

        attn_scores = queries @ keys.transpose(2, 3) * self.scale
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))

        context_vecs = (attn_weights @ values).transpose(1, 2)
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vecs)


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
        raise NotImplementedError("M1 exercise")

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """(b, num_tokens, d_in) -> (b, num_tokens, d_in)

        ``cos`` and ``sin`` come from ``RotaryEmbedding``, shaped
        (b, num_tokens, head_dim).
        """
        raise NotImplementedError("M1 exercise")
