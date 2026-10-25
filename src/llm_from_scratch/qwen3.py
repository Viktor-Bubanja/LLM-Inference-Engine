"""Qwen3: a Llama-family decoder with RoPE, RMSNorm, SwiGLU, grouped-query
attention and QK-norm.
"""

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from llm_from_scratch.attention import GroupedQueryAttention
from llm_from_scratch.causal_lm import ResidualHook, TokenContext
from llm_from_scratch.kv_cache import KVCache
from llm_from_scratch.layers import RMSNorm, SwiGLU
from llm_from_scratch.rope import RotaryEmbedding


@dataclass(frozen=True)
class Qwen3Config:
    """Architecture of a Qwen3 model. Read it from a checkpoint's ``config.json``
    with ``from_hf_config`` rather than writing the numbers by hand.
    """

    vocab_size: int
    context_length: int
    emb_dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    tie_embeddings: bool

    @classmethod
    def from_hf_config(cls, hf_config: dict[str, Any]) -> "Qwen3Config":
        if hf_config.get("model_type") != "qwen3":
            raise ValueError(
                f"Expected a qwen3 config, got {hf_config.get('model_type')!r}"
            )
        # Newer transformers versions nest rope_theta under rope_parameters.
        rope_theta = (
            hf_config.get("rope_theta") or hf_config["rope_parameters"]["rope_theta"]
        )
        return cls(
            vocab_size=hf_config["vocab_size"],
            context_length=hf_config["max_position_embeddings"],
            emb_dim=hf_config["hidden_size"],
            hidden_dim=hf_config["intermediate_size"],
            n_layers=hf_config["num_hidden_layers"],
            n_heads=hf_config["num_attention_heads"],
            n_kv_heads=hf_config["num_key_value_heads"],
            head_dim=hf_config["head_dim"],
            rope_theta=rope_theta,
            rms_norm_eps=hf_config["rms_norm_eps"],
            tie_embeddings=hf_config["tie_word_embeddings"],
        )


class Qwen3Block(nn.Module):
    """Pre-norm block: grouped-query attention, then a SwiGLU MLP."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.norm1 = RMSNorm(config.emb_dim, eps=config.rms_norm_eps)
        self.att = GroupedQueryAttention(
            d_in=config.emb_dim,
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.norm2 = RMSNorm(config.emb_dim, eps=config.rms_norm_eps)
        self.ff = SwiGLU(config.emb_dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer: int | None = None,
    ) -> torch.Tensor:
        """(b, num_tokens, emb_dim) -> (b, num_tokens, emb_dim)"""
        x = x + self.att(x=self.norm1(x), cos=cos, sin=sin, cache=cache, layer=layer)
        return x + self.ff(self.norm2(x))


class Qwen3Model(nn.Module):
    """Decoder-only Qwen3: token embeddings, a stack of blocks sharing one RoPE
    table, a final RMSNorm and a vocabulary head, tied to the embeddings when
    the config says so. Implements ``CausalLM``.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.trf_blocks = nn.ModuleList(
            [Qwen3Block(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.emb_dim, eps=config.rms_norm_eps)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.out_head.weight = self.tok_emb.weight
        self.residual_hooks: list[ResidualHook] = []

    def forward(
        self,
        token_ids: torch.Tensor,
        cache: KVCache | None = None,
        ctx: TokenContext | None = None,
    ) -> torch.Tensor:
        """(b, num_tokens) -> (b, num_tokens, vocab_size) logits

        With a ``cache``, ``token_ids`` are the tokens that follow the ones
        already cached: the whole prompt on the first call (prefill), then one
        or more new tokens per call (decode). Their keys and values are added
        to the cache.

        ``ctx`` gives the tokens' positions, which RoPE uses, and their turns
        and roles, which only the residual hooks use. It defaults to plain text
        continuing from ``cache.length``.
        """
        if ctx is None:
            batch_size, num_tokens = token_ids.shape
            start = cache.length if cache is not None else 0
            ctx = TokenContext.plain(batch_size, num_tokens, start, token_ids.device)
        cos, sin = self.rope(ctx.positions)

        x = self.tok_emb(token_ids)
        for layer, block in enumerate(self.trf_blocks):
            x = block(x, cos, sin, cache, layer)
            for hook in self.residual_hooks:
                x = hook(layer, x, ctx)
        x = self.final_norm(x)
        return self.out_head(x)

    def new_cache(self, batch_size: int, max_len: int) -> KVCache:
        """An empty KV cache for this model, on its device and in its dtype."""
        weight = self.tok_emb.weight
        return KVCache(
            n_layers=self.config.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.config.n_kv_heads,
            max_len=max_len,
            head_dim=self.config.head_dim,
            dtype=weight.dtype,
            device=weight.device,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue a prompt with pretrained Qwen3-0.6B-Base."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="The history of the alpaca begins",
        help="Text to continue.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=60, help="Tokens to generate."
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    from llm_from_scratch.device import get_device
    from llm_from_scratch.gpt_model import generate
    from llm_from_scratch.hf_loading import (
        QWEN3_0_6B_BASE,
        load_pretrained,
        load_tokenizer,
    )

    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    model = load_pretrained(QWEN3_0_6B_BASE, device=device)
    tokenizer = load_tokenizer(QWEN3_0_6B_BASE)

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    out = generate(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        context_size=model.config.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=tokenizer.eos_id,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
