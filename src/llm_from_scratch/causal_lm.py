"""The interface every model family implements, so generation, steering and
evaluation code works with any of them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import torch

from llm_from_scratch.kv_cache import KVCache


class Role(IntEnum):
    """Who a token belongs to. ``TEMPLATE`` covers chat-template markup, such
    as role headers, and any text outside a chat.
    """

    TEMPLATE = 0
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3


@dataclass(frozen=True)
class TokenContext:
    """Where each token of a forward pass sits in the conversation.

    ``positions`` are absolute token positions, used for RoPE. ``turn`` is the
    1-based index of the assistant turn a token belongs to, or -1 for tokens
    outside assistant turns. ``role`` holds ``Role`` values. All three are
    (b, num_tokens).
    """

    positions: torch.Tensor
    turn: torch.Tensor
    role: torch.Tensor

    @classmethod
    def plain(
        cls,
        batch_size: int,
        num_tokens: int,
        start: int = 0,
        device: torch.device | str | None = None,
    ) -> "TokenContext":
        """Context for text outside a chat, at positions ``start`` onwards."""
        positions = torch.arange(start, start + num_tokens, device=device)
        positions = positions.expand(batch_size, num_tokens)
        return cls(
            positions=positions,
            turn=torch.full_like(positions, -1),
            role=torch.full_like(positions, Role.TEMPLATE),
        )

    def cat(self, other: "TokenContext") -> "TokenContext":
        """This context followed by ``other``, along the token dimension."""
        return TokenContext(
            torch.cat((self.positions, other.positions), dim=1),
            torch.cat((self.turn, other.turn), dim=1),
            torch.cat((self.role, other.role), dim=1),
        )


ResidualHook = Callable[[int, torch.Tensor, TokenContext], torch.Tensor]
"""``(layer, x, ctx) -> x``, called on the residual stream after each block.
``x`` is (b, num_tokens, emb_dim) and ``ctx`` describes the same tokens.

Hooks receive the context explicitly, rather than through PyTorch forward
hooks, so that edits that depend on the turn or position, such as steering
scaled by ``α(turn)``, stay correct under cached decoding, where ``x`` holds
only the newest tokens.
"""


class CausalLM(Protocol):
    residual_hooks: list[ResidualHook]

    def __call__(
        self,
        token_ids: torch.Tensor,
        cache: KVCache | None = None,
        ctx: TokenContext | None = None,
    ) -> torch.Tensor:
        """(b, num_tokens) token ids -> (b, num_tokens, vocab_size) logits.

        ``ctx`` defaults to plain text at positions ``0 .. num_tokens - 1``,
        offset by ``cache.length`` when a cache is given.
        """
        ...

    def new_cache(self, batch_size: int, max_len: int) -> KVCache:
        """An empty KV cache for this model, on its device and in its dtype."""
        ...
