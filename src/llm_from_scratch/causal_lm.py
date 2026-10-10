"""The interface every model family implements, so generation, steering and
evaluation code works with any of them.
"""

from collections.abc import Callable
from typing import Protocol

import torch

ResidualHook = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
"""``(layer, x, positions) -> x``, called on the residual stream after each
block. ``x`` is (b, num_tokens, emb_dim) and ``positions`` is (b, num_tokens).

Hooks receive positions explicitly, rather than through PyTorch forward hooks,
so that position-dependent edits such as steering scaled by ``α(t)`` stay
correct under cached decoding, where ``x`` holds only the newest tokens.
"""


class CausalLM(Protocol):
    residual_hooks: list[ResidualHook]

    def __call__(
        self, token_ids: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(b, num_tokens) token ids -> (b, num_tokens, vocab_size) logits.

        ``positions`` defaults to ``0 .. num_tokens - 1`` for every row.
        """
        ...
