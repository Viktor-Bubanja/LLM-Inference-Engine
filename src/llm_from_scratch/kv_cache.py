"""Key/value cache for incremental decoding."""

import torch


class KVCache:
    """Preallocated keys and values for every layer of a model.

    Each layer gets K and V buffers shaped
    ``(batch_size, n_kv_heads, max_len, head_dim)``. Storing KV heads rather
    than query heads is what makes grouped-query attention save memory.

    Keys are stored after RoPE is applied: a token's position never changes,
    so neither does its rotated key.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        raise NotImplementedError("M2 exercise")

    @property
    def length(self) -> int:
        """How many tokens every layer has stored: the tokens from all completed
        forward passes. The next token fed to the model is at this position.
        """
        raise NotImplementedError("M2 exercise")

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store ``k`` and ``v`` for ``layer`` after the entries already stored
        for it, and return all of that layer's keys and values so far.

        k, v: (batch_size, n_kv_heads, num_new, head_dim)
        -> (keys, values), each (batch_size, n_kv_heads, stored + num_new, head_dim)

        Raises ``ValueError`` if the layer would hold more than ``max_len``
        tokens.
        """
        raise NotImplementedError("M2 exercise")
